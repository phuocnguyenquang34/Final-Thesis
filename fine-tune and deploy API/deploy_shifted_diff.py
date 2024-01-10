from pathlib import Path
from time import time
import datetime
import random
from tqdm import tqdm, trange
import numpy as np
import os
import sys
import copy
import torch
import clip

from model_lib.decoder.clip_prior import ClipPrior, Vocab
from model_lib.diffusion.script_util import create_sft_gaussian_diffusion as create_gaussian_diffusion_p2
from diffusers import StableDiffusionPipelineWithCLIP, EulerDiscreteScheduler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from transformers import AutoTokenizer, T5EncoderModel, T5Config
from RealESRGAN import RealESRGAN

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio

import cv2
from typing import Optional
import io
from starlette.responses import StreamingResponse

import warnings
import PIL
warnings.filterwarnings("ignore")

from os import chdir as cd

cd('/kaggle/working/Shifted_Diffusion')

app = FastAPI()
class MultiCLIP(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        model_32, preprocess = clip.load("/kaggle/input/datafortrainingshifted/ViT-B-32.pt", device=device)
        model_16, _ = clip.load("/kaggle/input/datafortrainingshifted/ViT-B-16.pt", device=device)
        model_101, _ = clip.load("/kaggle/input/datafortrainingshifted/RN101.pt", device=device)
        self.model_32 = model_32
        self.model_16 = model_16
        self.model_101 = model_101
        self.preprocess = preprocess

    def encode_image(self, image):
        with torch.no_grad():
            # image = self.preprocess(image)
            vectors = [self.model_16.encode_image(image), self.model_32.encode_image(image), self.model_101.encode_image(image)]
            return torch.cat(vectors, dim=-1)

    def encode_text(self, text, dtype, device):
        with torch.no_grad():
            text = clip.tokenize(text).to(device)
            vectors = [self.model_16.encode_text(text), self.model_32.encode_text(text), self.model_101.encode_text(text)]
            return torch.cat(vectors, dim=-1).to(dtype)

def convert_weights(model: torch.nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, torch.nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def gen_clip_prior(model, diffusion, clip_emb, t5_emb, device="cuda"):
    B, C = clip_emb.shape[:2]
    uncond_clip_emb = clip_emb#torch.zeros_like(clip_emb)
    uncond_t5_emb = torch.zeros_like(t5_emb)
    model_kwargs = dict(
        clip_sentence_emb = torch.cat((clip_emb, uncond_clip_emb), dim=0),
        t5_word_emb = torch.cat((t5_emb, uncond_t5_emb), dim=0),
        emb_4_vocab=torch.cat((clip_emb, clip_emb), dim=0)
    )

    def cfg_sampling(x_t, ts, guidance_scale=1.0, **kwargs):
        # for sampling
        half = x_t[: len(x_t) // 2] # x_t: torch.Size([bx2, 3, 64, 64])
        combined = torch.cat([half, half], dim=0) # combined: torch.Size([bx2, 3, 64, 64])
        model_out = model(combined, ts, **kwargs)  # model_out: torch.Size([bx2, 6, 64, 64])
        eps, rest = torch.split(model_out, model_out.shape[1] //2, dim=1) # mean & variance
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)  # eps torch.Size([bx2, 3, 64, 64])
        return torch.cat([eps, rest], dim=1)  # torch.Size([bx2, 6, 64, 64])

    gen_im_emb = diffusion.p_sample_loop(cfg_sampling, (B *2, C), device=device, clip_denoised=False,
                                         progress=True, model_kwargs=model_kwargs, cond_fn=None)[ :B ]
    return gen_im_emb

prior_path = '/kaggle/input/datafortrainingshifted/prior.pt'
model_path = "/kaggle/input/datafortrainingshifted/finetuned_coco/finetuned_coco"
std_path = '/kaggle/input/datafortrainingshifted/mean.pth'
mean_path = '/kaggle/input/datafortrainingshifted/std.pth'
t5_device = "cuda"  #use cpu to load model if your GPU memory is limited
t5_model = 'google/flan-t5-large'
scale = 5  
layers = 8  
log_std_init = torch.log(torch.load(std_path, map_location='cpu').view((-1, 1536)))[:1024].cuda()
mean_init = torch.load(mean_path, map_location='cpu').view((-1, 1536))[:1024].cuda()

#Load models to memory
with torch.no_grad():
    clip_model = MultiCLIP()
    convert_weights(clip_model)
    t5_encoder = T5EncoderModel.from_pretrained(t5_model, low_cpu_mem_usage=True,
                                                torch_dtype=torch.float16, use_auth_token=True).to(t5_device)
    tokenizer = AutoTokenizer.from_pretrained(t5_model, model_max_length=80, use_fast=False)
    model = ClipPrior(xf_width=2048, xf_layers=layers, xf_heads=32,
                      clip_width=512*3, learn_sigma=False, use_vocab=True, vocab_size=1024,
                      vocab_use_mean=True, vocab_sample_num=1, t5_dim=t5_encoder.config.d_model,
                      vocab_log_std_init=log_std_init, vocab_mean_init=mean_init, vocab_learnable=False,
                      vocab_std_scale=scale, vocab_exp=False)
    ckpt = torch.load(prior_path)
    model.load_state_dict(ckpt)
    diffusion_fast = create_gaussian_diffusion_p2(
        steps=1000,
        learn_sigma=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=True,
        predict_prev=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="8",
        p2_gamma=1,
        p2_k=1,
        vocab=model.vocab,
        beta_min=0.0001,
        beta_max=0.02,
    )


    model.to(t5_device)  #move model to save memory if you have limited GPU memory

    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    pipe = StableDiffusionPipelineWithCLIP.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16, device='auto')
    pipe.to("cuda")
    resr_model = RealESRGAN('cpu', scale=4)
    resr_model.load_weights('/kaggle/input/datafortrainingshifted/RealESRGAN_x4.pth')

#Function to upscale image to x4 its resolution
def upscale_im(im):
    im = resr_model.predict(im)
    return im

#Function to generate image
def gen_img(prompt='', negative_prompt='', height=512, width=512, guidance_rescale=2.0, num_inference_steps=50):
    cd('/kaggle/working/Shifted_Diffusion')
    clip_model.cuda()
    t5_encoder.cuda()
    model.cuda()
    #Input Prompt
    text = prompt
    t5_ids = tokenizer(
        text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to(t5_encoder.device)
    clip_text_emb = clip_model.encode_text(text, dtype=torch.float16, device="cuda")
    input_t5_emb = t5_encoder(input_ids=t5_ids).last_hidden_state.cuda()
    gen_emb = gen_clip_prior(model, diffusion_fast, clip_text_emb.float(), input_t5_emb.float())
  
    #Input Negative Prompt
    neg_prompt_embeds = None
    if negative_prompt is None:
        negative_prompt = None
        neg_prompt_embeds = torch.load("./small_sft_empty_embeds.th").cuda().to(dtype=torch.float16)
      
    #Generate image
    sum_pixels = height + width
    #Convert image to around 512x521 to assure picture's quality
    if sum_pixels != 1024:
        new_height = int((1024/(height+width)*height)/8)*8
        new_width = int((1024/(height+width)*width)/8)*8
        print(new_height, new_width)
        image = pipe(prompt=None, guidance_scale=guidance_rescale, num_inference_steps=num_inference_steps,
                prompt_embeds=gen_emb.to(dtype=torch.float16), negative_prompt=negative_prompt,
                negative_prompt_embeds=neg_prompt_embeds, img_emb=True, height=new_height, width=new_width).images[0]
        #Transform back to its original resolution
        transform = Resize((height,width), interpolation=BICUBIC)
        if sum_pixels > 1024:
            image = transform(upscale_im(image))
        else: image = transform(image)

    else:
        image = pipe(prompt=None, guidance_scale=guidance_rescale, num_inference_steps=num_inference_steps,
                    prompt_embeds=gen_emb.to(dtype=torch.float16), negative_prompt=negative_prompt,
                    negative_prompt_embeds=neg_prompt_embeds, img_emb=True, height=height, width=width).images[0]
      #Ensure height and width are divisible to 8
#     height = int(height/8)*8
#     width = int(width/8)*8
#     image = pipe(prompt=None, guidance_scale=guidance_rescale, num_inference_steps=num_inference_steps,
#               prompt_embeds=gen_emb.to(dtype=torch.float16), negative_prompt=negative_prompt,
#               negative_prompt_embeds=neg_prompt_embeds, img_emb=True, height=height, width=width).images[0]
    model.to(t5_device)
    return image

file_names = []

def save_im(im):
    imgio = io.BytesIO()
    im.save(imgio, 'PNG')
    imgio.seek(0)

    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".png", delete=False) as FOUT:
        FOUT.write(imgio.read())
        file_names.append(FOUT.name)
        return file_names[-1]

  origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):

    prompt : str
    negative_prompt : Optional[str] = ''
    height : Optional[int] = 512
    width : Optional[int] = 512
    guidance_rescale : float = 2.0
    num_inference_steps : int = 50

@app.post('/generate_image')
def generate_image(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    prompt = input_dictionary['prompt']
    negative_prompt = input_dictionary['negative_prompt']
    height = input_dictionary['height']
    width = input_dictionary['width']
    guidance_rescale = input_dictionary['guidance_rescale']
    num_inference_steps = input_dictionary['num_inference_steps']

    image = gen_img(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width, guidance_rescale=guidance_rescale, num_inference_steps=num_inference_steps)
    return save_im(image)

@app.get("/get_image")
async def get_image():
    image_path = Path(file_names[-1])
    if not image_path.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(image_path)

#API
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
