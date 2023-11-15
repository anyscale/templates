from io import BytesIO
from ray import serve
from fastapi import FastAPI
from fastapi.responses import Response
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
import os
 
import numpy as np
import torch
import torch.nn as nn
import torch_neuronx
from diffusers import DiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
 
import time
import copy
from IPython.display import clear_output
import ray 

app = FastAPI()


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
 
    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out_tuple = self.unet(sample,
                              timestep,
                              encoder_hidden_states,
                              added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                              return_dict=False)
        return out_tuple
    
    
class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device
 
    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        sample = self.unetwrap(sample,
                               timestep.float().expand((sample.shape[0],)),
                               encoder_hidden_states,
                               added_cond_kwargs["text_embeds"],
                               added_cond_kwargs["time_ids"])[0]
        return UNet2DConditionOutput(sample=sample)

 
@serve.deployment(ray_actor_options={"resources": {"neuron_cores": 2}})
@serve.ingress(app)
class StableDiffusion:
    def __init__(self):
        COMPILER_WORKDIR_ROOT = '/home/ray/default/sdxl_base_and_refiner_compile_dir_1024'
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

        unet_base_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet_base/model.pt')
        unet_refiner_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet_refiner/model.pt')
        decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
        post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

        self.pipe_base = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)

        # Load the compiled UNet (base) onto two neuron cores.
        self.pipe_base.unet = NeuronUNet(UNetWrap(self.pipe_base.unet))
        ids = ray.get_runtime_context().get_resource_ids()
        device_ids = [int(x) for x in ids["neuron_cores"]] 

        self.pipe_base.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_base_filename), device_ids, set_dynamic_batching=False)

        # Load other compiled models onto a single neuron core.
        self.pipe_base.vae.decoder = torch.jit.load(decoder_filename)
        self.pipe_base.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)

        # Load the compiled UNet (refiner) onto two neuron cores.
        self.pipe_refiner = DiffusionPipeline.from_pretrained(
            refiner_model_id,
            text_encoder_2=self.pipe_base.text_encoder_2,
            vae=self.pipe_base.vae,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        self.pipe_refiner.unet = NeuronUNet(UNetWrap(self.pipe_refiner.unet))
        self.pipe_refiner.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_refiner_filename), device_ids, set_dynamic_batching=False)

    @app.post("/generate")
    def generate(self, prompt):

        # Define how many steps and what % of steps to be run on each experts (80/20) here
        n_steps = 40
        high_noise_frac = 0.8
        start_time = time.time()
        image = self.run_refiner_and_base(self.pipe_base, self.pipe_refiner, prompt, n_steps, high_noise_frac)
        total_time = (time.time()-start_time)
        print("inference time: ", total_time, "seconds")
        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        return Response(content=file_stream.getvalue(), media_type="image/png")
    
    def run_refiner_and_base(self, base, refiner, prompt, n_steps=40, high_noise_frac=0.8):
        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        return image
    
sd_app = StableDiffusion.bind()



    
