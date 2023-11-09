import os
 
import numpy as np
import torch
import torch.nn as nn
import torch_neuronx
print(torch.__version__)
import accelerate
print(accelerate.__version__)

import diffusers
print(diffusers.__version__)

from diffusers import DiffusionPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
 
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import time
import copy
from IPython.display import clear_output

clear_output(wait=False)

def get_attention_scores_neuron(self, query, key, attn_mask):    
    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0,2,1)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2),
            self.scale
        )
        attention_probs = attention_scores.softmax(dim=-1)
  
    return attention_probs
 

def custom_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled
 

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

    
COMPILER_WORKDIR_ROOT = 'sdxl_base_and_refiner_compile_dir_1024'

# Model ID for SD XL version pipeline
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"



# --- Compile UNet (base) and save ---
pipe_base = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)

# Replace original cross-attention module with custom cross-attention module for better performance
Attention.get_attention_scores = get_attention_scores_neuron

# Apply double wrapper to deal with custom return type
pipe_base.unet = NeuronUNet(UNetWrap(pipe_base.unet))

# Only keep the model being compiled in RAM to minimze memory pressure
unet_base = copy.deepcopy(pipe_base.unet.unetwrap)
del pipe_base

# Compile base unet - fp32
sample_1b = torch.randn([1, 4, 128, 128])
timestep_1b = torch.tensor(999).float().expand((1,))
encoder_hidden_states_1b = torch.randn([1, 77, 2048])
added_cond_kwargs_1b = {"text_embeds": torch.randn([1, 1280]),
                        "time_ids": torch.randn([1, 6])}
example_inputs = (sample_1b, timestep_1b, encoder_hidden_states_1b, added_cond_kwargs_1b["text_embeds"], added_cond_kwargs_1b["time_ids"],)

unet_base_neuron = torch_neuronx.trace(
    unet_base,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet_base'),
    compiler_args=["--model-type=unet-inference"]
)

# save compiled unet
unet_base_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet_base/model.pt')
torch.jit.save(unet_base_neuron, unet_base_filename)

# delete unused objects
del unet_base
del unet_base_neuron



# --- Compile UNet (refiner) and save ---

pipe_refiner = DiffusionPipeline.from_pretrained(refiner_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)

# Apply double wrapper to deal with custom return type
pipe_refiner.unet = NeuronUNet(UNetWrap(pipe_refiner.unet))

# Only keep the model being compiled in RAM to minimze memory pressure
unet_refiner = copy.deepcopy(pipe_refiner.unet.unetwrap)
del pipe_refiner

# Compile refiner unet - fp32 - some shapes are different than the base model
encoder_hidden_states_refiner_1b = torch.randn([1, 77, 1280])
added_cond_kwargs_refiner_1b = {"text_embeds": torch.randn([1, 1280]),
                                "time_ids": torch.randn([1, 5])}
example_inputs = (sample_1b, timestep_1b, encoder_hidden_states_refiner_1b, added_cond_kwargs_refiner_1b["text_embeds"], added_cond_kwargs_refiner_1b["time_ids"],)

unet_refiner_neuron = torch_neuronx.trace(
    unet_refiner,
    example_inputs,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet_refiner'),
    compiler_args=["--model-type=unet-inference"]
)

# save compiled unet
unet_refiner_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet_refiner/model.pt')
torch.jit.save(unet_refiner_neuron, unet_refiner_filename)

# delete unused objects
del unet_refiner
del unet_refiner_neuron



# --- Compile VAE decoder and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
decoder = copy.deepcopy(pipe.vae.decoder)
del pipe

# # Compile vae decoder
decoder_in = torch.randn([1, 4, 128, 128])
decoder_neuron = torch_neuronx.trace(
    decoder, 
    decoder_in, 
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
)

# Save the compiled vae decoder
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
torch.jit.save(decoder_neuron, decoder_filename)

# delete unused objects
del decoder
del decoder_neuron


# --- Compile VAE post_quant_conv and save ---

# Only keep the model being compiled in RAM to minimze memory pressure
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
del pipe

# Compile vae post_quant_conv
post_quant_conv_in = torch.randn([1, 4, 128, 128])
post_quant_conv_neuron = torch_neuronx.trace(
    post_quant_conv, 
    post_quant_conv_in,
    compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
)

# Save the compiled vae post_quant_conv
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

# delete unused objects
del post_quant_conv
del post_quant_conv_neuron


