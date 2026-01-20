import ray.data
import pandas as pd
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline


#model_id = 'nitrosocke/Ghibli-Diffusion'
#model can be saved under /mnt/shared_storage for re-use
#model_id = '/mnt/shared_storage/kyle/huggingface/diffusers/models--nitrosocke--Ghibli-Diffusion/snapshots/441c2fde3e82de3720b1a35111b95b6d8afee9d6/'
device = "cuda"
class PredictCallable:
    def __init__(self, model_id: str, prompt: str, strength: float):
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16).to(device)
        self.prompt = prompt
        self.strengh = strength
    
    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        batch_paths = list(batch['path'])
        
        # Torch Tensor approach
        batch_images = torch.tensor(batch['image'])
        inp_images = torch.permute(batch_images,(0,3,1,2))
        
        # PIL.Image approach
        #batch_images = list(batch['image'])
        #inp_images=[Image.fromarray(img) for img in batch_images]
        
        #return out_images
        generator = [torch.Generator(device=device).manual_seed(i) for i in range(len(batch_images))]
        out_images = self.pipe(prompt=[self.prompt]*len(batch_images), image=inp_images, \
                              strength=self.strengh, guidance_scale=7.5, generator=generator)
        return [dict(zip(batch_paths, out_images.images))]
    
'''
img_path = './car_images'
inp_images = ray.data.read_images(img_path, size=(768, 768), include_paths=True).limit(10)
print(inp_images)
preds = (
    inp_images
    .map_batches(
        PredictCallable,
        batch_size=2,
        fn_constructor_kwargs=dict(model_id=model_id, prompt='cyberpunk style cars', strength=0.75),
        compute="actors",
        num_gpus=1
    )
)
res = preds.take_all()
'''