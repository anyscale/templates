import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image, ImageFilter
from torchvision import transforms as T
import ray

#
# borrowed URLs ideas and heavily modified from https://analyticsindiamag.com/how-to-run-python-code-concurrently-using-multithreading/
#

URLS = [
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/305821.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/509922.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/325812.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1252814.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1420709.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/963486.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1557183.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3023211.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1031641.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/439227.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/696644.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/911254.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1001990.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3518623.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/916044.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/2253879.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3316918.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/942317.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1090638.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1279813.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/434645.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1571460.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1080696.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/271816.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/421927.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/302428.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/443383.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3685175.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/2885578.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3530116.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/9668911.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/14704971.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/13865510.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/6607387.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/13716813.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/14690500.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/14690501.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/14615366.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/14344696.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/14661919.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/5977791.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/5211747.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/5995657.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/8574183.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/14690503.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/2100941.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/112460.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/116675.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3586966.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/313782.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/370717.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1323550.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/11374974.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/408951.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3889870.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1774389.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3889854.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/2196578.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/2885320.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/7189303.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/9697598.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/6431298.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/7131157.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/4840134.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/5359974.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3889854.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1753272.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/2328863.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/6102161.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/6101986.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3334492.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/5708915.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/5708913.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/6102436.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/6102144.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/6102003.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/6194087.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/5847900.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1671479.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3335507.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/6102522.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/6211095.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/720347.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3516015.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3325717.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/849835.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/302743.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/167699.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/259620.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/300857.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/789380.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/735987.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/572897.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/300857.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/760971.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/789382.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/1004665.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/facilities.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/3984080835_71b0426844_b.jpg",
    "https://anyscale-public-materials.s3.us-west-2.amazonaws.com/ray-summit/ray-core/33041.jpg"
]

THUMB_SIZE = (64, 64)


def extract_times(lst: Tuple[int, float]) -> List[float]:
    """
    Given a list of Tuples[batch_size, execution_time] extract the latter
    """
    times = [t[1] for t in lst]
    return times


def plot_times(batches: List[int], s_lst: List[float], d_lst: List[float]) -> None:
    """
    Plot the execution times for serail vs distributed for each respective batch size of images
    """
    s_times = extract_times(s_lst)
    d_times = extract_times(d_lst)
    data = {"batches": batches, "serial": s_times, "distributed": d_times}

    df = pd.DataFrame(data)
    df.plot(x="batches", y=["serial", "distributed"], kind="bar")
    plt.ylabel("Times in sec", fontsize=12)
    plt.xlabel("Number of Batches of Images", fontsize=12)
    plt.grid(False)
    plt.show()


def display_random_images(image_list: List[str], n: int = 3) -> None:
    """
    Display a grid of images, default 3 of images we want to process
    """
    random_samples_idx = random.sample(range(len(image_list)), k=n)
    plt.figure(figsize=(16, 8))
    for i, targ_sample in enumerate(random_samples_idx):
        plt.subplot(1, n, i + 1)
        img = Image.open(image_list[targ_sample])
        img_as_array = np.asarray(img)
        plt.imshow(img_as_array)
        title = f"\nshape: {img.size}"
        plt.axis("off")
        plt.title(title)
    plt.show()


def download_images(url: str, data_dir: str) -> None:
    """
    Given a URL and the image data directory, fetch the URL and save it in the data directory
    """
    img_data = requests.get(url).content
    img_name = url.split("/")[5]
    img_name = f"{data_dir}/{img_name}"
    with open(img_name, "wb+") as f:
        f.write(img_data)
        
def insert_into_object_store(img_name:str):
    """
    Insert the image into the object store and return its object reference
    """
    import ray
    
    img = Image.open(img_name)
    img_ref = ray.put(img)
    return img_ref


def transform_image(img_ref:object, fetch_image=True, verbose=False):
    """
    This is a deliberate compute intensive image transfromation and tensor operation
    to simulate a compute intensive image processing
    """
    import ray
    
    # Only fetch the image from the object store if called serially.
    if fetch_image:
        img = ray.get(img_ref)
    else:
        img = img_ref
    before_shape = img.size

    # Make the image blur with specified intensify
    # Use torchvision transformation to augment the image
    img = img.filter(ImageFilter.GaussianBlur(radius=20))
    augmentor = T.TrivialAugmentWide(num_magnitude_bins=31)
    img = augmentor(img)

    # Convert image to tensor and transpose
    tensor = torch.tensor(np.asarray(img))
    t_tensor = torch.transpose(tensor, 0, 1)

    # compute intensive operations on tensors
    random.seed(42)
    for _ in range(3):
        tensor.pow(3).sum()
        t_tensor.pow(3).sum()
        torch.mul(tensor, random.randint(2, 10))
        torch.mul(t_tensor, random.randint(2, 10))
        torch.mul(tensor, tensor)
        torch.mul(t_tensor, t_tensor)

    # Resize to a thumbnail
    img.thumbnail(THUMB_SIZE)
    after_shape = img.size
    if verbose:
        print(f"augmented: shape:{img.size}| image tensor shape:{tensor.size()} transpose shape:{t_tensor.size()}")

    return before_shape, after_shape
