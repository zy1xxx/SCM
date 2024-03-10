import os
import imageio
import numpy as np
from typing import Union

import torch

from tqdm import tqdm
from einops import rearrange
from PIL import Image

def get_images_ls(story: torch.Tensor):
    story = rearrange(story, "b c t h w -> b t c h w")
    outputs = []
    for b in story:
        single_video=[]
        for x in b:
            x = (x * 255).numpy().astype(np.uint8)
            x=x.transpose(1,2,0)
            single_video.append(Image.fromarray(x))
        outputs.append(single_video)
    return outputs
def concat_images_for_row(images):
    width=images[0].width
    height=images[0].height*len(images)
    new_img = Image.new('RGB', (width, height))
    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.height
    return new_img
def concat_images_for_col(images):
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    new_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    return new_img