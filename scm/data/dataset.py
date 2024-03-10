import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange
import os
from PIL import Image
import torch
import numpy as np

class StoryDataset(Dataset):
    def __init__(
            self,
            story_path: str,
            width: int = 512,
            height: int = 512,
    ):
        self.story_path = story_path
        with open(os.path.join(self.story_path,'caption.txt')) as f:
            prompts=f.readlines()
        self.prompts=[prompt.rstrip() for prompt in prompts]
        self.story_len=len(self.prompts)

        ##story images tensor
        torch_ls=[]
        for i in range(self.story_len):
            image_path=os.path.join(self.story_path,f"{i+1}.png")
            image_pil=Image.open(image_path).resize((width,height)).convert("RGB")
            torch_image=torch.from_numpy(np.array(image_pil))
            torch_ls.append(torch_image)
        story=torch.stack(torch_ls)
        story = rearrange(story, "f h w c -> f c h w")
        self.story=story / 127.5 - 1.0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        example = {
            "pixel_values": self.story,
            "prompts": self.prompts
            }
        return example
