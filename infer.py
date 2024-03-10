from scm.pipelines.scm_pipeline import SCMPipeline
import torch
import os
from scm.util import concat_images_for_row,concat_images_for_col,get_images_ls
from scm.ptp_utils import AttnStore,register_attention_control
import json
import time
device_name="cuda"

def get_subject_index_ls(prompts,tokenizer,subject_name):
    indexes_ls=[]
    subject_token = [tokenizer.decode(t)
                         for idx, t in enumerate(tokenizer(subject_name)['input_ids'])
                         if 0 < idx < len(tokenizer(subject_name)['input_ids']) - 1]
    for prompt in prompts:
        indexes=[]
        token_idx_to_word = {idx: tokenizer.decode(t)
                         for idx, t in enumerate(tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(tokenizer(prompt)['input_ids']) - 1}
        
        for token in subject_token:
            for idx,word in token_idx_to_word.items():
                if word==token:
                    indexes.append(idx)
        indexes_ls.append(indexes)
    return indexes_ls

if __name__ == "__main__":
    pretrained_model_path="runwayml/stable-diffusion-v1-5"
    eval_data_path="./eval_datasets"
    subject_dict=json.load(open(os.path.join(eval_data_path,'subject.json')))

    with os.scandir(eval_data_path) as entries:
        for entry in entries:
            if entry.is_dir():
                story_book_name="6"
                print(f"{story_book_name} is infering...")
                subject_name=subject_dict[story_book_name]
                with open(os.path.join(eval_data_path,story_book_name,'test.txt')) as f:
                    prompts=f.readlines()
                prompts_ls=[prompt.rstrip() for prompt in prompts]                
                output_base_path="./outputs"
                my_model_path=os.path.join(output_base_path,story_book_name)
                pipe = SCMPipeline.from_pretrained(my_model_path, torch_dtype=torch.float16).to(device_name)
                generator = torch.Generator("cuda")
                attn_store=AttnStore()
                register_attention_control(pipe.unet, attn_store)
                for p in prompts_ls:
                    print(p)   
                
                subject_index_ls=get_subject_index_ls(prompts_ls,pipe.tokenizer,subject_name)
                print(subject_index_ls)
                attn_store.set_subject_index(subject_index_ls) 

                save_batch=1
                col_ls=[]
                for i in range(save_batch):
                    generator.manual_seed((i+1)*20)
                    story_length=len(prompts_ls)  

                    attn_store.set_story_length(story_length)
                    start_time=time.time()
                    story = pipe([prompts_ls], latents=None, story_length=story_length, height=512, width=512, num_inference_steps=50, guidance_scale=10.5,generator=generator).story
                    
                    image_ls=get_images_ls(story)
                    
                    imgae_col=concat_images_for_row([img.resize((256,256)) for img in image_ls[0]])
                    col_ls.append(imgae_col)

                infer_save_path="eval_samples"
                os.makedirs(infer_save_path,exist_ok=True)
                concat_images_for_col(col_ls).save(f"{infer_save_path}/{story_book_name}.png")
    