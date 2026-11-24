import os
import time
import torch
import gradio as gr
import sdnq
from diffusers import ModularPipeline

from utils import KLEIN_ID, REF_SIZE, OUTPUT_DIR, free_gpu, load_images

# Size options (width, height) 
SIZES = {
    "2:3": (512, 768),
    "3:4": (576, 768),
    "4:5": (640, 800),
    "1:1": (640, 640),
    "4:3": (768, 576),
    "16:9": (768, 448),
}

# Step 2: Klein Image Generator
def generate(product_path, mood_paths, caption, use_mood_ref, seed, aspect):
    width, height = SIZES[aspect]
    if not product_path:
        raise gr.Error("Upload a product image first.")
    if not caption or not caption.strip():
        raise gr.Error("The caption is empty. Write or generate a caption first.")
    
    mood_refs = list(mood_paths or [])[:3] if use_mood_ref else []
    refs = load_images(product_path, mood_refs, REF_SIZE)

    blocks = ModularPipeline.from_pretrained(KLEIN_ID).blocks
    blocks.sub_blocks.pop("prompt_upsampler")
    pipe = blocks.init_pipeline(KLEIN_ID)
    pipe.load_components(dtype=torch.bfloat16)
    pipe.to("cuda")

    try:
        free_gpu()
        img = pipe(prompt=caption, image=refs, height=int(height), width=int(width),
                   generator=torch.Generator().manual_seed(int(seed)), output="images")[0]
        img.save(f"/kaggle/working/campaign_seed{int(seed)}.png")
    except Exception as e:
        if "out of memory" in str(e).lower():
            raise gr.Error("Out of GPU memory.")
        raise
    finally:
        del pipe, blocks
        free_gpu()
    return img