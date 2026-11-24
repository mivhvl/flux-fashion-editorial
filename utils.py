import gc
import os
import torch
from PIL import Image

QWEN_ID = "Qwen/Qwen3-VL-4B-Instruct"
KLEIN_ID = "OzzyGT/Flux_3_Image_Klein_sdnq_dynamic_4bit"
REF_SIZE = 512 # reference size for Flux Klein
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/kaggle/working")

def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def load_images(product_path, mood_paths, size):
    paths = [product_path] + list(mood_paths or [])
    ims = [Image.open(p).convert("RGB") for p in paths]
    for im in ims:
        im.thumbnail((size, size))
    return ims