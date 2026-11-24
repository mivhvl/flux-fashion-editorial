import torch
import gradio as gr
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from utils import QWEN_ID, free_gpu, load_images

SYSTEM = """You write captions for a FLUX 3 Imag Klein generator.
The FIRST image is the PRODUCT: it must appear clearly, with its exact shape, color, material and hardware.
The remaining images, if exist, are a MOOD BOARD: describe all you can see, borrowing their setting, lighting, palette, camera feel, composition, mood and attitude.
But EXPLICIT CONTROLS from the user ALWAYS override the mood board.
Write ONE full scene caption in this order: medium and style, scene, lighting and camera, each subject with colors, materials and position in the frame.
Make it a high-fashion editorial realistic campaign, not a catalog photo. No bullet points, no preamble. Output only the caption."""


FRAMING = {
    "full-body": "Mandatory: full-length vertical shot, where the entire body is visible from head to toe including the shoes, the model fills about two thirds of the frame height, with clear space above her head and below her feet.",
    "three-quarter length": "Mandatory: three-quarter length shot, where the frame cuts mid-thigh, the head, torso and hips are fully visible, with space above the head.",
    "waist-up": "Mandatory: waist-up shot, where the frame cuts at the waist, with the head, shoulders, arms and torso fully visible.",
    "close-up detail": "Mandatory: close-up detail shot, where the frame is tightly cropped on the product, which fills most of the image, with its texture, stitching, material and hardware in sharp focus; only a small part of the model is visible and the full body is not shown.",
    "close-up face": "Mandatory: close-up portrait, where the face fills most of the frame, cropped at the shoulders, eyes in sharp focus.",
}

# Qwen writes the caption proposition
def write_caption(product_path, mood_paths, request, framing, lighting, camera, setting, palette):
    if not product_path:
        raise gr.Error("Upload a product image first.")
    images = load_images(product_path, mood_paths, 1024)
    
    framing = FRAMING.get(framing, framing)
    controls = {"framing": framing, "lighting": lighting, "camera": camera,
                "setting": setting, "palette": palette}
    controls_text = "\n".join(f"- {k}: {v}" for k, v in controls.items() if v and v.strip()) or "None."
    text = (f"Reference images follow in order: first is the PRODUCT; remaining images are MOOD BOARD inspiration.\n"
            f"User request: {request}\n"
            f"Use the mood board images only for broad visual direction.\n"
            f"Explicit controls (must override the mood board characteristics):\n{controls_text}\n"
            f"Output the caption.")

    model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN_ID, dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(QWEN_ID)
    try:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": im} for im in images]
                                        + [{"type": "text", "text": text}]},
        ]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=450, do_sample=True, temperature=0.7)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        caption = processor.batch_decode(trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0].strip()
    finally:
        del model, processor
        free_gpu()
    return caption