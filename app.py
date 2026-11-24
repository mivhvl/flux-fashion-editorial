import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr

from captioner import write_caption
from generator import generate
from upscaler import upscale_image, finish_image


def generate_and_store(*args):
    img = generate(*args)
    return img, img

css = ".gradio-container {max-width: 680px !important; margin: 0 auto !important;}"

with gr.Blocks(title="Editorial campaign generator") as demo:
    gr.Markdown("### Prompt Generation")
    with gr.Row():
        product = gr.Image(type="filepath", label="Product photo reference", height=260)
        moods = gr.File(file_count="multiple", type="filepath", label="Mood board", height=260)
    request = gr.Textbox(label="Request for Qwen (optional)", value="Editorial realistic campaign shot, model wearing this product.")
    framing = gr.Dropdown(["", "full-body", "three-quarter length", "waist-up", "close-up detail", "close-up face"], value="full-body", label="Framing")
    with gr.Row():
        lighting = gr.Textbox(label="Lighting (optional)", placeholder="hard direct flash")
        camera = gr.Textbox(label="Camera (optional)", placeholder="35mm film, low angle")
    with gr.Row():
        setting = gr.Textbox(label="Setting (optional)", placeholder="studio gray background")
        palette = gr.Textbox(label="Palette (optional)", placeholder="greens")
    btn_caption = gr.Button("GENERATE PROMPT WITH QWEN", variant="secondary")

    gr.Markdown("### Image Generation")
    caption_box = gr.Textbox(label="Prompt", placeholder="Write your image prompt here, or use the optional Qwen button above.", lines=10)
    with gr.Row():
        aspect = gr.Dropdown(["1:1", "2:3", "3:4", "4:3", "4:5", "16:9"], value="1:1", label="Aspect ratio")
        seed = gr.Slider(1, 9999, value=1, step=1, label="Seed")
    use_mood_ref = gr.Checkbox(label="Pass up to moodboard", value=False)
    btn_gen = gr.Button("GENERATE IMAGE", variant="primary")
    result = gr.Image(label="Result", type="pil", height=400)
    current_img = gr.State(None)

    gr.Markdown("### FINISH AND UPSCALE")
    with gr.Row():
        upscale_factor = gr.Radio(["2x", "4x"], value="2x", label="Upscale factor")
    btn_up = gr.Button("UPSCALE IMAGE", variant="primary")
    upscaled = gr.Image(label="Finished image", type="pil", height=400)
    upscaled_base = gr.State(None)
    with gr.Row():
        finish_contrast = gr.Slider(0.85, 1.20, value=1.03, step=0.01, label="Contrast")
        finish_sharpen = gr.Slider(0.0, 0.8, value=0.15, step=0.05, label="Sharpen")
    with gr.Row():
        finish_warmth = gr.Slider(-15, 15, value=0, step=1, label="Color warmth")
        finish_grain = gr.Slider(0, 12, value=2, step=0.5, label="Film grain")

    btn_caption.click(write_caption, [product, moods, request, framing, lighting, camera, setting, palette], caption_box)
    btn_gen.click(generate_and_store, [product, moods, caption_box, use_mood_ref, seed, aspect], [result, current_img]).then(
        lambda: (None, None), outputs=[upscaled_base, upscaled])

    btn_up.click(upscale_image, [current_img, upscale_factor, finish_sharpen, finish_contrast, finish_warmth, finish_grain], [upscaled, upscaled_base])
    for finish_control in [finish_sharpen, finish_contrast, finish_warmth, finish_grain]:
        finish_control.release(finish_image, [upscaled_base, finish_sharpen, finish_contrast, finish_warmth, finish_grain], upscaled)

if __name__ == "__main__":
    demo.launch(share=True, debug=True, theme=gr.themes.Soft(), css=css)