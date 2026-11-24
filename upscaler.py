import os
import urllib.request
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from PIL import Image, ImageEnhance

from utils import OUTPUT_DIR, free_gpu

# Real-ESRGAN x4plus upscaling and finishing controls
ESRGAN_PATH = "/kaggle/working/RealESRGAN_x4plus.pth"
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))

def load_upscaler():
    if not os.path.exists(ESRGAN_PATH):
        urllib.request.urlretrieve(ESRGAN_URL, ESRGAN_PATH)
    net = RRDBNet()
    checkpoint = torch.load(ESRGAN_PATH, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    net.load_state_dict(state_dict, strict=True)
    return net.eval().to("cuda")

@torch.no_grad()
def upscale_tiled(net, img, tile=256, pad=32):
    """Upscale one PIL image 4x in tiles; works for portrait, landscape, and odd dimensions."""
    scale = 4
    if tile <= 0 or pad < 0:
        raise ValueError("tile must be positive and pad cannot be negative")

    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32).copy() / 255.0
    input_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    first_param = next(net.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)

    input_height, input_width = input_tensor.shape[-2:]
    output = torch.empty((1, 3, input_height * scale, input_width * scale), dtype=torch.float32, device="cpu")

    for top in range(0, input_height, tile):
        bottom = min(top + tile, input_height)
        tile_top = max(0, top - pad)
        tile_bottom = min(input_height, bottom + pad)
        core_height = (bottom - top) * scale
        crop_top = (top - tile_top) * scale

        for left in range(0, input_width, tile):
            right = min(left + tile, input_width)
            tile_left = max(0, left - pad)
            tile_right = min(input_width, right + pad)
            core_width = (right - left) * scale
            crop_left = (left - tile_left) * scale

            input_tile = input_tensor[:, :, tile_top:tile_bottom, tile_left:tile_right]
            output_tile = net(input_tile).to(device="cpu", dtype=torch.float32)
            expected_tile_shape = ((tile_bottom - tile_top) * scale, (tile_right - tile_left) * scale)
            if output_tile.ndim != 4 or output_tile.shape[0] != 1 or output_tile.shape[1] != 3 or output_tile.shape[-2:] != expected_tile_shape:
                raise RuntimeError(
                    f"Upscaler returned tile shape {tuple(output_tile.shape)}; expected 1x3x{expected_tile_shape[0]}x{expected_tile_shape[1]}."
                )

            core = output_tile[:, :, crop_top:crop_top + core_height, crop_left:crop_left + core_width]
            if core.shape[-2:] != (core_height, core_width):
                raise RuntimeError(
                    f"Tile crop is {tuple(core.shape[-2:])}; expected {(core_height, core_width)}. "
                    f"Check tile bounds: image={input_width}x{input_height}, x={left}:{right}, y={top}:{bottom}."
                )
            output[:, :, top * scale:bottom * scale, left * scale:right * scale] = core

    output = output.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((output * 255).round().astype(np.uint8))

def finish_image(img, sharpen, contrast, warmth, grain):
    if img is None:
        return gr.skip()
    out = ImageEnhance.Contrast(img).enhance(float(contrast))
    out = ImageEnhance.Sharpness(out).enhance(1.0 + float(sharpen))
    arr = np.asarray(out.convert("RGB"), dtype=np.float32).copy()
    warmth = float(warmth)
    arr[:, :, 0] += warmth
    arr[:, :, 2] -= warmth
    grain = float(grain)
    if grain > 0:
        grain_seed = zlib.adler32(img.tobytes())
        noise = np.random.default_rng(grain_seed).normal(0, grain, size=(arr.shape[0], arr.shape[1], 1))
        arr += noise.astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8))

def upscale_image(img, factor, sharpen, contrast, warmth, grain):
    if img is None:
        raise gr.Error("Generate an image first.")
    net = load_upscaler()
    try:
        base = upscale_tiled(net, img)
    finally:
        del net
        free_gpu()
    if factor == "2x":
        base = base.resize((base.width // 2, base.height // 2), Image.Resampling.LANCZOS)
    finished = finish_image(base, sharpen, contrast, warmth, grain)
    return finished, base