"""
Utility functions for loading FUnIE-GAN and enhancing underwater images.

Downloads the official pretrained generator weights directly from the xahidbuffon/FUnIE-GAN repo (MIT licensed) and rebuilds the matching 5-layer U-Net architecture.
"""

import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

WEIGHTS_URL = "https://raw.githubusercontent.com/xahidbuffon/FUnIE-GAN/master/PyTorch/models/funie_generator.pth"
INFER_SIZE = (256, 256)  # (width, height) the network was trained on


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


class GeneratorFunieGAN(nn.Module):
    """5-layer UNet-based generator, as described in the FUnIE-GAN paper."""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1)
        return self.final(u45)


def load_funie_generator(weights_path="funie_generator.pth", device="cpu"):
    """Download (if needed) and load the official pretrained FUnIE-GAN generator."""
    if not os.path.exists(weights_path):
        urllib.request.urlretrieve(WEIGHTS_URL, weights_path)

    generator = GeneratorFunieGAN().to(device)
    generator.load_state_dict(torch.load(weights_path, map_location=device))
    generator.eval()
    return generator


def enhance_frame(generator, pil_image, device="cpu"):
    """Run a single PIL image through the FUnIE-GAN generator, returning a
    PIL image resized back to the original resolution."""
    orig_size = pil_image.size  # (w, h)

    resized = pil_image.resize(INFER_SIZE, Image.BILINEAR)
    arr = np.array(resized).astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        out = generator(tensor)

    out = out.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    out = ((out + 1.0) * 127.5).clip(0, 255).astype(np.uint8)  # [-1,1] -> [0,255]
    return Image.fromarray(out).resize(orig_size, Image.BICUBIC)