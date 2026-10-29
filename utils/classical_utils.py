"""
Classical, deterministic underwater image enhancement pipeline.

- Grey-world white balance: corrects the colour cast (useful for Moreton Bay's greenish/brown sediment-heavy water, 
  which skews very differently from the blue-green Red Sea / tropical imagery most learned models train on).
- Edge-redness reduction: a radial correction that damps the red channel
  towards the frame border, where a red cast was observed in the BlueROV2 footage.
- Underwater Dark Channel Prior (UDCP) dehazing: estimates and removes backscatter/haze, since red attenuates fastest underwater.
- CLAHE (contrast-limited adaptive histogram equalisation) on the L channel in LAB space: 
  recovers local contrast without blowing out highlights.

"""

import cv2
import numpy as np
from PIL import Image


def grey_world_white_balance(img_rgb):
    """Simple grey-world assumption: scale each channel so channel means match."""
    img = img_rgb.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    overall_mean = means.mean()
    scale = overall_mean / (means + 1e-6)
    balanced = img * scale
    return np.clip(balanced, 0, 255).astype(np.uint8)


def apply_clahe(img_rgb, clip_limit=2.5, tile_grid_size=(8, 8)):
    """CLAHE on the L channel in LAB space, preserving colour."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def _underwater_dark_channel(img_rgb, patch_size=15):
    """Underwater dark channel: min over a patch of min(G, B, 255-R)."""
    img = img_rgb.astype(np.float32)
    inverted_r = 255.0 - img[:, :, 0]
    min_channel = np.minimum(np.minimum(img[:, :, 1], img[:, :, 2]), inverted_r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def _estimate_atmospheric_light(img_rgb, dark_channel, top_percent=0.001):
    h, w = dark_channel.shape
    num_pixels = max(int(h * w * top_percent), 1)
    flat_dark = dark_channel.reshape(-1)
    flat_img = img_rgb.reshape(-1, 3).astype(np.float32)
    indices = np.argsort(flat_dark)[-num_pixels:]
    brightest = flat_img[indices]
    return brightest.max(axis=0)


def udcp_dehaze(img_rgb, patch_size=15, omega=0.85, t0=0.1):
    """Underwater Dark Channel Prior dehazing."""
    dark_channel = _underwater_dark_channel(img_rgb, patch_size)
    atmospheric_light = _estimate_atmospheric_light(img_rgb, dark_channel)

    img = img_rgb.astype(np.float32)
    norm_img = img / (atmospheric_light + 1e-6)
    # recompute dark channel on normalised image for transmission estimate
    inverted_r = 1.0 - norm_img[:, :, 0]
    min_channel = np.minimum(np.minimum(norm_img[:, :, 1], norm_img[:, :, 2]), inverted_r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    norm_dark = cv2.erode(min_channel, kernel)

    transmission = 1 - omega * norm_dark
    transmission = np.clip(transmission, t0, 1.0)

    result = np.empty_like(img)
    for c in range(3):
        result[:, :, c] = (img[:, :, c] - atmospheric_light[c]) / transmission + atmospheric_light[c]

    return np.clip(result, 0, 255).astype(np.uint8)


def reduce_edge_redness(img, strength=0.5, start_radius=0.55):
    """
    Reduce red colour cast towards the edges of an image.

    strength:
        0.0 = no correction
        1.0 = maximum correction

    start_radius:
        Fraction of image radius where correction starts.
        Larger = only corrects the extreme edges.
    """
    img = img.astype(np.float32)

    h, w = img.shape[:2]

    y, x = np.ogrid[:h, :w]

    cx = (w - 1) / 2
    cy = (h - 1) / 2

    # Normalised radial distance from centre
    r = np.sqrt(((x - cx) / (w / 2)) ** 2 + ((y - cy) / (h / 2)) ** 2)
    # Start correction only towards edges
    mask = np.clip( (r - start_radius) / (1.0 - start_radius), 0, 1 )
    # Smooth transition
    mask = mask * mask * (3 - 2 * mask)

    # Reduce red progressively towards edges
    red = img[:, :, 0]
    reduction = strength * mask * 0.25
    img[:, :, 0] = red * (1 - reduction)
    return np.clip(img, 0, 255).astype(np.uint8)

def classical_enhance(pil_image, use_white_balance=True, reduce_redness=True, use_dehaze=True, use_clahe=True):
    """Full pipeline: white balance -> edge-redness reduction -> dehaze -> CLAHE. Any stage can be disabled."""
    img = np.array(pil_image.convert("RGB"))

    if use_white_balance:
        img = grey_world_white_balance(img)
    if reduce_redness:
        img = reduce_edge_redness(img)
    if use_dehaze:
        img = udcp_dehaze(img)
    if use_clahe:
        img = apply_clahe(img)

    return Image.fromarray(img)
