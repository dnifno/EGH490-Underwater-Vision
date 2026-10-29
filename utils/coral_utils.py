"""
Utility functions for loading Coralscapes SegFormer and running sliding-window semantic segmentation.

Usage:
    from coral_utils import load_coralscapes_model, segment_image, colourise_mask, coverage_stats
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_ID = "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"

# Official Coralscapes class table (from the dataset card)
CLASS_NAMES = {
    1: "seagrass", 2: "trash", 3: "other coral dead", 4: "other coral bleached",
    5: "sand", 6: "other coral alive", 7: "human", 8: "transect tools",
    9: "fish", 10: "algae covered substrate", 11: "other animal",
    12: "unknown hard substrate", 13: "background", 14: "dark",
    15: "transect line", 16: "massive/meandering bleached",
    17: "massive/meandering alive", 18: "rubble", 19: "branching bleached",
    20: "branching dead", 21: "millepora", 22: "branching alive",
    23: "massive/meandering dead", 24: "clam", 25: "acropora alive",
    26: "sea cucumber", 27: "turbinaria", 28: "table acropora alive",
    29: "sponge", 30: "anemone", 31: "pocillopora alive",
    32: "table acropora dead", 33: "meandering bleached", 34: "stylophora alive",
    35: "sea urchin", 36: "meandering alive", 37: "meandering dead",
    38: "crown of thorn", 39: "dead clam",
}
ID2LABEL = {0: "unlabelled", **CLASS_NAMES}

ID2COLOUR = {
    0: (0, 0, 0), 1: (125, 222, 125), 2: (255, 0, 134), 3: (114, 60, 61),
    4: (250, 224, 225), 5: (194, 178, 128), 6: (224, 118, 119), 7: (255, 0, 0),
    8: (8, 205, 12), 9: (255, 255, 0), 10: (125, 163, 125), 11: (0, 255, 255),
    12: (125, 125, 125), 13: (29, 162, 216), 14: (31, 31, 31), 15: (0, 255, 0),
    16: (255, 248, 228), 17: (236, 150, 21), 18: (161, 153, 128), 19: (252, 231, 240),
    20: (123, 50, 86), 21: (244, 150, 115), 22: (226, 91, 157), 23: (134, 86, 18),
    24: (189, 255, 234), 25: (236, 128, 255), 26: (0, 231, 255), 27: (228, 255, 119),
    28: (189, 119, 255), 29: (240, 80, 80), 30: (0, 255, 189), 31: (255, 146, 150),
    32: (85, 53, 116), 33: (251, 243, 216), 34: (255, 111, 194), 35: (0, 142, 255),
    36: (230, 193, 0), 37: (119, 100, 14), 38: (179, 245, 234), 39: (89, 155, 134),
}


def load_coralscapes_model(device):
    """
    Load the Coralscapes SegFormer model + preprocessor.

    Works around a bug in the hub config where id2label/label2id are malformed
    (mapping ints to ints instead of ints to class-name strings), which trips
    strict validation in recent huggingface_hub/transformers versions and, if
    naively stripped, silently shrinks the classifier head from 40 to 2 classes.
    They are replaced with a correctly-sized, correctly-typed mapping instead.
    """
    config_path = hf_hub_download(MODEL_ID, "config.json")
    with open(config_path) as f:
        cfg_dict = json.load(f)

    cfg_dict["id2label"] = {str(k): v for k, v in ID2LABEL.items()}
    cfg_dict["label2id"] = {v: k for k, v in ID2LABEL.items()}

    config = SegformerConfig.from_dict(cfg_dict)
    preprocessor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID, config=config).to(device)
    model.eval()
    return preprocessor, model


def resize_image(image, target_size=1024):
    w_img, h_img = image.size
    if h_img < w_img:
        new_h, new_w = target_size, int(w_img * (target_size / h_img))
    else:
        new_h, new_w = int(h_img * (target_size / w_img)), target_size
    return image.resize((new_w, new_h))


def segment_image(image, preprocessor, model, device, crop_size=(1024, 1024), num_classes=40):
    """
    Sliding-window semantic segmentation (matches the approach recommended in the Coralscapes model card 
    for images that aren't near the training resolution). Returns an (H, W) numpy array of predicted class 
    ids at the original image resolution.
    """
    h_crop, w_crop = crop_size
    resized = resize_image(image, target_size=1024)
    img = torch.Tensor(np.array(resized).transpose(2, 0, 1)).unsqueeze(0)
    _, _, h_img, w_img = img.size()

    h_grids = int(np.round(3 / 2 * h_img / h_crop)) if h_img > h_crop else 1
    w_grids = int(np.round(3 / 2 * w_img / w_crop)) if w_img > w_crop else 1
    h_stride = int((h_img - h_crop + h_grids - 1) / (h_grids - 1)) if h_grids > 1 else h_crop
    w_stride = int((w_img - w_crop + w_grids - 1) / (w_grids - 1)) if w_grids > 1 else w_crop

    preds = torch.zeros((1, num_classes, h_img, w_img))
    count_mat = torch.zeros((1, 1, h_img, w_img))

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1, x1 = h_idx * h_stride, w_idx * w_stride
            y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
            y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]

            with torch.no_grad():
                pil_crop = Image.fromarray(crop_img[0].numpy().transpose(1, 2, 0).astype(np.uint8))
                inputs = preprocessor(pil_crop, return_tensors="pt").to(device)
                outputs = model(**inputs)

            resized_logits = F.interpolate(
                outputs.logits, size=crop_img.shape[-2:], mode="bilinear", align_corners=False
            ).cpu()
            preds[:, :, y1:y2, x1:x2] += resized_logits
            count_mat[:, :, y1:y2, x1:x2] += 1

    preds = preds / count_mat
    preds = preds.argmax(dim=1)
    preds = F.interpolate(
        preds.unsqueeze(0).float(), size=(image.size[1], image.size[0]), mode="nearest"
    ).squeeze().numpy().astype(np.uint8)
    return preds


def colourise_mask(pred):
    """Convert an (H, W) class-id array to an (H, W, 3) RGB colour image using the official palette."""
    h, w = pred.shape
    colour_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, colour in ID2COLOUR.items():
        colour_mask[pred == class_id] = colour
    return colour_mask


def coverage_stats(pred):
    """Return a list of (class_name, percent_coverage) sorted by coverage, descending."""
    labels, counts = np.unique(pred, return_counts=True)
    total = pred.size
    stats = [(ID2LABEL.get(int(l), f"class_{l}"), 100 * c / total) for l, c in zip(labels, counts)]
    return sorted(stats, key=lambda x: -x[1])
