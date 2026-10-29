"""
Model loading, sliding-window inference and metric helpers for the Coralscapes SegFormer checkpoints (V1 SegFormer-B2 and V2 SegFormer-B5).

Inference follows the approach recommended on the Coralscapes model card: the frame is resized so its 
short side is 1024 px, a 1024x1024 window slides across it, logits from overlapping windows are averaged, 
and the arg-max gives the class map, which is resized back to the original frame size.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pick_device():
    """Prefer CUDA, then Apple GPU (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_reference(checkpoint, device, preprocessor_checkpoint=None, id2label_file=None):
    """
    Load a SegformerImageProcessor and SegformerForSemanticSegmentation.

    The published V1 checkpoint ships an id2label/label2id that maps integers to
    integers, which current transformers versions reject, and deleting them makes
    the classifier head shrink from 40 outputs to 2. When id2label_file
    (a JSON {id: name} table) is given, the hub config is patched with that table
    instead. Without it the checkpoint loads unmodified (CoralscapesV2).
    """
    from huggingface_hub import hf_hub_download
    from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor

    preprocessor = SegformerImageProcessor.from_pretrained(preprocessor_checkpoint or checkpoint)
    if id2label_file:
        with open(hf_hub_download(checkpoint, "config.json")) as f:
            cfg_dict = json.load(f)
        with open(id2label_file) as f:
            table = {int(k): v for k, v in json.load(f).items()}
        cfg_dict["id2label"] = {str(k): v for k, v in table.items()}
        cfg_dict["label2id"] = {v: k for k, v in table.items()}
        model = SegformerForSemanticSegmentation.from_pretrained(checkpoint, config=SegformerConfig.from_dict(cfg_dict))
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(checkpoint)
    model = model.to(device)
    model.eval()
    return preprocessor, model


def resize_image(image, target_size=1024):
    """Resize a PIL image so its short side equals target_size, keeping the aspect ratio."""
    w_img, h_img = image.size
    if h_img < w_img:
        new_h, new_w = target_size, int(w_img * (target_size / h_img))
    else:
        new_h, new_w = int(h_img * (target_size / w_img)), target_size
    return image.resize((new_w, new_h))


def segment_image(image, preprocessor, model, device, crop_size=(1024, 1024), return_conf=False):
    """
    Sliding-window semantic segmentation of one PIL image.

    Returns an (H, W) uint8 array of predicted class ids at the original image resolution. 
    With return_conf=True, returns (ids, conf) where conf is the (H, W) float32 softmax probability of the winning class.
    """
    num_classes = model.config.num_labels
    h_crop, w_crop = crop_size
    resized = resize_image(image, target_size=1024)
    img = torch.Tensor(np.array(resized).transpose(2, 0, 1)).unsqueeze(0)
    _, _, h_img, w_img = img.size()

    h_grids = int(np.round(3 / 2 * h_img / h_crop)) if h_img > h_crop else 1
    w_grids = int(np.round(3 / 2 * w_img / w_crop)) if w_img > w_crop else 1
    h_stride = int((h_img - h_crop + h_grids - 1) / (h_grids - 1)) if h_grids > 1 else h_crop
    w_stride = int((w_img - w_crop + w_grids - 1) / (w_grids - 1)) if w_grids > 1 else w_crop

    logits_sum = torch.zeros((1, num_classes, h_img, w_img))
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

            window_logits = F.interpolate(
                outputs.logits, size=crop_img.shape[-2:], mode="bilinear", align_corners=False
            ).cpu()
            logits_sum[:, :, y1:y2, x1:x2] += window_logits
            count_mat[:, :, y1:y2, x1:x2] += 1

    logits = logits_sum / count_mat
    out_size = (image.size[1], image.size[0])
    ids = F.interpolate(logits.argmax(dim=1).unsqueeze(0).float(), size=out_size, mode="nearest")
    ids = ids.squeeze().numpy().astype(np.uint8)
    if not return_conf:
        return ids
    conf = F.interpolate(logits.softmax(dim=1).max(dim=1).values.unsqueeze(0), size=out_size, mode="nearest")
    return ids, conf.squeeze().numpy().astype(np.float32)


def update_confusion(conf, gt, pred, num_classes, ignore_index):
    """
    Add one frame to a (K, K+1) confusion matrix. Rows are ground-truth classes,
    columns are predicted classes, and column K counts predictions that fall outside the project scheme ('unmapped'). 
    Ground-truth pixels equal to ignore_index are skipped. Returns the updated matrix.
    """
    valid = (gt != ignore_index) & (gt < num_classes)
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    p = np.where((p >= 0) & (p <= num_classes), p, num_classes)
    conf += np.bincount(g * (num_classes + 1) + p, minlength=num_classes * (num_classes + 1)).reshape(
        num_classes, num_classes + 1
    )
    return conf


def metrics_from_confusion(conf, names):
    """
    Per-class and aggregate metrics from a (K, K+1) confusion matrix.

    Returns {'mIoU', 'pixel_acc', 'per_class': {name: {gt_pixels, iou, recall, precision}}}.
    mIoU is the mean IoU over classes that have at least one ground-truth pixel, and pixel_acc is correct labelled pixels over all labelled pixels.
    Precision is optimistic when only part of each frame is annotated, because false positives on unannotated pixels are never counted.
    """
    k = len(names)
    per_class, present_ious = {}, []
    for i, name in enumerate(names):
        tp = int(conf[i, i])
        gt_pixels = int(conf[i, :].sum())
        fp = int(conf[:, i].sum()) - tp
        fn = gt_pixels - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
        per_class[name] = {
            "gt_pixels": gt_pixels,
            "iou": iou,
            "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        }
        if gt_pixels > 0 and np.isfinite(iou):
            present_ious.append(iou)
    total = int(conf.sum())
    return {
        "mIoU": float(np.mean(present_ious)) if present_ious else float("nan"),
        "pixel_acc": int(np.trace(conf[:, :k])) / total if total else float("nan"),
        "per_class": per_class,
    }
