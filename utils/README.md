# /utils

Shared Python modules and small command-line scripts. Notebooks import them with `sys.path.append("../utils")`, so this folder never needs installing as a package.

## Modules

| File | Purpose | Used by |
|---|---|---|
| `classical_utils.py` | Classical enhancement: grey-world white balance, edge-redness reduction, UDCP dehazing and CLAHE. `classical_enhance(pil_image, ...)` runs the stages in that order and each stage has an on/off flag. No weights. | `enhancement/01_...`, `zeroshot/coral_enhancement_comparison.ipynb` |
| `funie_utils.py` | FUnIE-GAN: rebuilds the 5-layer U-Net generator, downloads the official pretrained weights on first use, and enhances a PIL image. Inference runs at 256x256 and the result is resized back to the original resolution. | `enhancement/02_...`, `zeroshot/coral_enhancement_comparison.ipynb` |
| `seathru_utils.py` | Sea-thru (from `hainh/sea-thru`, MIT) with range maps estimated by Depth Anything V2 Small. `load_seathru(device)` loads the depth model; `enhance_frame(...)` returns the enhanced frame at the original resolution. | `enhancement/03_...`, `zeroshot/coral_enhancement_comparison.ipynb` |
| `coral_utils.py` | Coralscapes V1 loader (`load_coralscapes_model`), sliding-window inference (`segment_image`), the official class table and colour palette, `colourise_mask` and `coverage_stats`. | `coralscapes/coralscapes_v1.ipynb`, `enhancement/01_...`, `enhancement/03_...` |
| `engine.py` | Model loading (`load_reference`), sliding-window inference with optional confidence output (`segment_image`), device selection (`pick_device`) and confusion-matrix metrics (`update_confusion`, `metrics_from_confusion`). | `zeroshot/` notebooks |
| `labels.py` | `load_config` for `zeroshot/configs/config.yaml`, class-name resolution (`resolve_id2label`) and the name-based Coralscapes-to-project lookup table (`pick_mapping`). | `zeroshot/` notebooks, `prepare_cvat_export.py` |

`coral_utils.py` and `engine.py` implement the same sliding-window inference. `coral_utils.py` serves the exploratory notebooks, while `engine.py` serves the scored evaluation and adds confidence output and metrics.

## Scripts

| File | Purpose | Example (run from the folder shown) |
|---|---|---|
| `prepare_cvat_export.py` | Converts a CVAT "Segmentation mask 1.1" export into single-channel class-index masks (255 = not annotated) and prints the label mapping and class balance. `coral_eval.ipynb` calls it automatically. | `zeroshot/`: `python ../utils/prepare_cvat_export.py --export-dir ../data/cvat_annotations` |
| `extract_cvat_frames.py` | Saves the 11 annotated frames from the onboard video as 1920x1080 PNG, ready for upload to CVAT. | repository root: `python utils/extract_cvat_frames.py` |

## Metric definitions (engine.py)

- **Confusion matrix:** `(K, K+1)`, rows are annotated classes, columns are predicted classes, and the last column counts predictions outside the project class scheme ("unmapped"). Unannotated pixels are skipped.
- **IoU, recall, precision:** per class from the confusion matrix. Precision is optimistic when frames are only partly annotated, because false positives on unannotated pixels are never counted.
- **mIoU:** mean IoU over classes with at least one annotated pixel.
- **Labelled-pixel accuracy:** correctly classified annotated pixels over all annotated pixels.
