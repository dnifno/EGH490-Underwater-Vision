# /zeroshot

Scored evaluation of the pretrained Coralscapes model on Moreton Bay frames, with no fine-tuning, against expert-assisted CVAT annotations. This folder holds the notebooks behind RQ2 and RQ3.

The CVAT export is read from `../data/cvat_annotations/`, shared code from `../utils/`, and all results are written to `work/` (git-ignored).

## Contents

| Path | Description |
|---|---|
| `coral_eval.ipynb` | Loads the model, converts the CVAT export to class-index masks, scores the zero-shot model on every annotated frame (overlays, per-class IoU, recall and precision, confusion matrix) and reports what the model's own classes predicted on the annotated pixels. |
| `coral_enhancement_comparison.ipynb` | Runs the same model on raw, FUnIE-GAN, Sea-thru and classical-enhanced versions of each annotated frame, plus a 640 px resolution control, and compares them with per-class metrics and a paired bootstrap over frames. Requires `coral_eval.ipynb` to have been run first. |
| `configs/config.yaml` | The single configuration file (see below). |
| `configs/classes_coralscapes_v1.json` | Corrected 40-entry class table patched into the V1 checkpoint's config. |

## Run order

1. Run `coral_eval.ipynb` from top to bottom. It writes `work/own/` (images, masks and `summary.json`) and `work/eval_notebook/` (CSV tables).
2. Run `coral_enhancement_comparison.ipynb`. It reads `work/own/` and writes `work/enhancement_comparison/`.

## Configuration (`configs/config.yaml`)

| Section | Purpose |
|---|---|
| `classes` | The ten project classes (branching, massive, tabular, encrusting, foliose, corymbose, marine_species, background, sand, rubble), their display colours and the ignore index (255). |
| `cvat.aliases` | Translates CVAT label names into project classes. CVAT exports unannotated pixels as black "background", so explicit background polygons need a different CVAT label (for example "negative"). |
| `v1_coralscapes_mapping` | Groups Coralscapes V1 classes into project classes, matched by **name**. Classes that are not listed (for example `other coral *` and `millepora`) are ignored: never scored, and counted as "unmapped" if predicted. |
| `coralscapes_mapping` | The equivalent mapping for the 95 CoralscapesV2 classes. |
| `model` | Active checkpoint and which mapping section it uses. V1 is active. To switch to V2, follow the comment in the file. |
| `data.own_dir` | Where the converted annotations are written (`work/own`). |

The mapping selected is a deliberate choice: expert-provided growth forms (branching, massive, tabular, encrusting, foliose, corymbose) mix with Coralscapes' genus-level classes, so there is no one-to-one conversion. Encrusting has no V1 equivalent.

## Results reported in the project

Scored on 11 frames from a single Myora Bay clip, where only 15.6% of pixels were annotated (regions expertly identified).

| Input | mIoU | Labelled-pixel accuracy | mIoU difference to raw (95% CI) | Verdict |
|---|---|---|---|---|
| Raw (1080p) | 0.120 | 0.251 | n/a | n/a |
| Classical | 0.124 | 0.249 | +0.004 (-0.030, +0.033) | Inconclusive |
| Raw at 640 px (control) | 0.101 | 0.201 | -0.019 (-0.045, +0.012) | Inconclusive |
| Sea-thru | 0.039 | 0.011 | -0.081 (-0.134, -0.050) | Worse |
| FUnIE-GAN | 0.0001 | 0.0003 | -0.120 (-0.182, -0.086) | Worse |

N.B. when reading these numbers:

- Branching and tabular coral supply almost all of the mIoU, and tabular coral is about 90% of annotated pixels, so rare classes are unreliable.
- Precision and IoU are inflated because false positives on unannotated pixels are never counted, so recall is the most reliable per-class metric.
- Enhancement was tested only as a pre-processing step for a frozen model, with fixed settings. The results apply to those configurations, not to enhancement in general.
- Domain shift and image quality could not be separated with this data -> the 4K DJI footage that could help was not annotated in time.
