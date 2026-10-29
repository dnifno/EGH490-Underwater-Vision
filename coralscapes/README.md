# /coralscapes

Runs pretrained Coralscapes SegFormer models on BlueROV2 footage with no ground truth and no fine-tuning. These notebooks are qualitative: they show what the models predict on Moreton Bay imagery. The scored evaluation is in `../zeroshot/`.

Working directory: this folder. Videos are read from `../data/bluerov_videos/` and shared code from `../utils/`.

## Notebooks

| Notebook | Model | Footage | Notes |
|---|---|---|---|
| `coralscapes_v1.ipynb` | Coralscapes V1, SegFormer-B2 (`EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024`), 39 benthic classes plus an unlabelled class | Onboard BlueROV2 camera, 1080p (`Myora-BlueROV.mp4`) | Samples a frame every 50 s, segments it and prints the five most common classes. Uses `utils/coral_utils.py`. |
| `coralscapes_v2.ipynb` | Coralscapes V2, SegFormer-B5 (`...-coralscapesv2-1024-1024-95_class` by default, 95 classes) | 4K DJI Osmo Action 5 Pro (`Myora-DJI-4K.mp4`) | Self-contained; reads class names and colours from the checkpoint and dataset at runtime. The `39` variant gives the legacy V1 label set. |

## How inference works

Both notebooks follow the sliding-window approach recommended on the Coralscapes model card: each frame is resized so its short side is 1024 px (aspect ratio kept), overlapping 1024x1024 windows are processed at the scale used in training, their logits are averaged where windows overlap, and the arg-max gives the class map, which is resized back to the original frame size with nearest-neighbour interpolation. Inference uses 32-bit floats.

## V1 checkpoint workaround

The published V1 `config.json` has malformed `id2label` and `label2id` entries (integers mapped to integers). Current library versions reject them, and deleting them makes the classifier head silently shrink from 40 outputs to 2, so the loader overwrites both with a correctly typed 40-entry table (index 0 `unlabeled`, indices 1 to 39 from the dataset card).

## Observations reported in the project

- **V1 on the onboard footage:** `background` covered 73 to 93% of pixels in every frame after the first, unknown hard substrate covered a further 4 to 19% where the seabed was visible, and coral classes rarely reached the top five. Coverage statistics alone could not separate domain shift from image quality, which motivated the annotated evaluation in `../zeroshot/`.
- **V2 on the 4K footage:** predictions covered a larger share of the frame, background no longer dominated, and small organisms missed in the V1 run appeared. The comparison is confounded (different model version and different camera) and the 4K footage is not annotated, so it is encouraging rather than conclusive.
