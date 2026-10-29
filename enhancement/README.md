# /enhancement

One notebook per open-source underwater enhancement method. They show each method on BlueROV2 footage with before and after visualisations; they do not score segmentation against annotations. The scored comparison is `../zeroshot/coral_enhancement_comparison.ipynb`, which sits in `zeroshot/` because it reuses the converted CVAT masks and the config file produced there.

Working directory: this folder. Videos are read from `../data/bluerov_videos/`, shared code from `../utils/`, and results are written to `outputs/` (git-ignored).

## Notebooks

| Notebook | Method | Type | What it does |
|---|---|---|---|
| `01_enhancement_classical.ipynb` | Grey-world white balance, edge-redness reduction, UDCP dehazing, CLAHE | Classical, training-free | Shows each stage on one frame, enhances frames sampled from `Shortened.mp4`, saves them, then runs enhanced frames from `Myora-BlueROV.mp4` through Coralscapes V1 as a qualitative check. |
| `02_enhancement_funie_gan.ipynb` | FUnIE-GAN | Learned (conditional GAN) | Downloads the official pretrained weights, enhances a video, builds before and after comparisons, and tests three ways of handling the 256x256 network size (raw output, linear versus cubic upscaling, near-native-resolution inference). |
| `03_enhancement_seathru.ipynb` | Sea-thru + Depth Anything V2 Small | Physics-based | Estimates a relative depth map, rescales it to an assumed metric range, runs Sea-thru, sweeps the range assumptions and compares Coralscapes predictions before and after. |

## Three methods summary

- **FUnIE-GAN.** A GAN (generative adversarial network) is trained with two competing networks: a generator that produces an enhanced image and a discriminator that judges whether it looks realistic. FUnIE-GAN is a fast conditional GAN trained on the EUVP dataset for underwater colour correction. Its input size is fixed at 256x256.
- **Classical pipeline.** Deterministic image processing with no weights. It is a transparent baseline and cannot model range-dependent attenuation, because every stage is global.
- **Sea-thru.** Estimates backscatter and range-dependent attenuation and inverts both. It needs a per-pixel range map; here a monocular depth estimate is rescaled to an assumed range (0.5 to 2.5m), so its output is exploratory.

## Results reported in the project (from `zeroshot/`)

Against the 11 expert-annotated frames, none of the methods gave a statistically supported improvement over raw frames. The classical pipeline was inconclusive (mIoU 0.124 versus 0.120 raw, confidence interval spanning zero), while FUnIE-GAN (0.0001) and Sea-thru (0.039) substantially reduced performance. These apply to the tested configurations with fixed settings, pretrained EUVP weights at 256px, an assumed Sea-thru range, and a frozen segmentation model.

## Outputs if saved locally

| Folder | Written by |
|---|---|
| `outputs/classical/` | `01_enhancement_classical.ipynb` |
| `outputs/funie/` | `02_enhancement_funie_gan.ipynb` |
| `outputs/seathru/` | `03_enhancement_seathru.ipynb` |
