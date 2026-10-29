![Python](https://img.shields.io/badge/python-3.10-blue)
[![Coralscapes](https://img.shields.io/badge/CoralScapes-Repository-181717?logo=github)](https://github.com/eceo-epfl/coralscapesScripts)
[![FUnIE-GAN](https://img.shields.io/badge/FUnIE--GAN-GitHub-181717?logo=github)](https://github.com/xahidbuffon/FUnIE-GAN)
[![Sea-Thru](https://img.shields.io/badge/Sea--Thru-GitHub-181717?logo=github)](https://github.com/UIE-Lab/Sea-Thru)
[![Coralscapes V1](https://img.shields.io/badge/CoralScapes-V1-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/EPFL-ECEO/coralscapes)
[![Coralscapes V2](https://img.shields.io/badge/CoralScapes-V2-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/josauder/coralscapesV2)

# 🪸 Edge AI and Robotics for Monitoring Underwater Ecosystems 🤖

Code for a QUT capstone engineering research project, on AI-driven underwater perception with a commercial BlueROV2 in Moreton Bay, South-East Queensland.

The repository collects the notebooks and utility code behind the segmentation and enhancement experiments investigation how well a Coralscapes-pretrained SegFormer transfers to turbid Moreton Bay imagery without further training, and whether open-source underwater enhancement (FUnIE-GAN, Sea-thru, a classical pipeline) improves it.

## Research questions

| | Question | Where it is addressed |
|---|---|---|
| RQ1 | How do the existing seagrass (BOSG) and YOLO-based models perform on a rugged Ubuntu workstation, and is a Jetson Orin Nano needed? | See [Scope](#Scope) section below.  |
| RQ2 | How well does a Coralscapes-pretrained segmentation model perform on Moreton Bay imagery without further training, and is any degradation better explained by domain shift or image quality? | `coralscapes/`, `zeroshot/coral_eval.ipynb` |
| RQ3 | How effective are open-source underwater image enhancement methods, including GAN-based approaches, at improving that model's performance? | `enhancement/`, `zeroshot/coral_enhancement_comparison.ipynb` |

## Repository layout

```
.
├── README.md                  🫧 you are here
├── requirements.txt           Python dependencies for a virtual environment
├── .gitignore                 
│
├── data/                      inputs (see data/README.md)
│   ├── cvat_annotations/      CVAT export for segmentation annotations
│   └── bluerov_videos/        placeholder for videos (git-ignored)
│
├── utils/                     shared Python modules and scripts (see utils/README.md)
│   ├── classical_utils.py     classical enhancement pipeline
│   ├── funie_utils.py         FUnIE-GAN generator and enhancement
│   ├── seathru_utils.py       Sea-thru with Depth Anything V2 range maps
│   ├── coral_utils.py         Coralscapes V1 loader and sliding-window inference
│   ├── engine.py              model loading, inference and metrics for the evaluation notebooks
│   ├── labels.py              config loading and class-name mapping
│   ├── prepare_cvat_export.py CVAT export -> class-index masks
│   └── extract_cvat_frames.py extracts the 11 annotated frames from the video
│
├── coralscapes/               running Coralscapes with no ground truth (see coralscapes/README.md)
│   ├── coralscapes_v1.ipynb   Coralscapes V1 (SegFormer-B2) on the onboard BlueROV2 footage
│   └── coralscapes_v2.ipynb   Coralscapes V2 (SegFormer-B5) on the 4K DJI footage
│
├── enhancement/               one notebook per enhancement method (see enhancement/README.md)
│   ├── 01_enhancement_classical.ipynb
│   ├── 02_enhancement_funie_gan.ipynb
│   └── 03_enhancement_seathru.ipynb
│
└── zeroshot/                  scored evaluation against expert annotations (see zeroshot/README.md)
    ├── coral_eval.ipynb                  zero-shot evaluation of Coralscapes V1
    ├── coral_enhancement_comparison.ipynb  raw versus enhanced inputs, with paired bootstrap
    └── configs/
        ├── config.yaml                   classes, CVAT aliases, Coralscapes mapping, model choice
        └── classes_coralscapes_v1.json   corrected 40-entry class table for the V1 checkpoint
```

## Setup (as originally run)

The project was developed and run on a **MacBook Pro with an Apple M1 chip, 16 GB memory, macOS Tahoe 26.1**, in a Python 3.10 virtual environment. PyTorch ran on the Apple GPU through Metal Performance Shaders (MPS), with CPU as the fallback.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

N.B.:

- The code picks the device automatically (CUDA, then MPS, then CPU), so the same notebooks also run on Linux or Windows machines, only more slowly without a GPU.
- If MPS raises an unsupported-operator error, set `PYTORCH_ENABLE_MPS_FALLBACK=1` before starting Jupyter, or switch the device variable in the affected notebook to `"cpu"` (the Sea-thru notebook has a `DEPTH_DEVICE` variable for this).
- `transformers==4.46.3` and `huggingface_hub<1` are pinned because the Coralscapes V1 checkpoint used a malformed `config.json` that newer versions reject (`utils/engine.py` and `utils/coral_utils.py` work around it).
- Model weights download automatically on first use (Coralscapes and Depth Anything V2 from Hugging Face, FUnIE-GAN from its GitHub repository), so the first run needs internet access.

## Running the notebooks

Each notebook uses paths relative to its own folder, so Jupyter or VS Code must run it with that folder as the working directory (the default when a notebook is opened from the file browser). Shared modules are imported from `../utils`, and outputs are written next to the notebook (`outputs/` in `enhancement/`, `work/` in `zeroshot/`). All of these locations are git-ignored.

1. Add the inputs described in `data/README.md` (videos and the CVAT export).
2. `coralscapes/` notebooks: qualitative runs of the pretrained models, no setup beyond the videos.
3. `enhancement/` notebooks: each method on its own, with before and after visualisations.
4. `zeroshot/coral_eval.ipynb`: converts the CVAT export and scores the zero-shot model.
5. `zeroshot/coral_enhancement_comparison.ipynb`: reuses the masks from step 4 to score every enhancement method with the same model.

Approximate runtimes reported in the project: FUnIE-GAN and the classical pipeline take 0.4 to 0.5s per 1080p frame, Sea-thru about 9s per frame.

## Scope

RQ1 (benchmarking the seagrass and YOLO models on the workstation and the planned Jetson comparison) was implemented inside the research group's `submarroo` ROS repository and is not part of this repository. Everything here concerns RQ2 and RQ3.

## Data and licences

- Videos are not committed (see `data/README.md`).
- [Coralscapes](https://josauder.github.io/coralscapes/) models and datasets: EPFL-ECEO on Hugging Face (`EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024`, `josauder/coralscapesV2`).
- FUnIE-GAN: official code and weights by Islam, Xia and Sattar (MIT licence), https://github.com/xahidbuffon/FUnIE-GAN.
- Sea-thru: Akkaynak and Treibitz, CVPR 2019. The code in `utils/seathru_utils.py` is from https://github.com/hainh/sea-thru (MIT licence). Range maps come from Depth Anything V2 Small (Apache-2.0).
- The BlueROV2 and related hardware belong to the QUT Centre for Robotics.

## Author
<a href="https://github.com/dnifno">
        <img src="https://github.com/dnifno.png?size=80" width="80" style="border-radius:50%"/><br/>
</a>
      <b>Dani Fensom</b><br/>
