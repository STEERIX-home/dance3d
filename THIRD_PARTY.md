# Third-party software, models and data

Nothing in the first two tables is stored in this repository: the installer (`install/install.sh`,
`install/get_models.py`) downloads each part from its original source, at a pinned version, after you accepted
each model licence once. Only the data in the last table is in the repository. Read every licence yourself: the summaries below are pointers, not
legal advice. Each licence page was opened on 2026-11-21; where a page could not be opened, the row says so.

**Commercial use**: "no" means the licence allows only non-commercial (research, education, non-profit) use.
Several components are non-commercial (ExMR, SMPL-X, MANO, InsightFace models), so the programs as a whole can
only be used non-commercially unless you obtain other licences for those parts.

## Programs and code

| Component | Version | Licence | Commercial use | Who gets it |
| --- | --- | --- | --- | --- |
| [ExMR](https://github.com/twehrbein/ExMR) (motion capture; derived from GVHMR) | commit `eac72e4` | [ExMR / GVHMR terms](https://github.com/twehrbein/ExMR/blob/main/LICENSE): educational, research and non-profit use only; modifications must be open source | no | installer (git) |
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) code (ExMR submodule) | commit `b5c765a` | [SAM License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE) | yes, under its terms | installer (git) |
| [DINOv3](https://github.com/facebookresearch/dinov3) code (SAM 3D Body's backbone) | GitHub `main` at first use | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) | yes, under its terms | SAM 3D Body itself, through `torch.hub`, the first time a joints run needs it |
| [HaMeR](https://github.com/geopavlakos/hamer) code | commit `3a01849` | [MIT](https://github.com/geopavlakos/hamer/blob/main/LICENSE.md) | yes | installer (pip, from git) |
| [Sapiens2](https://github.com/facebookresearch/sapiens2) code | commit `7e5bae8` | [Sapiens2 License](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md) (use restrictions, e.g. no biometric processing, no deepfakes impersonating a person) | yes, under its terms | installer (pip, from git) |
| [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO11 tracking, SAM 3 runner) | 8.4.83 | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | yes, under AGPL-3.0 (or an Ultralytics enterprise licence) | installer (pip) |
| [CLIP, Ultralytics fork](https://github.com/ultralytics/CLIP) (SAM 3 text prompts) | commit `7ffa84b` | [AGPL-3.0](https://github.com/ultralytics/CLIP/blob/main/LICENSE) (the fork's licence file; the original OpenAI CLIP is MIT) | yes, under AGPL-3.0 | installer (pip, from git) |
| [MoGe](https://github.com/microsoft/MoGe) and [utils3d-moge](https://github.com/EasternJournalist/utils3d-moge) | commits `74fbce0`, `62f09d5` | [MIT](https://github.com/microsoft/MoGe/blob/main/LICENSE) | yes | installer (pip, from git) |
| [smplx](https://github.com/vchoutas/smplx) Python package | 0.1.28 | [SMPL-X licence](https://github.com/vchoutas/smplx/blob/main/LICENSE) (non-commercial) | no | installer (pip) |
| [PyTorch](https://pytorch.org) | 2.8.0 (CUDA 12.8), 2.13.0 (CUDA 13.0) | [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE); the CUDA libraries are under NVIDIA's licences | yes | installer (pip) |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | 0.44.0, commit `f4f3ded` | [GPL-3.0](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE) | yes, under GPL-3.0 | installer (git) |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | commit `6ea2651` | [Apache-2.0](https://github.com/city96/ComfyUI-GGUF/blob/main/LICENSE) | yes | installer (git) |
| [InsightFace](https://github.com/deepinsight/insightface) code | 0.7.3 | [MIT](https://github.com/deepinsight/insightface) (code) | yes | installer (pip; compiled) |
| [ONNX Runtime](https://onnxruntime.ai) | 1.23.2 (CPU) | MIT ([PyPI](https://pypi.org/project/onnxruntime/1.23.2/)) | yes | installer (pip) |
| [Blender](https://www.blender.org) | 4.5.16 | GNU GPL (see `copyright.txt` in the Blender folder; blender.org/about/license refused automated access, so the licence was read from the release archive) | yes | installer (download.blender.org, SHA-256 checked) |
| [uv](https://github.com/astral-sh/uv) and its Python 3.12.15 builds | 0.13.1 | [Apache-2.0 or MIT](https://github.com/astral-sh/uv/blob/main/LICENSE-APACHE); Python: PSF licence | yes | installer (GitHub release, SHA-256 checked) |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) (only for YouTube links) | 2026.10.26 | [Unlicense](https://github.com/yt-dlp/yt-dlp/blob/master/LICENSE) | yes; downloading videos may break the site's terms or the uploader's rights | installer (pip, `--joints`) |
| nvdiffrast | not installed | NVIDIA Source Code License (non-commercial) | no | **not installed**: only ExMR's own preview videos and `programs/duo-scene/render_check.py` use it |

## Model files

| Model | Files (size) | Licence | Commercial use | Who gets it |
| --- | --- | --- | --- | --- |
| [SCAIL-2](https://huggingface.co/zai-org/SCAIL-2) (video model, default) | GGUF Q3_K_M 8.1 GB / Q4_K_M 10.9 GB / Q6_K 13.8 GB from [realrebelai/SCAIL-2_GGUF](https://huggingface.co/realrebelai/SCAIL-2_GGUF), or fp8 17.7 GB from [Comfy-Org/SCAIL-2](https://huggingface.co/Comfy-Org/SCAIL-2); DPO LoRA 1.2 GB (relight LoRA 1.2 GB, optional) | MIT (model cards) | yes | installer (Hugging Face) |
| Wan 2.1 helper files for SCAIL-2, from [Comfy-Org/Wan-Animate-2](https://huggingface.co/Comfy-Org/Wan-Animate-2) | umt5-xxl fp8 text encoder 6.7 GB, VAE 0.25 GB, CLIP vision H 1.3 GB, lightx2v step-distill LoRA 0.74 GB | Apache-2.0 (model card) | yes | installer (Hugging Face) |
| [SAM 3](https://huggingface.co/facebook/sam3) | `sam3.pt` 3.5 GB | [SAM License](https://huggingface.co/facebook/sam3/blob/main/LICENSE) | yes, under its terms | installer; **gated**: request access on the model page, then `HF_TOKEN` |
| [SMPL-X](https://smpl-x.is.tue.mpg.de) body model | `SMPLX_NEUTRAL.npz` 0.1 GB from `models_smplx_v1_1.zip` | [SMPL-X Model License](https://smpl-x.is.tue.mpg.de/modellicense.html) | no | **you**: register, accept, download the zip; the installer takes it from there |
| [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` (detection, recognition) and `inswapper_128` | 0.2 GB + 0.55 GB | InsightFace models: non-commercial research only ([README](https://github.com/deepinsight/insightface#license)) | no | installer (GitHub release) |
| [ExMR](https://huggingface.co/twehrbein/ExMR) weights (`--joints`) | 0.24 GB | ExMR licence (as the code) | no | installer (Hugging Face) |
| [Sapiens2 1B pose](https://huggingface.co/facebook/sapiens2-pose-1b) (`--joints`) | 6.1 GB | [Sapiens2 License](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md) | yes, under its terms | installer (Hugging Face) |
| [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3) (`--joints`) | 2.8 GB | [SAM License](https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/LICENSE) | yes, under its terms | installer; **gated** like SAM 3 |
| HaMeR checkpoint ([authors' demo Space](https://huggingface.co/spaces/geopavlakos/HaMeR)) (`--joints`) | 2.7 GB | MIT (HaMeR) | yes | installer (Hugging Face Space) |
| [MANO](https://mano.is.tue.mpg.de) hand model (`--joints`) | `MANO_RIGHT.pkl` from `mano_v1_2.zip` | [MANO License](https://mano.is.tue.mpg.de/license.html) | no | **you**: register and download the zip; ExMR's own downloader converts it |
| YOLO11x / YOLO11m (`--joints`) | 0.15 GB | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | yes, under AGPL-3.0 | installer (GitHub release) |
| [MoGe-2 ViT-L normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal) (`--joints`) | 1.3 GB | MIT | yes | installer (Hugging Face) |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (optional describer, `--all`) | 16.6 GB | Apache-2.0 | yes | installer (Hugging Face) |
| [MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3) + [character swap LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA) + [turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo) (optional, `--h3`) | 42.2 GB | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE); the LoRAs are derivatives of H3, so its licence applies (the character swap LoRA names it; the turbo LoRA names no licence of its own) | with conditions (Section IV); **not in the EU, the UK, the Republic of Korea or the USA** (Sections I.5 and V.4), which also covers using, distributing or displaying its outputs there | installer, only with `--h3` and an explicit acceptance of the territory clause |

## Data in this repository

| What | Licence and conditions |
| --- | --- |
