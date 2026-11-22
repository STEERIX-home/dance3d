# Setup and production notes

## Watch the movie

Clone either private mirror with an authorized GitHub account, or use **Code → Download ZIP** and extract it. Open `index.html` in a browser. All media, styles and scripts are local. Playback starts muted, with an explicit sound control and a play-button fallback if the browser blocks autoplay. No public hosting is configured.

The README GIFs animate in GitHub unless the viewer has disabled animated images or enabled reduced motion. A full-resolution MP4 embedded in GitHub's file viewer still uses GitHub's own playback controls.

## Run generation

This is an archive of the working Linux pipeline, with deployment paths converted to environment variables. A clean-machine, end-to-end GPU install has **not** been performed from this archive. The existing production run produced the included movie; publication checks exercise the player, syntax, input validation, audio preservation and segment assembly on CPU.

Required components:

1. Linux, Python 3.12, FFmpeg/ffprobe, Blender 4.5, Noto CJK and DejaVu fonts.
2. ExMR and its dependencies. The source revisions used by the production machine are in [`source-versions.json`](source-versions.json). `pipeline/install-runtime.sh` records the separate ExMR PyTorch 2.8 / CUDA 12.8 environment; create that virtual environment before running the script.
3. ComfyUI with H3 support and KJNodes, the six models listed in `integrations/h3/config.py`, and `integrations/h3/custom_nodes/h3_local` installed in ComfyUI's custom nodes directory. The controller and the custom node must inherit the same environment. Set `H3_ROOT` to the absolute `integrations/h3` directory so the workflow endpoint can find its JSON.
4. An existing GPU dashboard / job helper that implements `status`, `available`, `acquire`, `renew`, `release`, `job-init`, `artifact-register` and `leases`. This integration is required by the current controllers. The helper and its credentials are intentionally not part of this repository. Do not replace managed GPU leases with unrestricted GPU launches.
5. Authorized model assets: ExMR, SMPL-X, MANO, HaMeR, Sapiens2, SAM 3D Body and YOLO. `pipeline/settings.py` lists the expected layout. Model weights are not included. Obtain them under their respective terms, then use `pipeline/prepare_models.py` for validation/import. It can read a Hugging Face token from the standard local credential store or `HF_TOKEN_PATH`; no token should be saved in source control.
6. Optional identity correction uses a separate InsightFace / ONNX Runtime environment. The production correction environment used InsightFace 0.7.3 and ONNX Runtime 1.23.2 on CPU. Keep its model licenses separate from the pipeline code's permissions.

This pipeline uses **MiniMax H3**. The orchestrator, source footage, model code and model weights have separate permissions and license terms; see [dependencies](DEPENDENCIES.md).

## Configure paths

Copy `.env.example` to a local environment file and replace every placeholder with paths from your installation. The Python scripts read exported environment variables; they do not automatically load a dotenv file. Load the configured file in your shell or your existing user systemd service.

| Variable | Meaning |
| --- | --- |
| `RGBMOCAP_JOBS` | Managed job root; default `/srv/training/jobs` |
| `RGBMOCAP_INSTALL`, `H3_INSTALL`, `H3_MODEL_INSTALL` | Existing installation job roots created with `job-init` |
| `GPU_AGENT_HELPER`, `GPU_DASHBOARD_URL` | Lease/artifact helper and status service |
| `EXMR_ROOT`, `EXMR_PYTHON` | ExMR checkout and its Python interpreter |
| `RGBMOCAP_CHECKPOINTS` | Validated ExMR/body/hand checkpoint root |
| `BLENDER_BIN`, `JPEG_LIB` | Blender executable and libjpeg-turbo library directory |
| `RGBMOCAP_UI_PYTHON` | Python with Gradio, Pillow and NumPy |
| `H3_ROOT`, `COMFYUI_ROOT`, `H3_PYTHON`, `H3_MODELS` | H3 controller, ComfyUI, interpreter and model directories |
| `FACE_PYTHON`, `FACE_MODELS` | Optional identity correction interpreter and models |
| `RGBMOCAP_DESKTOP_RESULTS` | Location for verified user-facing result copies |

`pipeline/control.py` expects existing user services named `rgbmocap-h3.service` and `h3-character-swap.service`. Their ExecStart commands should run their respective `control.py serve`; their cleanup hooks should use `control.py cleanup`. The service environment must contain the configured paths. Review each controller for its stop/release behavior before integrating it with another scheduler.

Once the dependencies, models, services and environment are configured:

```bash
python3 pipeline/control.py status
python3 pipeline/control.py start --open
# Stop the UI and its owned generation work:
python3 pipeline/control.py stop
```

The UI accepts a dance video, a full-body character photo, and an optional face-detail photo. Input validation checks format, size, decoding, image orientation, duration, requested range and generation dimensions before GPU work. Video source inputs are limited to 1 GiB. Validation does not guarantee pose tracking or identity quality.

## Long videos and cameras

`full_video.py` reconstructs the motion sequence, uses overlapping H3-compatible segments, then merges them to retain the full timeline and source audio. Longer output is produced through segmentation, rather than treating five seconds as the entire-video limit. Segment boundaries and generated poses may still show temporal differences.

`blender_guide.py` creates the orbit guide. `blender_dynamic_camera.py` adds a dynamic path to the reconstructed animation, and `blender_free_viewpoints.py` renders four cameras around the same frozen pose. These scripts render 3D guides; character appearance generation is a separate H3 stage. Before any GPU rendering or inference, acquire a scheduler lease and select only its assigned physical GPUs.

The production explainer scripts expect a prepared managed job with `edit-plan.json`, camera manifests, character references and generated clips. Those private intermediates are not included. `verify_free_camera_explainer.py` accepts `--orbit-sha256` instead of referencing the original machine's historical job path.

## CPU regression checks

Use an already-created managed job directory and a Python environment containing NumPy, Pillow, OpenCV and SciPy:

```bash
export RGBMOCAP_TEST_ROOT="$RGBMOCAP_TEST_JOB/outputs/regression"
export PYTHONDONTWRITEBYTECODE=1
cd pipeline
python -m unittest test_pipeline test_full_video
```

The phone-photo regression requires a local multi-picture MPO fixture via `RGBMOCAP_MPO_TEST_IMAGE`; it is skipped when absent. No personal photo fixture is committed. Other checks cover invalid inputs, two-reference conditioning, cancellation ownership, original-audio preservation, verified result copying and full frame coverage after segment assembly.

The managed-job disk reserve remains 5%, and completed job artifacts must be registered for operator retention decisions. Configuration changes in this archive do not alter the running production installation.
