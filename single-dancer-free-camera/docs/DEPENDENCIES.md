# Dependencies and attribution

This private archive does not grant a new blanket license for third-party code, model weights, reference material or music. Obtain access and permissions from the respective owners before reuse. No pretrained weights or original source video are redistributed here.

| Component | Role and upstream source |
| --- | --- |
| ExMR | Motion recovery: <https://github.com/twehrbein/ExMR> |
| RGBMoCap | Workflow reference; its Windows GUI is not bundled: <https://github.com/toyxyz/toyxyz_RGBMocap> |
| Blender | Scene, animated mesh and camera rendering: <https://www.blender.org/> |
| ComfyUI | H3 execution graph: <https://github.com/Comfy-Org/ComfyUI> |
| KJNodes | ComfyUI workflow utilities: <https://github.com/kijai/ComfyUI-KJNodes> |
| MiniMax H3 | Generative video model: <https://huggingface.co/MiniMaxAI/MiniMax-H3> — read its current model license, including territory and use restrictions |
| Comfy H3 weights | Execution format: <https://huggingface.co/Comfy-Org/MiniMax-H3> |
| Character Swap V1.5 | Appearance conditioning LoRA: <https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA> |
| FL2V Turbo | Eight-step H3 adaptation: <https://huggingface.co/lightx2v/Minimax-h3-Turbo> |
| H3 workflow | Adapted workflow: <https://github.com/TheBeast-Spec/minimax-h3-character-swap-lora-comfyui> |
| SAM 3D Body | Human mesh features used by ExMR: <https://github.com/facebookresearch/sam-3d-body> |
| SMPL-X and MANO | Parametric body and hand assets; obtain authorized downloads from their official projects |
| InsightFace | Optional face correction: <https://github.com/deepinsight/insightface> — pretrained model permissions are separate from code permissions |

The included H3 workflow retains its upstream node metadata. ExMR, ComfyUI and KJNodes themselves are installed separately. Exact source revisions recorded from the production installation are in [`source-versions.json`](source-versions.json).

The movie identifies its motion reference only as **아일릿 체포하겠어 댄스 챌린지 영상**. Original source footage and its URL are omitted. The included video contains the generated Reze appearance and source music, and remains restricted to the private repository's authorized viewers.
