#!/usr/bin/env python3
"""The model files of the dance programs: download them, check them, or take them from folders you already have.

    python3 install/get_models.py --challenge --vram 12        what the challenge video needs (12 GB card)
    python3 install/get_models.py --joints                     adds what "video -> 3D joints" needs
    python3 install/get_models.py --all                        challenge + joints + describer + relight LoRA
    python3 install/get_models.py --h3                         adds MiniMax H3 (optional; see its licence)
    python3 install/get_models.py --check [--verify]           what is there (size; --verify: also SHA-256)
    python3 install/get_models.py --models-from DIR [DIR ...]  link (or --copy) files you already have, then download the rest
    python3 install/get_models.py --list                       the table: every file, its source, size, licence
    python3 install/get_models.py --verify-sources             ask every source that the pinned file is still there

Every licence is shown once and must be accepted (type yes; without a terminal: --accept <licence id>). The
accepted licences are kept in <install root>/licences-accepted.json. Files come from pinned revisions and are
checked against their SHA-256; an interrupted download goes on from where it stopped (<file>.part).
SMPL-X and MANO need a registration on their websites: download the zip yourself, the script finds it.
Gated Hugging Face models (SAM 3, SAM 3D Body) need approved access and a token (HF_TOKEN).
Only the Python standard library is used (Python 3.8 or newer).
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import namedtuple
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
USER_AGENT = 'dance3d-get-models/1'
TIERS = (8, 12, 16, 24)  # GB of graphics memory; must match programs/duo-scene/scail_tiers.py

Row = namedtuple('Row', 'group level tier licence source dest size sha256')
# group, level, VRAM tier (None = every tier), licence id, source, file inside the models folder, bytes, SHA-256.
# source: "<repo>@<revision>:<file>" (Hugging Face; "spaces/<repo>" for a Space), an https URL, "<archive>:<member>"
# (a file inside one of ARCHIVES), or "user:<zip>:<member>" (a zip the user downloads after registering).
# level: challenge (one picture -> challenge video), joints (video -> 3D joints; also needs everything of challenge),
# relight / describer (optional extras, in --all), h3 (optional MiniMax H3 backend).
ROWS = [Row(*row) for row in [
    ('scail-2', 'challenge', 8, 'mit', 'realrebelai/SCAIL-2_GGUF@16e572157fbc2ffa76782dd5134a9262dc1a40cf:SCAIL-2-Q3_K_M.gguf', 'comfy/unet/SCAIL-2-Q3_K_M.gguf', 8108662432, '86c718053716c217f30ff887c01f0f3e247c2f4319944e32f59632ee54976b3d'),
    ('scail-2', 'challenge', 12, 'mit', 'realrebelai/SCAIL-2_GGUF@16e572157fbc2ffa76782dd5134a9262dc1a40cf:SCAIL-2-Q4_K_M.gguf', 'comfy/unet/SCAIL-2-Q4_K_M.gguf', 10862812832, 'ced59d192230b56990c7cb2ce905ae72cac08f564639f6198edff57e035f2646'),
    ('scail-2', 'challenge', 16, 'mit', 'realrebelai/SCAIL-2_GGUF@16e572157fbc2ffa76782dd5134a9262dc1a40cf:SCAIL-2-Q6_K.gguf', 'comfy/unet/SCAIL-2-Q6_K.gguf', 13755408032, 'e2b0975d3f8207a0eb30f5cc80c71a9a58db935945f8e0780e51bf675ff6c932'),
    ('scail-2', 'challenge', 24, 'mit', 'Comfy-Org/SCAIL-2@fe3c728bc793ba21ca674688f822afb709ad44fb:diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors', 'comfy/diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors', 17694586857, '11513b4697ecf566de0cb74660c478f301fb6699a62b10369e91a6ed0fd6b083'),
    ('scail-2', 'challenge', None, 'mit', 'Comfy-Org/SCAIL-2@fe3c728bc793ba21ca674688f822afb709ad44fb:loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors', 'comfy/loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors', 1226936552, 'b106522036f64e50f5f8ae3b808973515ff442cc2fac27b65d875eafb95b89e2'),
    ('scail-2', 'relight', None, 'mit', 'Comfy-Org/SCAIL-2@fe3c728bc793ba21ca674688f822afb709ad44fb:loras/wan2.1_SCAIL_2_relight_lora_bf16.safetensors', 'comfy/loras/wan2.1_SCAIL_2_relight_lora_bf16.safetensors', 1226936552, '538056c306179bb82e88f8e965609e7ebf99fbf2caa8b9e49db6a718974c3d7b'),
    ('wan-2.1', 'challenge', None, 'apache-2.0', 'Comfy-Org/Wan-Animate-2@e7181fe1896b7f2ce34120d1a3663413548240d7:loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors', 'comfy/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors', 738005744, '85c4a61c30e0497aa44b91d93a893b624708461a56fe5485183b28fa07e2dfb3'),
    ('wan-2.1', 'challenge', None, 'apache-2.0', 'Comfy-Org/Wan-Animate-2@e7181fe1896b7f2ce34120d1a3663413548240d7:text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'comfy/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors', 6735906897, 'c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68'),
    ('wan-2.1', 'challenge', None, 'apache-2.0', 'Comfy-Org/Wan-Animate-2@e7181fe1896b7f2ce34120d1a3663413548240d7:vae/Wan2_1_VAE_bf16.safetensors', 'comfy/vae/Wan2_1_VAE_bf16.safetensors', 253806278, '1ab9a32cc2c740f6e39d80d367ce5dcc28db8c71b79b28670546b8973e9d75f9'),
    ('wan-2.1', 'challenge', None, 'apache-2.0', 'Comfy-Org/Wan-Animate-2@e7181fe1896b7f2ce34120d1a3663413548240d7:clip_vision/clip_vision_h.safetensors', 'comfy/clip_vision/clip_vision_h.safetensors', 1264219396, '64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161'),
    ('sam-3', 'challenge', None, 'sam3', 'facebook/sam3@3c879f39826c281e95690f02c7821c4de09afae7:sam3.pt', 'sam3/sam3.pt', 3450062241, '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'),
    ('exmr', 'joints', None, 'exmr', 'twehrbein/ExMR@8debd2e34262fcc63384f7e04d39ca784b0aec21:exmr.safetensors', 'motion/exmr/exmr.safetensors', 235766520, '8d3947310c30444ec59dae4662574d7a32bf83af30320eeac87146529ff2fda5'),
    ('exmr', 'joints', None, 'exmr', 'twehrbein/ExMR@8debd2e34262fcc63384f7e04d39ca784b0aec21:config.json', 'motion/exmr/config.json', 180, 'b29ad86d7916a7f54154c1a20fb07d3bc9d794da4d671c88ba0cf75a5cec8c43'),
    ('sapiens2', 'joints', None, 'sapiens2', 'facebook/sapiens2-pose-1b@f5fed8b97b99698d5eea1d14ff0855d0b4c3f000:sapiens2_1b_pose.safetensors', 'motion/sapiens2/sapiens2_1b_pose.safetensors', 6079194752, '2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f'),
    ('sam-3d-body', 'joints', None, 'sam-3d-body', 'facebook/sam-3d-body-dinov3@11aaa346c7204874a1cbafe3d39a979080b2c55a:model.ckpt', 'motion/sam-3d-body-dinov3/model.ckpt', 2109129346, 'b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf'),
    ('sam-3d-body', 'joints', None, 'sam-3d-body', 'facebook/sam-3d-body-dinov3@11aaa346c7204874a1cbafe3d39a979080b2c55a:model_config.yaml', 'motion/sam-3d-body-dinov3/model_config.yaml', 1488, '1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d'),
    ('sam-3d-body', 'joints', None, 'sam-3d-body', 'facebook/sam-3d-body-dinov3@11aaa346c7204874a1cbafe3d39a979080b2c55a:assets/mhr_model.pt', 'motion/sam-3d-body-dinov3/assets/mhr_model.pt', 696110248, '352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'),
    ('hamer', 'joints', None, 'mit', 'spaces/geopavlakos/HaMeR@aabc78ee4fd1f1ccf1a55fdd4225bf84df0e680a:_DATA/hamer_ckpts/checkpoints/hamer.ckpt', 'motion/hamer/hamer_ckpts/checkpoints/hamer.ckpt', 2689536166, 'e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53'),
    ('hamer', 'joints', None, 'mit', 'spaces/geopavlakos/HaMeR@aabc78ee4fd1f1ccf1a55fdd4225bf84df0e680a:_DATA/hamer_ckpts/model_config.yaml', 'motion/hamer/hamer_ckpts/model_config.yaml', 2084, '2217f6bcf27ae4c074e2a33a3409d822bbfd02931373deedb9f71287adeabe16'),
    ('hamer', 'joints', None, 'mit', 'spaces/geopavlakos/HaMeR@aabc78ee4fd1f1ccf1a55fdd4225bf84df0e680a:_DATA/hamer_ckpts/dataset_config.yaml', 'motion/hamer/hamer_ckpts/dataset_config.yaml', 1369, '392d10aece296152769328f14bb1c0137d6ea2f50f7e69f716ee00e9ad6ac343'),
    ('hamer', 'joints', None, 'mit', 'spaces/geopavlakos/HaMeR@aabc78ee4fd1f1ccf1a55fdd4225bf84df0e680a:_DATA/data/mano_mean_params.npz', 'motion/hamer/data/mano_mean_params.npz', 1178, 'efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988'),
    ('moge-2', 'joints', None, 'mit', 'Ruicheng/moge-2-vitl-normal@cb0e8bbd6b1e243589717c78e750b1ba4c093acf:model.pt', 'moge/model.pt', 1323815904, '280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:README.md', 'describer/qwen2.5-vl-7b-instruct/README.md', 18574, '1fa65dbb08bc9ffe0b020409c8686f08b23008c5a68554353305fd2de6f2b81e'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:chat_template.json', 'describer/qwen2.5-vl-7b-instruct/chat_template.json', 1050, 'ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:config.json', 'describer/qwen2.5-vl-7b-instruct/config.json', 1374, '77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:generation_config.json', 'describer/qwen2.5-vl-7b-instruct/generation_config.json', 216, '0a3aea82869fe29f20dc95ccf3e2bcff380eca1f5ad6447a4a4b37110b08e43e'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:merges.txt', 'describer/qwen2.5-vl-7b-instruct/merges.txt', 1671839, '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:model-00001-of-00005.safetensors', 'describer/qwen2.5-vl-7b-instruct/model-00001-of-00005.safetensors', 3900233256, 'e97b877e47fde53a6c6e77aafb36e58e91ee9d95c4a3eeac6f1b5c0e6a1c986e'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:model-00002-of-00005.safetensors', 'describer/qwen2.5-vl-7b-instruct/model-00002-of-00005.safetensors', 3864726320, 'a9a300a43b4724eee2abe7c18ceb26768d0ab011eb0cad19d9bfd2476a24d024'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:model-00003-of-00005.safetensors', 'describer/qwen2.5-vl-7b-instruct/model-00003-of-00005.safetensors', 3864726424, '111223d173e00bbee81cba1216fad28668df3476706b7fd26f4d5b50f8b3a507'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:model-00004-of-00005.safetensors', 'describer/qwen2.5-vl-7b-instruct/model-00004-of-00005.safetensors', 3864733680, 'ef47f634fa57d46ee134edcc09f34085a47da1e16c12a2abe0d67118be6d72ed'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:model-00005-of-00005.safetensors', 'describer/qwen2.5-vl-7b-instruct/model-00005-of-00005.safetensors', 1089994880, '0c859795ad3a627a9b95bcb762e059d5b768a4a36fdd4affeff269d93fdecc67'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:model.safetensors.index.json', 'describer/qwen2.5-vl-7b-instruct/model.safetensors.index.json', 57619, '73b333b0b16e5286ddba615d2caebcd495cf7e616f52eb217a81781393d79de9'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:preprocessor_config.json', 'describer/qwen2.5-vl-7b-instruct/preprocessor_config.json', 350, 'f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:tokenizer.json', 'describer/qwen2.5-vl-7b-instruct/tokenizer.json', 7031645, 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:tokenizer_config.json', 'describer/qwen2.5-vl-7b-instruct/tokenizer_config.json', 5702, '4abd3520120e266da84c0864fee064d1fb10806f02225911a47253dd38dc5f56'),
    ('qwen2.5-vl', 'describer', None, 'apache-2.0', 'Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5:vocab.json', 'describer/qwen2.5-vl-7b-instruct/vocab.json', 2776833, 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'),
    ('minimax-h3', 'h3', None, 'minimax-h3', 'Comfy-Org/MiniMax-H3@54472221d27c695fca9621b0c3ed52f06c4af707:diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'h3/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors', 20970379616, '9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779'),
    ('minimax-h3', 'h3', None, 'minimax-h3', 'Comfy-Org/MiniMax-H3@54472221d27c695fca9621b0c3ed52f06c4af707:text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'h3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 15687142551, '35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6'),
    ('minimax-h3', 'h3', None, 'minimax-h3', 'Comfy-Org/MiniMax-H3@54472221d27c695fca9621b0c3ed52f06c4af707:vae/minimax_h3_video_vae_int8_convrot.safetensors', 'h3/vae/minimax_h3_video_vae_int8_convrot.safetensors', 2811065184, '52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e'),
    ('minimax-h3', 'h3', None, 'minimax-h3', 'Comfy-Org/MiniMax-H3@54472221d27c695fca9621b0c3ed52f06c4af707:vae/minimax_h3_audio_vae_fp32.safetensors', 'h3/vae/minimax_h3_audio_vae_fp32.safetensors', 605254808, '8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48'),
    ('minimax-h3', 'h3', None, 'minimax-h3', 'akatz-ai/MiniMax-H3-Character-Swap-LoRA@96ea3407b230b6f7df71c58c70e31df96bd7e533:h3_character_swap_v1.5.safetensors', 'h3/loras/h3_character_swap_v1.5.safetensors', 155110792, '8f2c3c3eba161df8d0c6099598b4730f641f5ce01e04cd16d2427cff8a35218f'),
    ('minimax-h3', 'h3', None, 'minimax-h3', 'lightx2v/Minimax-h3-Turbo@30f485ed874aa44c8183348ad361c5eabad318da:minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors', 'h3/loras/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors', 1956193000, '08cfe946033af7d27719b964b6e0a0e50c32138daabbd6ce4137e23df6bf9980'),
]]
ROWS += [Row(*row) for row in [
    ('insightface', 'challenge', None, 'insightface', 'buffalo_l.zip:det_10g.onnx', 'face/models/buffalo_l/det_10g.onnx', 16923827,
     '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91'),
    ('insightface', 'challenge', None, 'insightface', 'buffalo_l.zip:w600k_r50.onnx', 'face/models/buffalo_l/w600k_r50.onnx', 174383860,
     '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'),
    ('insightface', 'challenge', None, 'insightface',
     'https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx', 'face/inswapper_128.onnx',
     554253681, 'e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af'),
    ('smpl-x', 'challenge', None, 'smpl-x', 'user:models_smplx_v1_1.zip:smplx/SMPLX_NEUTRAL.npz',
     'motion/body_models/smplx/SMPLX_NEUTRAL.npz', 108752058, '376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992'),
    ('yolo11', 'joints', None, 'agpl-3.0', 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt',
     'motion/yolo/yolo11x.pt', 114636239, '7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24'),
    ('yolo11', 'joints', None, 'agpl-3.0', 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt',
     'motion/yolo/yolo11m.pt', 40684120, 'd5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95'),
    # converted by ExMR's own downloader (tools/download_models.py --only mano); the result differs by Python version
    ('mano', 'joints', None, 'mano', 'user:mano_v1_2.zip:MANO_RIGHT.pkl', 'motion/hamer/data/mano/MANO_RIGHT.pkl', None, None),
]]
ARCHIVES = {  # archive -> (URL, bytes, SHA-256); only the listed members are kept
    'buffalo_l.zip': ('https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip', 288621354,
                      '80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f'),
}
REGISTRATION = {  # zips only their owners hand out, after a registration on their website
    'models_smplx_v1_1.zip': ('SMPL-X', 'https://smpl-x.is.tue.mpg.de/register.php',
                              'https://smpl-x.is.tue.mpg.de/download.php', 'SMPL-X v1.1 (NPZ+PKL)'),
    'mano_v1_2.zip': ('MANO', 'https://mano.is.tue.mpg.de/register.php', 'https://mano.is.tue.mpg.de/download.php',
                      'Models & Code (mano_v1_2.zip)'),
}
GATED = {'facebook/sam3', 'facebook/sam-3d-body-dinov3'}

# id -> (name, link, commercial use, what to know). Read the licence itself: these lines are only pointers.
LICENCES = {
    'mit': ('MIT License', 'https://opensource.org/license/mit', 'yes',
            'Keep the copyright and licence notice.'),
    'apache-2.0': ('Apache License 2.0', 'https://www.apache.org/licenses/LICENSE-2.0', 'yes', 'Keep the licence and the NOTICE files.'),
    'sam3': ('SAM License (Meta), SAM 3', 'https://huggingface.co/facebook/sam3/blob/main/LICENSE', 'yes, under its terms',
             'Gated on Hugging Face (request access). Trade-control and acceptable-use terms apply.'),
    'sam-3d-body': ('SAM License (Meta), SAM 3D Body', 'https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/LICENSE',
                    'yes, under its terms', 'Gated on Hugging Face (request access).'),
    'dinov3': ('DINOv3 License (Meta)', 'https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md', 'yes, under its terms',
               'Not downloaded here: SAM 3D Body fetches the DINOv3 code from GitHub on its first use.'),
    'sapiens2': ('Sapiens2 License (Meta)', 'https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md',
                 'yes, under its terms', 'Its use restrictions include biometric processing and deepfakes impersonating a person.'),
    'exmr': ('ExMR licence (GVHMR terms, Leibniz University Hannover / Zhejiang University)',
             'https://github.com/twehrbein/ExMR/blob/main/LICENSE', 'no',
             'Educational, research and non-profit use only; modifications must be open source.'),
    'agpl-3.0': ('GNU AGPL-3.0 (Ultralytics YOLO11 weights)', 'https://github.com/ultralytics/ultralytics/blob/main/LICENSE',
                 'yes, under AGPL-3.0 (or an Ultralytics enterprise licence)', 'Copyleft: share the source of what you build on it.'),
    'insightface': ('InsightFace pretrained models (buffalo_l, inswapper_128)', 'https://github.com/deepinsight/insightface#license',
                    'no', 'Non-commercial research purposes only.'),
    'smpl-x': ('SMPL-X Model License (Max Planck)', 'https://smpl-x.is.tue.mpg.de/modellicense.html', 'no',
               'Non-commercial research, education or artistic projects; register and download the zip yourself.'),
    'mano': ('MANO License (Max Planck)', 'https://mano.is.tue.mpg.de/license.html', 'no',
             'Non-commercial research; register and download the zip yourself.'),
    'minimax-h3': ('MiniMax H3 Community License', 'https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE',
                   'with conditions (Section IV)',
                   'EXCLUDED TERRITORIES (Sections I.5 and V.4): you may not use MiniMax H3, nor use, distribute or display '
                   'anything it makes, in the European Union, the United Kingdom, the Republic of Korea or the United States '
                   'without separate written authorization from MiniMax.'),
}
ALSO = {'sam-3d-body': ['dinov3']}  # licences that come with another one (code fetched at run time)
LEVELS = ('challenge', 'joints', 'relight', 'describer', 'h3')


def gb(n):
    return f'{n / 1e9:.2f} GB'


def install_root(text=None):
    return Path(text or os.environ.get('DANCE3D_HOME') or REPO / '.dance3d').expanduser().absolute()


def levels_of(args):
    chosen = set()
    if args.all:
        chosen |= {'challenge', 'joints', 'relight', 'describer'}
    if args.joints:
        chosen |= {'challenge', 'joints'}
    if args.challenge or not (args.joints or args.all):
        chosen.add('challenge')
    if args.h3:
        chosen.add('h3')
    for extra in args.extra or []:
        chosen.add(extra)
    return chosen


def card_memory_gib():
    """Memory of the first visible graphics card in GiB, from nvidia-smi; None when it cannot be read."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip()
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [line.split(',') for line in out.strip().splitlines() if ',' in line]
    for index, memory in rows:
        if not visible or index.strip() == visible:
            return float(memory) / 1024
    return float(rows[0][1]) / 1024 if rows else None


def choose_tier(vram):
    """The SCAIL-2 tier (GB) for --vram: the largest tier not larger than the card (as scail_tiers.choose)."""
    if vram == 'auto':
        memory = card_memory_gib()
        if memory is None:
            raise SystemExit('The memory of the graphics card could not be read (nvidia-smi): give --vram 8, 12, 16 or 24.')
        wanted = memory
    else:
        wanted = float(vram)
    fitting = [tier for tier in TIERS if tier <= wanted + 0.5]
    if not fitting:
        raise SystemExit(f'SCAIL-2 needs a graphics card with at least {TIERS[0]} GB (asked for {wanted:g} GB).')
    return fitting[-1]


def check_tier_table():
    """When the programs' own tier table is there, it must name the same files as this table."""
    sys.path.insert(0, str(REPO / 'programs/duo-scene'))
    try:
        import scail_tiers
    except ImportError:
        return
    finally:
        sys.path.pop(0)
    ours = {row.dest for row in ROWS if row.group in ('scail-2', 'wan-2.1')}
    for tier in TIERS:
        for folder, name in scail_tiers.model_files(scail_tiers.choose(tier)):
            if f'comfy/{folder}/{name}' not in ours:
                raise SystemExit(f'install/get_models.py does not know comfy/{folder}/{name} (tier {tier} GB of '
                                 'programs/duo-scene/scail_tiers.py): the two tables disagree.')


def selection(levels, tier):
    return [row for row in ROWS if row.level in levels and (row.tier is None or row.tier == tier)]


# ---------- local state ----------

def sha256_of(path, label=None):
    digest = hashlib.sha256()
    size = path.stat().st_size
    done, shown = 0, time.time()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 23), b''):
            digest.update(chunk)
            done += len(chunk)
            if label and size > 2e9 and time.time() - shown > 15:
                print(f'    checking {label}: {done * 100 // size} %', flush=True)
                shown = time.time()
    return digest.hexdigest()


# --check --config: where a config (config.local.json) says each file is; prefix of the table's file -> key
CONFIG_PLACES = [('motion/body_models/smplx/SMPLX_NEUTRAL.npz', 'smplx'), ('motion/yolo/yolo11x.pt', 'models.detector'),
                 ('sam3/sam3.pt', 'models.sam3'), ('moge/model.pt', 'models.moge'), ('comfy/', 'models.comfy'),
                 ('motion/', 'models.motion'), ('face/', 'models.face'), ('h3/', 'models.h3'),
                 ('describer/qwen2.5-vl-7b-instruct/', 'models.describer')]
_PLACES = {}


def use_config(file):
    """Check the files where the config says they are, instead of under <root>/models."""
    config = json.loads(Path(file).read_text(encoding='utf-8'))
    for prefix, key in CONFIG_PLACES:
        value = config
        for part in key.split('.'):
            value = value.get(part) if isinstance(value, dict) else None
        if value:
            path = Path(str(value)).expanduser()
            _PLACES[prefix] = path if path.is_absolute() else Path(file).parent / path


def target_of(row, models):
    for prefix, _ in CONFIG_PLACES:
        if row.dest.startswith(prefix) and prefix in _PLACES:
            return _PLACES[prefix] if not prefix.endswith('/') else _PLACES[prefix] / row.dest[len(prefix):]
    return models / row.dest


def status(row, models, verify):
    path = target_of(row, models)
    if not path.exists():
        return 'broken link' if path.is_symlink() else 'missing'
    if row.size is not None and path.stat().st_size != row.size:
        return f'wrong size ({path.stat().st_size} bytes)'
    if verify and row.sha256 and sha256_of(path, row.dest) != row.sha256:
        return 'wrong SHA-256'
    return 'ok'


# ---------- licences ----------

def accepted_file(root):
    return root / 'licences-accepted.json'


def ask_licences(rows, root, given):
    """The licences of `rows` the user accepts (asked once per licence; remembered). Returns the accepted ids."""
    file = accepted_file(root)
    try:
        accepted = json.loads(file.read_text()) if file.is_file() else {}
    except ValueError:
        accepted = {}
    needed = []
    for row in rows:
        for licence in [row.licence] + ALSO.get(row.licence, []):
            if licence not in needed:
                needed.append(licence)
    changed = False
    for licence in needed:
        if licence in accepted:
            continue
        name, link, commercial, note = LICENCES[licence]
        groups = sorted({row.group for row in rows if row.licence == licence}) or ['SAM 3D Body (DINOv3 backbone)']
        if licence in given:
            ok = True
        elif sys.stdin.isatty():
            print(f'\nLicence: {name}\n  used by: {", ".join(groups)}\n  read it: {link}\n  commercial use: {commercial}\n  {note}')
            answer = input('  Type yes to accept these terms (anything else skips these files): ').strip().lower()
            ok = answer == 'yes'
            if ok and licence == 'minimax-h3':
                answer = input('  Type yes to confirm that you will not use MiniMax H3, nor use, distribute or display its '
                               'outputs, in the EU, the UK, the Republic of Korea or the USA: ').strip().lower()
                ok = answer == 'yes'
        else:
            print(f'\nNot accepted: {name} ({link}), used by {", ".join(groups)}.\n'
                  f'  Read it, then run again with --accept {licence}' + (' (this also confirms the territory clause)'
                                                                         if licence == 'minimax-h3' else ''))
            ok = False
        if ok:
            accepted[licence] = dict(name=name, link=link, accepted_at=datetime.datetime.now().isoformat(timespec='seconds'))
            changed = True
    if changed:
        root.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(accepted, indent=1) + '\n')
    return set(accepted)


# ---------- network ----------

def hf_token():
    """The Hugging Face token (HF_TOKEN, or the one `hf auth login` saved). Never printed."""
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        return token.strip()
    home = Path(os.environ.get('HF_HOME') or Path.home() / '.cache/huggingface')
    file = home / 'token'
    return file.read_text().strip() if file.is_file() else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def open_url(url, headers=None, method='GET', timeout=60):
    """Open `url`, following redirects by hand: the token goes only to huggingface.co, never to the storage hosts."""
    token = hf_token()
    for _ in range(10):
        sent = {'User-Agent': USER_AGENT, **(headers or {})}
        host = urllib.parse.urlsplit(url).hostname or ''
        if token and host == 'huggingface.co':  # not its storage hosts (cdn-lfs.huggingface.co, *.hf.co)
            sent['Authorization'] = f'Bearer {token}'
        try:
            return _OPENER.open(urllib.request.Request(url, headers=sent, method=method), timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code in (301, 302, 303, 307, 308) and error.headers.get('Location'):
                url = urllib.parse.urljoin(url, error.headers['Location'])
                continue
            raise
    raise RuntimeError(f'too many redirects: {url}')


def source_url(row_source):
    """https URL of a table source."""
    if row_source.startswith('https://'):
        return row_source
    repo, path = row_source.split(':', 1)
    repo, revision = repo.split('@', 1)
    return f'https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(path)}'


def gated_text(repo):
    return (f'{repo} is gated on Hugging Face. 1) Log in at https://huggingface.co, open https://huggingface.co/{repo} and '
            'request access (Meta approves by hand; it can take a while). 2) Make a token with read access at '
            'https://huggingface.co/settings/tokens. 3) export HF_TOKEN=<your token> (or run `hf auth login` once), then run '
            'this again. The token is only sent to huggingface.co.')


def download(url, target, size, sha256, label):
    """Download to target (resuming target.part), check size and SHA-256, then move into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    for attempt in range(1, 9):
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have > size:
            part.unlink()
            have = 0
        if size is not None and have == size:
            break
        try:
            response = open_url(url, {'Range': f'bytes={have}-'} if have else None)
        except urllib.error.HTTPError as error:
            if error.code == 416 and have:
                break  # already complete
            raise
        except (urllib.error.URLError, socket.timeout, ConnectionError) as error:
            print(f'    network problem ({error}); trying again ({attempt}/8)', flush=True)
            time.sleep(min(60, 5 * attempt))
            continue
        if have and response.status == 200:  # the server ignored the range: start again
            have = 0
        total = size or int(response.headers.get('Content-Length') or 0) + have
        done, shown, started = have, 0.0, time.time()
        try:
            with response, open(part, 'ab' if have else 'wb') as handle:
                for chunk in iter(lambda: response.read(1 << 20), b''):
                    handle.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - shown > (2 if sys.stdout.isatty() else 30):
                        speed = (done - have) / max(now - started, 1e-6) / 1e6
                        line = f'    {label}: {done / 1e9:.2f} / {total / 1e9:.2f} GB  {speed:.0f} MB/s'
                        print(('\r' + line) if sys.stdout.isatty() else line, end='' if sys.stdout.isatty() else '\n', flush=True)
                        shown = now
            if sys.stdout.isatty():
                print()
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as error:
            print(f'\n    download interrupted ({error}); going on from {done / 1e9:.2f} GB ({attempt}/8)', flush=True)
            time.sleep(min(60, 5 * attempt))
            continue
        if size is None or part.stat().st_size >= size:
            break
    else:
        raise RuntimeError(f'{label}: the download did not finish after 8 attempts; run again to go on')
    if size is not None and part.stat().st_size != size:
        raise RuntimeError(f'{label}: got {part.stat().st_size} bytes, expected {size}')
    if sha256:
        actual = sha256_of(part, label)
        if actual != sha256:
            part.unlink()
            raise RuntimeError(f'{label}: SHA-256 mismatch (expected {sha256}, got {actual}); the partial file was removed')
    os.replace(part, target)


# ---------- fetching ----------

def find_user_zip(name, extra_dirs, root):
    for place in [root / 'downloads', Path.home() / 'Downloads', REPO] + [Path(d).expanduser() for d in extra_dirs]:
        if (place / name).is_file():
            return place / name
    return None


def fetch_row(row, models, root, args):
    """Get one missing file. Returns None when done, else the reason it is still missing."""
    target = models / row.dest
    if row.source.startswith('user:'):
        _, name, member = row.source.split(':', 2)
        zip_path = getattr(args, 'smplx_zip' if name.startswith('models_smplx') else 'mano_zip') or \
            find_user_zip(name, args.models_from or [], root)
        if zip_path is None:
            what, register, page, label = REGISTRATION[name]
            return (f'{what} needs a registration: register at {register}, accept the licence, download "{label}" '
                    f'from {page} and put {name} into {root / "downloads"} (or pass --{"smplx" if what == "SMPL-X" else "mano"}-zip). '
                    'Then run this again.')
        if row.group == 'mano':
            return convert_mano(Path(zip_path), models, root, args)
        with zipfile.ZipFile(zip_path) as archive:
            names = [n for n in archive.namelist() if n.endswith(member)]
            if not names:
                return f'{zip_path} has no {member}: is it the right zip ({name})?'
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + '.part')
            with archive.open(names[0]) as source, open(part, 'wb') as handle:
                shutil.copyfileobj(source, handle, 1 << 20)
        if row.sha256 and sha256_of(part) != row.sha256:
            part.unlink()
            return f'{member} in {zip_path} is not the tested file (SHA-256 differs)'
        os.replace(part, target)
        return None
    if row.source.split(':', 1)[0] in ARCHIVES:
        name, member = row.source.split(':', 1)
        url, size, sha = ARCHIVES[name]
        archive_path = root / 'downloads' / name
        if not archive_path.is_file():
            download(url, archive_path, size, sha, name)
        with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source:
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + '.part')
            with open(part, 'wb') as handle:
                shutil.copyfileobj(source, handle, 1 << 20)
        if sha256_of(part) != row.sha256:
            part.unlink()
            return f'{member} in {name} is not the tested file (SHA-256 differs)'
        os.replace(part, target)
        if all(status(other, models, False) == 'ok' for other in ROWS if other.source.startswith(name + ':')):
            archive_path.unlink()  # every member is out: the archive is not needed any more
        return None
    repo = row.source.split('@', 1)[0]
    try:
        download(source_url(row.source), target, row.size, row.sha256, row.dest)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403) and repo in GATED:
            return gated_text(repo)
        return f'HTTP {error.code} from {source_url(row.source)}'
    return None


def convert_mano(zip_path, models, root, args):
    """MANO_RIGHT.pkl: ExMR's own downloader extracts it from the zip and converts it (no chumpy needed)."""
    python = Path(args.motion_python or root / 'envs/motion/bin/python')
    exmr = Path(args.exmr or root / 'src/ExMR')
    if not python.exists() or not (exmr / 'tools/download_models.py').is_file():
        return f'MANO is converted with the motion environment: install it first (install.sh --joints), then run this again'
    command = [str(python), '-B', str(exmr / 'tools/download_models.py'), '--checkpoint_dir', str(models / 'motion'),
               '--only', 'mano', '--mano_zip', str(zip_path)]
    result = subprocess.run(command, cwd=str(exmr), env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    return None if result.returncode == 0 else f'ExMR could not convert {zip_path} (exit code {result.returncode})'


def import_from(rows, models, dirs, copy, verify):
    """Link (or copy) the missing files of `rows` from folders that already hold them: same name and size,
    and the same SHA-256 unless --no-verify. Returns the number of files taken."""
    index = {}
    skip = {'.git', '__pycache__', 'node_modules', '.cache'}
    own = models.resolve()
    for folder in dirs:
        folder = Path(folder).expanduser()
        if not folder.is_dir():
            print(f'  not a folder: {folder}')
            continue
        for path, subdirs, files in os.walk(folder):
            subdirs[:] = [d for d in subdirs if d not in skip and Path(path, d).resolve() != own]
            for name in files:
                index.setdefault(name, []).append(Path(path, name))
    taken = 0
    for row in rows:
        target = models / row.dest
        if status(row, models, False) == 'ok':
            continue
        name = Path(row.dest).name
        for candidate in index.get(name, []):
            try:
                if row.size is not None and candidate.stat().st_size != row.size:
                    continue
            except OSError:
                continue
            if row.size is None and not candidate.as_posix().endswith('mano/' + name):
                continue
            if verify and row.sha256 and sha256_of(candidate, str(candidate)) != row.sha256:
                print(f'  same name and size but another SHA-256, not used: {candidate}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.exists():
                target.unlink()
            if copy:
                part = target.with_name(target.name + '.part')
                shutil.copyfile(candidate, part)
                os.replace(part, target)
            else:
                os.symlink(os.path.abspath(candidate), target)
            print(f'  {"copied" if copy else "linked"}  {row.dest}  <-  {candidate}')
            taken += 1
            break
    return taken


# ---------- source check ----------

def verify_sources(rows):
    """Every source still serves the pinned file: Hugging Face metadata at the pinned revision (size, SHA-256;
    small files without an LFS hash are downloaded and hashed), HEAD for other URLs. Downloads no big file."""
    problems = 0
    by_repo = {}
    for row in rows:
        if row.source.startswith(('user:', 'https://')) or row.source.split(':', 1)[0] in ARCHIVES:
            continue
        repo, path = row.source.split(':', 1)
        by_repo.setdefault(repo, []).append((path, row))
    for repo, files in sorted(by_repo.items()):
        name, revision = repo.split('@', 1)
        kind = 'spaces' if name.startswith('spaces/') else 'models'
        api = f'https://huggingface.co/api/{kind}/{name.split("/", 1)[1] if kind == "spaces" else name}/revision/{revision}?blobs=true'
        try:
            with open_url(api) as response:
                info = json.load(response)
        except urllib.error.HTTPError as error:
            print(f'  FAIL  {name}@{revision[:8]}: HTTP {error.code}')
            problems += len(files)
            continue
        siblings = {item['rfilename']: item for item in info.get('siblings', [])}
        gated = info.get('gated')
        for path, row in files:
            item = siblings.get(path)
            if item is None:
                print(f'  FAIL  {name}@{revision[:8]}: {path} is not in this revision')
                problems += 1
                continue
            remote = (item.get('lfs') or {}).get('sha256')
            if remote is None and item.get('size', 0) < 64e6:  # a small git file: download and hash it
                with open_url(source_url(row.source)) as response:
                    remote = hashlib.sha256(response.read()).hexdigest()
            good = item.get('size') == row.size and remote == row.sha256
            problems += not good
            print(f'  {"ok  " if good else "FAIL"}  {name}@{revision[:8]}: {path} {gb(item.get("size") or 0)}'
                  + (' (gated)' if gated else '') + ('' if good else f' upstream size {item.get("size")} sha {remote}'))
    seen = set()
    for row in rows:
        if row.source.startswith('https://'):
            url, size = row.source, row.size
        elif row.source.split(':', 1)[0] in ARCHIVES:
            url, size, _ = ARCHIVES[row.source.split(':', 1)[0]]
        else:
            if row.source.startswith('user:'):
                print(f'  user  {row.source.split(":")[1]}: after registration at {REGISTRATION[row.source.split(":")[1]][1]}')
            continue
        if url in seen:
            continue
        seen.add(url)
        try:
            with open_url(url, method='HEAD') as response:
                length = int(response.headers.get('Content-Length') or -1)
        except urllib.error.HTTPError as error:
            length = f'HTTP {error.code}'
        good = length == size
        problems += not good
        print(f'  {"ok  " if good else "FAIL"}  {url} ({length} bytes, expected {size})')
    return problems


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--challenge', action='store_true', help='files for the challenge video (default)')
    parser.add_argument('--joints', action='store_true', help='also the files for video -> 3D joints')
    parser.add_argument('--all', action='store_true', help='challenge + joints + describer + relight LoRA (not H3)')
    parser.add_argument('--h3', action='store_true', help='also MiniMax H3 (optional backend; see its licence)')
    parser.add_argument('--extra', action='append', choices=['relight', 'describer'], help='one optional extra')
    parser.add_argument('--only', action='append', metavar='GROUP', help='only this group of files (repeatable): ' +
                        ', '.join(sorted({row.group for row in ROWS})))
    parser.add_argument('--vram', default='auto', help='graphics card memory in GB (8, 12, 16, 24) or auto (default)')
    parser.add_argument('--root', help='install root (default: $DANCE3D_HOME or <repo>/.dance3d); models go to <root>/models')
    parser.add_argument('--accept', action='append', default=[], metavar='LICENCE', help='accept a licence without asking: ' +
                        ', '.join(LICENCES))
    parser.add_argument('--check', action='store_true', help='only report which files are there')
    parser.add_argument('--verify', action='store_true', help='with --check: also compare the SHA-256 of every file (slow)')
    parser.add_argument('--config', help='with --check: look for the files where this config (config.local.json) says they are')
    parser.add_argument('--models-from', action='append', metavar='DIR', help='take files from this folder (link; repeatable)')
    parser.add_argument('--copy', action='store_true', help='with --models-from: copy instead of a symbolic link')
    parser.add_argument('--no-verify', action='store_true', help='with --models-from: trust name and size (no SHA-256 check)')
    parser.add_argument('--no-download', action='store_true', help='download nothing (with --models-from: only link)')
    parser.add_argument('--smplx-zip', help='models_smplx_v1_1.zip, if it is not in <root>/downloads or ~/Downloads')
    parser.add_argument('--mano-zip', help='mano_v1_2.zip, if it is not in <root>/downloads or ~/Downloads')
    parser.add_argument('--motion-python', help='Python of the motion environment (default <root>/envs/motion/bin/python)')
    parser.add_argument('--exmr', help='ExMR folder (default <root>/src/ExMR)')
    parser.add_argument('--list', action='store_true', help='print the table of every file and exit')
    parser.add_argument('--verify-sources', action='store_true', help='check that every source still serves the pinned files')
    args = parser.parse_args()
    unknown = [licence for licence in args.accept if licence not in LICENCES]
    if unknown:
        parser.error(f'unknown licence id {", ".join(unknown)}; known: {", ".join(LICENCES)}')
    check_tier_table()
    root = install_root(args.root)
    models = root / 'models'
    levels = levels_of(args)

    if args.list:
        print(f'{"level":9} {"tier":>4}  {"licence":11} {"size":>9}  file (in <root>/models) <- source')
        for row in ROWS:
            size = gb(row.size) if row.size else 'n/a'
            print(f'{row.level:9} {row.tier or "":>4}  {row.licence:11} {size:>9}  {row.dest} <- {row.source}')
        print('\nDownload size per choice (GB):')
        for tier in TIERS:
            base = sum(r.size or 0 for r in selection({'challenge'}, tier))
            joints = sum(r.size or 0 for r in selection({'challenge', 'joints'}, tier))
            print(f'  --vram {tier:2}: --challenge {base / 1e9:6.1f}   --joints {joints / 1e9:6.1f}')
        for level in ('relight', 'describer', 'h3'):
            print(f'  {level:9} adds {sum(r.size or 0 for r in ROWS if r.level == level) / 1e9:6.1f}')
        return 0

    if args.config:
        if not args.check:
            parser.error('--config goes with --check')
        use_config(args.config)
    tier = choose_tier(args.vram)
    rows = selection(levels, tier)
    if args.only:
        unknown = sorted(set(args.only) - {row.group for row in ROWS})
        if unknown:
            parser.error(f'unknown group {", ".join(unknown)}')
        rows = [row for row in rows if row.group in args.only]
    print(f'Models for: {", ".join(level for level in LEVELS if level in levels)}; SCAIL-2 tier {tier} GB; folder {models}')

    if args.verify_sources:
        problems = verify_sources(rows)
        print(f'\n{"All sources serve the pinned files." if not problems else f"{problems} problem(s)."}')
        return 1 if problems else 0

    if not args.check:
        accepted = ask_licences(rows, root, set(args.accept))
        allowed = [row for row in rows if row.licence in accepted]
        skipped = [row for row in rows if row.licence not in accepted]
        if args.models_from:
            print('\nTaking files that are already there:')
            taken = import_from(allowed, models, args.models_from, args.copy, not args.no_verify)
            print(f'  {taken} file(s) taken')
        missing = [row for row in allowed if status(row, models, False) != 'ok']
        if missing and not args.no_download:
            need = sum(row.size or 0 for row in missing) + sum(size for name, (_, size, _) in ARCHIVES.items()
                                                                 if any(row.source.startswith(name + ':') for row in missing))
            models.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(models).free
            if free < need + 2e9:
                print(f'\nNot enough disk space at {models}: {gb(need)} to download, {gb(free)} free (keep 2 GB spare).')
                return 1
            print(f'\nGetting {len(missing)} file(s), {gb(need)}:')
            reasons = {}
            for row in missing:
                print(f'  {row.dest} ({gb(row.size) if row.size else "from your zip"})', flush=True)
                try:
                    reason = fetch_row(row, models, root, args)
                except Exception as error:  # one failed file does not stop the others
                    reason = str(error)
                if reason:
                    reasons[row.dest] = reason
                    print('    not done (why: see the end)')
            if reasons:
                print('\nStill missing:')
                for dest, reason in reasons.items():
                    print(f'  {dest}: {reason}')
        if skipped:
            print('\nSkipped (licence not accepted): ' + ', '.join(sorted({row.dest for row in skipped})))

    print('\nCheck:')
    incomplete = 0
    for row in rows:
        state = status(row, models, args.verify)
        incomplete += state != 'ok'
        size = gb(row.size) if row.size else ''
        shown = target_of(row, models) if _PLACES else row.dest
        print(f'  {"ok  " if state == "ok" else "MISS"}  {size:>9}  {shown}' + ('' if state == 'ok' else f'  ({state})'))
    total = sum(row.size or 0 for row in rows)
    print(f'\n{len(rows) - incomplete} of {len(rows)} files present ({gb(total)} in all for this choice).')
    return 2 if incomplete else 0


if __name__ == '__main__':
    sys.exit(main())
