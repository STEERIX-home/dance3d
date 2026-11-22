"""SCAIL-2 settings by the graphics card's memory: the ONE place that decides which model file, text
encoder placement, picture size and ComfyUI memory settings a run uses.

    tier = choose(vram, card_gib)   vram: 'auto' (the card's own memory) or a number of GB (8, 12, 16, 24)

A card is put in the largest tier that is not larger than it (a 10 GB card gets the 8 GB tier).
When a size smaller than the card is asked for (e.g. --vram 12 on a 32 GB card), ComfyUI runs as on a card
of that size (cap_gb: comfy_capped.py makes ComfyUI see a card of that size, OS_SHARE_GB of it taken by a
desktop, and gives its dynamic-VRAM library the same budget), whatever other programs hold on the real card;
that is how a smaller card is tested on a larger one, and how a run leaves the rest of a card to others.
The pipeline measures what the run really holds (run_support.MemoryWatch).
`measured` in each row says what was actually run with it; None = not run yet (untested). It is a note: the
pipeline leaves it out of a run's tier.json.

Model files (all in the ComfyUI models folder of the config, models.comfy):
  diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors   Comfy-Org/SCAIL-2 (licence: MIT, zai-org/SCAIL-2)
  unet/SCAIL-2-Q*_*.gguf                                         realrebelai/SCAIL-2_GGUF (licence: MIT); needs the
                                                                 ComfyUI-GGUF custom node (city96, Apache-2.0)
  loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors                 Comfy-Org/SCAIL-2 (MIT)
  loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors   Kijai/WanVideo_comfy (lightx2v, Apache-2.0)
  text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors           Comfy-Org/Wan_2.1_ComfyUI_repackaged (Apache-2.0)
  clip_vision/clip_vision_h.safetensors                          Comfy-Org/Wan_2.1_ComfyUI_repackaged
  vae/Wan2_1_VAE_bf16.safetensors                                Kijai/WanVideo_comfy (Wan2.1 VAE, Apache-2.0)
"""

COMMON = dict(dpo='wan2.1_SCAIL_2_DPO_lora_bf16.safetensors',
              distill='lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors',
              relight='wan2.1_SCAIL_2_relight_lora_bf16.safetensors',
              text='umt5_xxl_fp8_e4m3fn_scaled.safetensors', vae='Wan2_1_VAE_bf16.safetensors',
              vision='clip_vision_h.safetensors')

# gb: the card size the row is for. unet: the diffusion model file; loader 'gguf' (ComfyUI-GGUF, folder
# unet/) or 'native' (folder diffusion_models/). size: generation width x height of a portrait video (both
# divisible by 32: the pose video is encoded at half size); the finished video is enlarged to the 1080x1920
# render. text_device: where the text encoder runs ('cpu' keeps the card for the video model; it runs once a
# chunk, a few seconds). comfy_flags: extra ComfyUI options. describer_gb: graphics memory the picture-reading
# model may hold (the rest of it waits in the computer's memory and is brought in layer by layer).
TIERS = [
    dict(gb=8, unet='SCAIL-2-Q3_K_M.gguf', loader='gguf', size=(480, 832), text_device='cpu', comfy_flags=[], describer_gb=5.0,
         measured=None),
    dict(gb=12, unet='SCAIL-2-Q4_K_M.gguf', loader='gguf', size=(512, 896), text_device='cpu', comfy_flags=[], describer_gb=8.5,
         measured='2026-11-22, 32 GB card capped to 12 GB, three untouched runs: dance 02 + reze.webp (14.4 s, 10 shots) 26 and '
                  '32.6 min, dance 05 + makima.webp (18.6 s, 11 shots) 50 min (14.6 min of it remakes asked by two false alarms, '
                  'fixed since); peak of the run\'s processes 9,292-10,028 MiB (ComfyUI), describer up to 8,646 MiB; 41-frame '
                  'chunk 84-92 s with two reference views, 99-108 s with three'),
    dict(gb=16, unet='SCAIL-2-Q6_K.gguf', loader='gguf', size=(576, 1024), text_device='default', comfy_flags=[], describer_gb=12.0,
         measured=None),
    dict(gb=24, unet='wan2.1_14B_SCAIL_2_fp8_scaled.safetensors', loader='native', size=(704, 1248), text_device='default',
         comfy_flags=[], describer_gb=20.0, measured=None),
]
OS_SHARE_GB = 0.6  # what a desktop and the driver keep of a real card's memory besides the program


def choose(vram, card_gib=None):
    """The tier for `vram` ('auto' or GB). card_gib: the card's real memory (GiB), needed for 'auto' and to keep
    the rest of a larger card free. Returns a dict: the tier row with COMMON's files, the cap and the sizes."""
    if vram == 'auto':
        if card_gib is None:
            raise ValueError('그래픽 카드 메모리를 읽지 못했습니다: --vram 8, 12, 16, 24 중 하나를 주세요. / The memory of the graphics '
                             'card could not be read: give --vram 8, 12, 16 or 24.')
        wanted = card_gib
    else:
        wanted = float(vram)
    rows = [row for row in TIERS if row['gb'] <= wanted + 0.5]
    if not rows:
        raise ValueError(f'SCAIL-2 에는 그래픽 카드 메모리가 {TIERS[0]["gb"]} GB 이상 필요합니다 ({wanted:g} GB). / SCAIL-2 needs a graphics '
                         f'card with at least {TIERS[0]["gb"]} GB ({wanted:g} GB given).')
    row = dict(rows[-1])
    row.update(COMMON)
    if card_gib is not None and vram != 'auto' and wanted < card_gib - 0.5:
        # a smaller card asked for on a larger one: ComfyUI runs as on a card of `wanted` GB
        # (of which a real card leaves OS_SHARE_GB to the desktop and the driver)
        row['cap_gb'] = round(wanted, 2)
        row['describer_gb'] = min(row['describer_gb'], wanted - 3.5)
    else:
        row['cap_gb'] = None  # the whole card, ComfyUI's own defaults
    row.update(asked=vram, behaves_as_gb=round(wanted, 2), card_gib=None if card_gib is None else round(card_gib, 2))
    return row


def model_files(row):
    """(folder inside the ComfyUI models folder, file name) of every file the tier needs."""
    return [('unet' if row['loader'] == 'gguf' else 'diffusion_models', row['unet']), ('loras', row['dpo']),
            ('loras', row['distill']), ('text_encoders', row['text']), ('vae', row['vae']), ('clip_vision', row['vision'])]


def comfy_options(row):
    """The ComfyUI options of the tier (added to comfy_serve.py's own)."""
    options = list(row.get('comfy_flags') or [])
    if row.get('cap_gb') is not None:
        options += ['--vram-cap-gb', f'{row["cap_gb"]:g}', '--vram-os-share-gb', f'{OS_SHARE_GB:g}']
    return options
