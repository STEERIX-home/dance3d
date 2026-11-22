"""Write, for each character picture, the sentence that says what the character looks like.

The video model keeps two characters apart much better when the instruction text describes each
of them (seen 11-13: with only "the character of <Picture 1>" it put one character's head on the
other one's body while that one was hidden, and left a mannequin unreplaced). Nobody should have
to write that description, so a small picture-reading model writes it from the picture alone.

    describe_characters.py --model DIR --picture A=character_A.png --picture B=... --output DIR
                           [--closeup A=head_A.png ...]

Give it the pictures as the user supplied them: a cut-out on plain grey reads as "a 3D render".
With --closeup (a crop of the same picture around the head, make_reference.py writes it; used for SCAIL-2) the
sentence names the hair colour but not the hairstyle: the hairstyle is left to the reference pictures. This model
misreads hairstyles (11-22: reze.webp's small low bun became "a ponytail", makima.webp's single braid "pigtails",
and asked about the hairstyle with the close-up "twin tails" with "a red ribbon"), and SCAIL-2 draws what the
sentence says wherever the picture does not show the back of the head: reze got long ponytails and braids.

Runs on the GPU for a few seconds per picture. With --gpu-gb (a card smaller than the model, about 16 GB) only
as many of the model's layers stay on the card as fit; the others wait in the computer's memory and are
brought to the card one at a time while the text is written (the same sums, only slower: about half a
minute a picture). Outputs: DIR/description_<LABEL>.txt
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def _bytes(module):
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))


def _decoder_layers(model):
    """The model's list of decoder layers (the largest list of modules named ...DecoderLayer)."""
    lists = [module for module in model.modules() if isinstance(module, torch.nn.ModuleList) and len(module)
             and all(type(child).__name__.endswith('DecoderLayer') for child in module)]
    if not lists:
        raise RuntimeError('The picture-reading model has no list of decoder layers to bring in one at a time')
    return max(lists, key=_bytes)


def load_on_card(model, gpu_gb):
    """Put the model on the card; with gpu_gb, keep only what fits in it there (1.5 GB left for the picture's
    working memory) and bring the other decoder layers in for their turn (from pinned memory)."""
    if gpu_gb is None:
        return model.to('cuda')
    layers = _decoder_layers(model)
    budget = (gpu_gb - 1.5) * 2**30
    used = _bytes(model) - sum(_bytes(layer) for layer in layers)
    streamed = []
    for index, layer in enumerate(layers):
        if used + _bytes(layer) <= budget:
            used += _bytes(layer)
        else:
            streamed.append(index)
    away = {index: layers[index] for index in streamed}
    for index in streamed:  # out of the model while the rest moves to the card
        layers[index] = torch.nn.Identity()
    model.to('cuda')
    for index, layer in away.items():
        layers[index] = layer
        tensors = list(layer.parameters()) + list(layer.buffers())
        copies = [(tensor, tensor.data.pin_memory()) for tensor in tensors]
        for tensor, copy in copies:
            tensor.data = copy

        def bring(module, args, copies=copies):
            for tensor, copy in copies:
                tensor.data = copy.to('cuda', non_blocking=True)

        def send_back(module, args, output, copies=copies):
            for tensor, copy in copies:
                tensor.data = copy
        layer.register_forward_pre_hook(bring)
        layer.register_forward_hook(send_back)
    print(f'{len(layers) - len(streamed)} of {len(layers)} decoder layers on the card ({used / 2**30:.1f} GiB), '
          f'{len(streamed)} brought in for their turn', flush=True)
    return model

QUESTION = ('Describe the character in this picture for a video prompt, in one sentence of at most 55 words. Say in this order: '
            'who it is (for example a young child, a woman, a man, an anime girl, a robot), the hair colour and hairstyle, the eye '
            'colour if it is clear, each piece of outer clothing with its colour from top to bottom, the footwear or bare feet, and '
            'the visual style (a photograph of a real person, a flat cel-shaded anime drawing, a 3D render, ...). Describe only '
            'what is visible. Never mention underwear, the background, the pose or the expression. Answer with the sentence only.')
QUESTION_WITH_CLOSEUP = (
    'The first picture shows a character, the second picture is a close-up of the same character\'s head. Describe the '
    'character for a video prompt, in one sentence of at most 55 words. Say in this order: who it is (for example a young '
    'child, a woman, a man, an anime girl, a robot); the hair colour only (never how the hair is cut, styled or tied: the '
    'pictures show that); the eye colour if it is clear; each piece of outer clothing with its colour from top to bottom, '
    'saying where a ribbon or tie is worn; the legwear; the footwear or bare feet; and the visual style (a photograph of a '
    'real person, a flat cel-shaded anime drawing, a 3D render, ...). Describe only what is visible. Never mention hair '
    'accessories, underwear, the background, the pose or the expression. Answer with the sentence only.')
UNWANTED = ('underwear', 'panties', 'panty', 'lingerie', 'bra ', 'bra,', 'briefs', 'thong')  # never part of a dance outfit's description


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='folder of the picture-reading model (Qwen2.5-VL-7B-Instruct)')
    parser.add_argument('--picture', action='append', required=True, help='LABEL=picture of that character')
    parser.add_argument('--closeup', action='append', default=[], help='LABEL=close-up of that character\'s head (a crop '
                        'of the same picture)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-gb', type=float, help='graphics memory the model may hold (default: all of it on the card)')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = load_on_card(AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).eval(), args.gpu_gb)
    closeups = dict(entry.split('=', 1) for entry in args.closeup)
    for entry in args.picture:
        label, picture = entry.split('=', 1)
        image = Image.open(picture).convert('RGB')
        image.thumbnail((1024, 1024))
        content = [dict(type='image', image=image), dict(type='text', text=QUESTION)]
        if label in closeups:
            head = Image.open(closeups[label]).convert('RGB')
            head.thumbnail((768, 768))
            content = [dict(type='image', image=image), dict(type='image', image=head), dict(type='text', text=QUESTION_WITH_CLOSEUP)]
        messages = [dict(role='user', content=content)]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors='pt').to(model.device)
        with torch.inference_mode():
            made = model.generate(**inputs, max_new_tokens=110, do_sample=False)
        text = processor.batch_decode(made[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        text = ' '.join(text.split()).strip().strip('"')
        text = ', '.join(part for part in text.split(', ') if not any(word in part.lower() + ' ' for word in UNWANTED))
        if len(text.split()) < 6:
            raise RuntimeError(f'No usable description of character {label}: {text!r}')
        if not text.endswith('.'):
            text += '.'
        (out / f'description_{label}.txt').write_text(text)
        print(f'{label}: {text}', flush=True)


if __name__ == '__main__':
    main()
