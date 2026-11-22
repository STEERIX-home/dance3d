"""The instruction text for giving every mannequin of the 3D guide its character (pair_run.py).

The text has the same form for every video and every character: it says which mannequin colour
becomes the character of which picture. What a character looks like is not written by a person:
it is the sentence describe_characters.py wrote from the character's own picture (without it the
text still works, but the video model mixes two characters up more easily). The only things a
user may add are a character's height and other clothes for it.

    pair_prompt.py --characters 2 [--description 1="..."] [--height 1=130] [--outfit 2="red hoodie, jeans"] > prompt.txt
"""
import argparse

COLOURS = ['blue', 'red', 'green', 'yellow']  # mannequin colours of blender_duo_scene.py, in the order of the dancers


def build_prompt(characters=2, closeups=True, heights=None, descriptions=None, outfits=None):
    """characters: how many dancers. Pictures 1..N are the characters' whole pictures in the
    order of the dancers; with closeups, pictures N+1..2N are their head close-ups.
    heights: {dancer number (1-based): centimetres}; descriptions: {number: sentence about the
    character}; outfits: {number: clothes to wear instead of those in the picture}."""
    if not 1 <= characters <= len(COLOURS):
        raise ValueError(f'1 to {len(COLOURS)} characters are supported')
    colours = COLOURS[:characters]
    heights, descriptions, outfits = heights or {}, descriptions or {}, outfits or {}
    many = characters > 1

    def joined(parts):
        return ' and '.join(parts) if len(parts) <= 2 else ', '.join(parts[:-1]) + ', and ' + parts[-1]

    text = [joined([('Replace' if i == 0 else 'replace') + f' the {colour} human mannequin in <Video 1> with the character shown in <Picture {i + 1}>'
                    for i, colour in enumerate(colours)]) + '.']
    if many:
        text.append(joined([('The' if i == 0 else 'the') + f' {colour} mannequin always becomes the character of <Picture {i + 1}>'
                            for i, colour in enumerate(colours)])
                    + '; never swap, merge or duplicate them, and do not add other people.')
    for i, colour in enumerate(colours):
        if descriptions.get(i + 1):
            text.append(f'The character of <Picture {i + 1}> (the {colour} mannequin): {descriptions[i + 1].rstrip(".")}.')
        if closeups:
            text.append(f'<Picture {characters + i + 1}> is a close-up of the same character: use this exact face and hairstyle.')
    if many:
        text.append('Each mannequin is one whole character from head to toe: the face and hair of one character never appear on '
                    'another mannequin, also while a mannequin is hidden behind another one.')
    text.append('Match each reference character\'s face, hairstyle, skin, body proportions, apparent age, outfit, hands and '
                'footwear exactly as in its pictures.')
    text.append('Keep each character in the visual style of its own pictures: a photographed person stays photorealistic, '
                'a drawn character stays a drawing with the same lines and colours.')
    for number, outfit in sorted(outfits.items()):
        text.append(f'The character of <Picture {number}> wears {outfit.strip().rstrip(".")} instead of the clothes in its pictures; '
                    'its face, hair and body stay as in its pictures.')
    for number, centimetres in sorted(heights.items()):
        text.append(f'The character of <Picture {number}> is about {centimetres:.0f} cm tall, as tall as its mannequin.')
    text.append('Follow each mannequin\'s full body motion, hand gestures, foot placement and dance timing exactly, including '
                'every quick shake, every hit and every moving pose' + (', and keep who stands in front of whom.' if many else '.'))
    text.append('The mannequins have blank faces, but the characters do not: give every character lively, natural facial '
                'expressions that keep changing with the dance, such as smiling, mouthing the song and playful looks; never a '
                'frozen face.')
    text.append('Hair and clothes move naturally with the body: fabric swings and folds, hair bounces and sways.')
    text.append('Preserve the camera movement, framing, floor, walls, lights and background of <Video 1>.')
    text.append(('The characters really stand' if many else 'The character really stands') + ' in this room: lit by the same '
                'light as the room, with the feet resting firmly on the floor, and casting the soft shadow on the floor that the '
                'mannequin casts in <Video 1>.')
    text.append('The pictures only define the characters: never show a picture itself, and keep the room of <Video 1> behind the '
                'characters in every frame.')
    text.append('Keep the outfits opaque and securely covering the bodies while dancing.')
    text.append('Maintain the same faces, hairstyles and characters throughout the video.')
    return ' '.join(text)


def numbered(entries, convert=str):
    return {int(number): convert(value) for number, value in (entry.split('=', 1) for entry in entries)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--characters', type=int, default=2)
    parser.add_argument('--no-closeups', action='store_true')
    parser.add_argument('--height', action='append', default=[], help='NUMBER=centimetres')
    parser.add_argument('--description', action='append', default=[], help='NUMBER=sentence about that character')
    parser.add_argument('--outfit', action='append', default=[], help='NUMBER=other clothes for that character')
    args = parser.parse_args()
    print(build_prompt(args.characters, not args.no_closeups, numbered(args.height, float), numbered(args.description),
                       numbered(args.outfit)))


if __name__ == '__main__':
    main()
