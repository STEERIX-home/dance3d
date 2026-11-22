"""Check the user's files and bring them into the forms the pipeline reads.

    prepare_inputs.py --video VIDEO --character PICTURE --character PICTURE --output DIR [--start S] [--seconds S]

  source.mp4          the video as H.264 (the tools cannot read every codec), at its own frame rate
                      up to 60 per second, the long side at most 1920, sound kept if there is any
  character_A.png ..  each picture upright (its EXIF turn applied), first image of a multi-image
                      file, plain RGB
  inputs.json         what was found: sizes, frame rate, length, sound, checksums
The user's own files are never changed.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageOps

LABELS = 'ABCD'


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)],
                         capture_output=True, text=True)
    if out.returncode:
        raise ValueError(f'영상을 읽을 수 없습니다: {path} ({out.stderr.strip()[:200]})')
    return json.loads(out.stdout)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--character', action='append', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start', type=float, default=0.0, help='first second of the video to use')
    parser.add_argument('--seconds', type=float, help='how long a part to use (default: to the end)')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    found = probe(args.video)
    videos = [s for s in found['streams'] if s['codec_type'] == 'video']
    if not videos:
        raise ValueError('이 파일에는 영상이 없습니다.')
    stream = videos[0]
    numerator, denominator = (float(v) for v in stream['avg_frame_rate'].split('/'))
    fps = numerator / denominator
    width, height = int(stream['width']), int(stream['height'])
    if abs(int(stream.get('tags', {}).get('rotate', 0) or 0)) in (90, 270) or any(
            abs(int(side.get('rotation', 0))) in (90, 270) for side in stream.get('side_data_list', [])):
        width, height = height, width  # ffmpeg applies the turn when it decodes
    has_audio = any(s['codec_type'] == 'audio' for s in found['streams'])
    filters = []
    if fps > 60.5:
        fps = 60.0
        filters.append('fps=60')
    if max(width, height) > 1920:
        scale = 1920 / max(width, height)
        width, height = int(round(width * scale / 2)) * 2, int(round(height * scale / 2)) * 2
        filters.append(f'scale={width}:{height}:flags=lanczos')
    elif width % 2 or height % 2:
        width, height = width - width % 2, height - height % 2
        filters.append(f'crop={width}:{height}:0:0')
    command = ['ffmpeg', '-v', 'error', '-y', '-ss', f'{args.start:.3f}']
    if args.seconds:
        command += ['-t', f'{args.seconds:.3f}']
    command += ['-i', str(args.video), '-map', '0:v:0']
    if has_audio:
        command += ['-map', '0:a:0', '-c:a', 'aac', '-b:a', '192k']
    if filters:
        command += ['-vf', ','.join(filters)]
    command += ['-c:v', 'libx264', '-crf', '12', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(out / 'source.mp4')]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise ValueError(f'영상을 변환하지 못했습니다: {done.stderr.strip()[:300]}')
    made = probe(out / 'source.mp4')
    video = [s for s in made['streams'] if s['codec_type'] == 'video'][0]
    frames = int(subprocess.run(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0', '-show_entries',
                                 'stream=nb_read_packets', '-of', 'csv=p=0', str(out / 'source.mp4')],
                                capture_output=True, text=True, check=True).stdout.strip().strip(','))
    duration = frames / fps
    if duration < 2.5:
        raise ValueError(f'영상이 너무 짧습니다({duration:.1f}초). 2.5초 이상이 필요합니다.')
    pictures = []
    for label, path in zip(LABELS, args.character):
        try:
            picture = Image.open(path)
            picture.seek(0)  # a multi-image file (MPO, animated WebP): its first image
            picture = ImageOps.exif_transpose(picture).convert('RGB')
        except Exception as error:
            raise ValueError(f'캐릭터 그림을 읽을 수 없습니다: {path} ({error})')
        if min(picture.size) < 256:
            raise ValueError(f'캐릭터 그림이 너무 작습니다: {path} ({picture.size[0]}×{picture.size[1]}). 짧은 변이 256 이상이어야 합니다.')
        target = out / f'character_{label}.png'
        picture.save(target)
        pictures.append(dict(label=label, source=str(path), sha256=sha256(path), width=picture.size[0], height=picture.size[1],
                             file=target.name))
    (out / 'inputs.json').write_text(json.dumps(dict(
        video=dict(source=str(args.video), sha256=sha256(args.video), start=args.start, seconds=args.seconds, file='source.mp4',
                   width=int(video['width']), height=int(video['height']), fps=fps, frames=frames, duration=duration,
                   has_audio=has_audio, portrait=int(video['height']) >= int(video['width'])),
        characters=pictures), ensure_ascii=False, indent=1))
    print(f'video {video["width"]}x{video["height"]}, {fps:.3f} fps, {frames} frames ({duration:.2f} s), sound: {has_audio}; '
          f'{len(pictures)} character pictures')


if __name__ == '__main__':
    main()
