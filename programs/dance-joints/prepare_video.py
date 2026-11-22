"""Bring a downloaded or given video into the form the motion tools read.

    prepare_video.py --video VIDEO --output DIR [--flip]

  source.mp4   H.264 (the tools cannot read every codec) at the video's own frame rate up to 60 per
               second, the long side at most 1920, both sides divisible by 4; with --flip turned
               left-right, which brings a mirrored practice video back to the dance as it is danced
  inputs.json  what was found: size, frame rate, length, sound, checksum
The given file is never changed.
"""
import argparse
import json
import subprocess
from pathlib import Path

from prepare_inputs import probe, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--flip', action='store_true', help='turn the picture left-right')
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
    # sizes divisible by 4: the tools also make half-size pictures, and a video encoder takes even sizes only
    if max(width, height) > 1920:
        scale = 1920 / max(width, height)
        width, height = int(round(width * scale / 4)) * 4, int(round(height * scale / 4)) * 4
        filters.append(f'scale={width}:{height}:flags=lanczos')
    elif width % 4 or height % 4:
        width, height = width - width % 4, height - height % 4
        filters.append(f'crop={width}:{height}:0:0')
    if args.flip:
        filters.append('hflip')
    command = ['ffmpeg', '-v', 'error', '-y', '-i', str(args.video), '-map', '0:v:0']
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
    (out / 'inputs.json').write_text(json.dumps(dict(
        source=str(args.video), sha256=sha256(args.video), file='source.mp4', flipped_left_right=args.flip,
        width=int(video['width']), height=int(video['height']), fps=fps, frames=frames, duration=duration,
        has_audio=has_audio), ensure_ascii=False, indent=1))
    print(f'video {video["width"]}x{video["height"]}, {fps:.3f} fps, {frames} frames ({duration:.2f} s), sound: {has_audio}'
          + ('; turned left-right' if args.flip else ''))


if __name__ == '__main__':
    main()
