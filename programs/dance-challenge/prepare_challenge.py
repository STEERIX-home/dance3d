"""Check the user's files and copy what the run needs into the job.

    prepare_challenge.py --joints RESULT --character PICTURE --output DIR [--music FILE] [--title TEXT]

RESULT is one video's result of the dance-joints program: its folder in the results folder
(<results_dir>/<run>/NN_title, with results.npz and 정보.json), its folder in that program's run
(<run>/outputs/videos/NN) or the results.npz file itself. The music is the sound of the dance video the
joints were taken from (it is in that program's run folder, found from the result's own note or under
runs_dir of the config); --music gives another file instead.

  results.npz        the joints result, copied
  info.json          title, frame rate, length, notes of the joints result
  character_A.png    the picture upright (EXIF turn applied), first image of a multi-image file, RGB
  music.m4a          the sound only (none if the source has no sound)
  inputs.json        what was found and where it came from
The user's own files are never changed.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def run_folders(name, info):
    """Where the dance-joints run called `name` may be: the folder its result note names, then runs_dir of the config."""
    places = [Path(info['run'])] if info.get('run') else []
    try:
        import paths
        places.append(paths.RUNS / name)
    except Exception:  # no config (a tool run by hand): only the note's own folder
        pass
    return places


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def has_sound(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0',
                          str(path)], capture_output=True, text=True)
    return out.returncode == 0 and 'audio' in out.stdout


def find_result(given):
    """(results.npz, info dict, the source video of the joints program or None)."""
    given = Path(given).expanduser()
    if given.is_file() and given.name.endswith('.npz'):
        folder = given.parent
        results = given
    elif given.is_dir():
        folder = given
        results = next((p for p in (given / 'results.npz', given / 'work/joints/results.npz', given / 'joints/results.npz')
                        if p.is_file()), None)
        if results is None:
            raise ValueError(f'이 폴더에서 관절 결과(results.npz)를 찾지 못했습니다: {given}\n'
                             '「춤 관절 정보 만들기」 결과 폴더 안의 영상 폴더(예: 02_L을가져가)를 골라 주세요.')
    else:
        raise ValueError(f'관절 결과가 없습니다: {given}')
    info, source = {}, None
    if (folder / '정보.json').is_file():  # the results folder: <run name>/NN_title
        info = json.loads((folder / '정보.json').read_text())
        if info.get('number') is not None:
            for run in run_folders(folder.parent.name, info):
                source = run / f'outputs/videos/{int(info["number"]):02d}/work/input/source.mp4'
                if source.is_file():
                    break
    else:  # inside the job: outputs/videos/NN[/work/joints]
        video_folder = next((p for p in (folder, folder.parent.parent, folder.parent) if (p / 'about.json').is_file()), None)
        if video_folder is not None:
            info = json.loads((video_folder / 'about.json').read_text())
            source = video_folder / 'work/input/source.mp4'
        report = results.parent / 'report.json'
        if report.is_file():
            info['report'] = json.loads(report.read_text())
    return results, info, source if source and source.is_file() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joints', required=True)
    parser.add_argument('--character', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--music')
    parser.add_argument('--title')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results, info, source = find_result(args.joints)
    data = np.load(results)
    needed = ('smplx_global_transl', 'smplx_global_global_orient', 'smplx_global_body_pose', 'joints_global', 'fps',
              'camera_R_c2w', 'camera_center', 'reliable')
    missing = [key for key in needed if key not in data.files]
    if missing:
        raise ValueError(f'관절 결과의 형식이 다릅니다(없는 항목: {", ".join(missing)}). 「춤 관절 정보」 프로그램의 results.npz가 필요합니다.')
    fps, frames = float(data['fps']), len(data['smplx_global_transl'])
    if frames / fps < 3.0:
        raise ValueError(f'관절 결과가 너무 짧습니다({frames / fps:.1f}초). 3초 이상이 필요합니다.')
    shutil.copy2(results, out / 'results.npz')

    try:
        picture = Image.open(args.character)
        picture.seek(0)  # a multi-image file (MPO, animated WebP): its first image
        picture = ImageOps.exif_transpose(picture)
        if picture.mode in ('RGBA', 'LA', 'P'):  # transparent parts become white, as on a plain drawing sheet
            picture = picture.convert('RGBA')
            sheet = Image.new('RGBA', picture.size, (255, 255, 255, 255))
            picture = Image.alpha_composite(sheet, picture)
        picture = picture.convert('RGB')
    except Exception as error:
        raise ValueError(f'캐릭터 그림을 읽을 수 없습니다: {args.character} ({error})')
    if min(picture.size) < 256:
        raise ValueError(f'캐릭터 그림이 너무 작습니다({picture.size[0]}×{picture.size[1]}). 짧은 변이 256 이상이어야 합니다.')
    picture.save(out / 'character_A.png')

    music = Path(args.music).expanduser() if args.music else source
    sound = None
    if music is not None:
        if not music.is_file():
            raise ValueError(f'음악 파일이 없습니다: {music}')
        if has_sound(music):
            done = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(music), '-vn', '-map', '0:a:0', '-c:a', 'aac', '-b:a', '256k',
                                   str(out / 'music.m4a')], capture_output=True, text=True)
            if done.returncode:
                raise ValueError(f'음악을 읽을 수 없습니다: {music} ({done.stderr.strip()[:200]})')
            sound = 'music.m4a'
        elif args.music:
            raise ValueError(f'이 파일에는 소리가 없습니다: {music}')

    title = (args.title if args.title is not None else info.get('title') or (info.get('youtube') or {}).get('title') or '').strip()
    (out / 'info.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
    (out / 'inputs.json').write_text(json.dumps(dict(
        joints=dict(source=str(results), sha256=sha256(results), fps=fps, frames=frames, seconds=round(frames / fps, 3),
                    flipped_left_right=bool(data['flipped_left_right']) if 'flipped_left_right' in data.files else None),
        character=dict(source=str(args.character), sha256=sha256(args.character), width=picture.size[0], height=picture.size[1],
                       file='character_A.png'),
        music=dict(source=str(music) if music else None, file=sound, from_dance_video=args.music is None),
        title=title), ensure_ascii=False, indent=1))
    print(f'joints {frames} frames at {fps:g} fps ({frames / fps:.1f} s); picture {picture.size[0]}x{picture.size[1]}; '
          f'music: {music if sound else "none"}; title: {title!r}')


if __name__ == '__main__':
    main()
