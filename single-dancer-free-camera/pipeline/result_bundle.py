"""Create actual desktop files and a video showing input, motion guide and final output."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from settings import DESKTOP_RESULTS


def copy_inputs(job, swap_job, guide_job):
    out = job/'outputs'
    sources = {
        '입력_캐릭터.png': swap_job/'outputs/inputs/character.png',
        '3D_춤_가이드.mp4': swap_job/'outputs/inputs/guide.mp4',
        '최종_캐릭터_춤.mp4': swap_job/'outputs/character-swap.mp4',
        'H3_얼굴보정전.mp4': swap_job/'outputs/character-swap-raw.mp4',
        '참고_얼굴.png': swap_job/'outputs/face-reference.png',
        '얼굴_보정전후.jpg': swap_job/'outputs/face-comparison.jpg',
        '얼굴_처리기록.json': swap_job/'outputs/face-report.json',
    }
    if guide_job:
        sources.update({
            '원본_춤_선택구간.mp4': guide_job/'outputs/inputs/motion.mp4',
            'Blender_춤과카메라.blend': guide_job/'outputs/blender/scene.blend',
            '3D_첫프레임.glb': guide_job/'outputs/blender/preview.glb',
            '카메라_경로.json': guide_job/'outputs/blender/camera-path.json',
            '설정.json': guide_job/'outputs/settings.json',
        })
        for p in (guide_job/'outputs/mocap').glob('*.bvh'):
            sources[p.name] = p
        for p in (guide_job/'outputs/mocap').glob('*_amass.npz'):
            sources[p.name] = p
    for name, source in sources.items():
        if source.is_file():
            shutil.copy2(source, out/name)
    report_file = out/'얼굴_처리기록.json'
    face_text = '얼굴 보정 꺼짐.'
    if report_file.is_file():
        data = json.loads(report_file.read_text())
        face_text = (f'얼굴 보정: {data["corrected_frames"]}/{data["frames"]} 프레임. '
                     f'얼굴이 안 보이거나 감지가 불확실한 {data["skipped_frames"]}프레임은 보정하지 않았습니다.')
    (out/'사용안내.txt').write_text(
        '최종_캐릭터_춤.mp4: 선택한 원본 춤 구간의 속도를 유지한 최종 영상.\n'
        '입력부터_결과까지.mp4: 입력 소개 → 원본/이미지/3D/결과 비교 → 최종 결과.\n'
        '얼굴_보정전후.jpg: 입력 얼굴, H3 생성 얼굴, 보정 후 얼굴 비교.\n'
        'Blender_춤과카메라.blend: 움직이는 3D 인체 가이드와 카메라.\n'
        '3D_첫프레임.glb는 정지 미리보기입니다. 최종 캐릭터 영상은 H3가 생성한 2D 영상이며, '
        '캐릭터 외형을 텍스처로 가진 3D 모델을 출력하는 방식은 아닙니다.\n'
        + face_text + '\n'
        '얼굴 확대본은 입력 사진을 잘라낸 것입니다. 작은 얼굴의 잃어버린 세부를 복원한 것은 아닙니다.\n'
        '원본에 보이지 않는 옆/뒤 모습은 모델이 추정하므로 실제 외형과 다를 수 있습니다.\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for data in iter(lambda: source.read(8*1024**2), b''):
            h.update(data)
    return h.hexdigest()


def export_desktop(job):
    if DESKTOP_RESULTS.is_symlink():
        raise RuntimeError('바탕화면 결과 폴더가 바로가기입니다. 실제 폴더 경로가 필요합니다.')
    DESKTOP_RESULTS.mkdir(parents=True, exist_ok=True)
    destination = DESKTOP_RESULTS/job.name
    # A unique job ID means reruns never overwrite earlier results.
    destination.mkdir(exist_ok=False)
    checksums = {}
    for source in sorted((job/'outputs').iterdir()):
        if source.is_file() and source.name != 'completed.json':
            target = destination/source.name
            shutil.copy2(source, target)
            checksum = digest(source)
            if digest(target) != checksum:
                raise RuntimeError('바탕화면 복사 검증 실패: '+target.name)
            checksums[target.name] = checksum
    (destination/'sha256.json').write_text(json.dumps(checksums, ensure_ascii=False, indent=2))
    return destination


def make_video(job):
    import cv2
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    out = job/'outputs'
    reference = Image.open(out/'입력_캐릭터.png').convert('RGB')
    original = out/'원본_춤_선택구간.mp4'
    has_original = original.is_file()
    if not has_original:
        original = out/'3D_춤_가이드.mp4'
    paths = [original, out/'3D_춤_가이드.mp4', out/'최종_캐릭터_춤.mp4']
    captures = [cv2.VideoCapture(str(p)) for p in paths]
    count = int(captures[2].get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0 or not all(cap.isOpened() for cap in captures):
        raise ValueError('통합 영상의 입력 파일을 읽을 수 없습니다.')
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title_font = ImageFont.truetype(font_path, 30, index=1)
    label_font = ImageFont.truetype(font_path, 23, index=1)
    small_font = ImageFont.truetype(font_path, 20, index=1)
    silent = job/'cache/overview-silent.mp4'
    proc = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1280x960', '-r', '24', '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-crf', '19', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        '-threads', '4', str(silent)], stdin=subprocess.PIPE)

    def panel(canvas, im, box, title):
        x, y, w, h = box
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((x, y, x+w, y+h), radius=12, fill='#192338')
        draw.text((x+15, y+8), title, font=label_font, fill='#a9d8ff')
        fitted = ImageOps.contain(im, (w-16, h-52))
        canvas.paste(fitted, (x+(w-fitted.width)//2, y+46+(h-52-fitted.height)//2))

    chapters = ['1 · 입력 영상과 캐릭터 이미지', '2 · 원본 춤 → 3D 동작 → 캐릭터 영상',
                '3 · 최종 캐릭터 춤']
    input_label = '입력 · 원본 춤' if has_original else '입력 · 외부 3D 가이드'
    try:
        for chapter, title in enumerate(chapters):
            for cap in captures:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for frame_index in range(count):
                images = []
                for cap in captures:
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError('통합 영상 입력의 프레임 수가 서로 다릅니다.')
                    images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                canvas = Image.new('RGB', (1280, 960), '#0b1220')
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 14), title, font=title_font, fill='white')
                if chapter == 0:
                    panel(canvas, images[0], (20, 75, 610, 815), input_label)
                    panel(canvas, reference, (650, 75, 610, 815), '입력 · 캐릭터 이미지')
                elif chapter == 1:
                    for im, box, label in zip([images[0], reference, images[1], images[2]],
                            [(20, 75, 610, 395), (650, 75, 610, 395),
                             (20, 490, 610, 400), (650, 490, 610, 400)],
                            [input_label, '입력 · 캐릭터 이미지', '3D · 움직임과 카메라', '출력 · 캐릭터 춤']):
                        panel(canvas, im, box, label)
                else:
                    panel(canvas, images[2], (20, 75, 1240, 815), '생성 결과 · 원본 춤의 속도 유지')
                draw.text((24, 914), f'{frame_index/24:.2f}초 / {count/24:.2f}초   ·   24 fps',
                          font=small_font, fill='#b9c8da')
                proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('통합 영상 인코딩 실패')
    finally:
        for cap in captures:
            cap.release()
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(silent),
        '-stream_loop', '2', '-i', str(paths[2]), '-map', '0:v:0', '-map', '1:a:0?',
        '-c:v', 'copy', '-c:a', 'aac', '-t', str(count*3/24), '-movflags', '+faststart',
        str(out/'입력부터_결과까지.mp4')], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    make_video(parser.parse_args().job)
