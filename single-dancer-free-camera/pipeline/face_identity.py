"""Isolated CPU worker for explicit, local face reference and identity transfer."""
import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


def engine(root):
    required = ['models/buffalo_l/det_10g.onnx', 'models/buffalo_l/w600k_r50.onnx',
                'inswapper_128.onnx']
    for name in required:
        if not (root / name).is_file():
            raise ValueError('얼굴 보정 모델이 설치되지 않았습니다: ' + name)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    app = FaceAnalysis(name='buffalo_l', root=str(root), allowed_modules=['detection', 'recognition'],
                       providers=['CPUExecutionProvider'], sess_options=opts)
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.6)
    return app, opts


def load_image(path):
    with Image.open(path) as im:
        rgba = ImageOps.exif_transpose(im).convert('RGBA')
        bg = Image.new('RGBA', rgba.size, 'white')
        bg.alpha_composite(rgba)
        return cv2.cvtColor(np.asarray(bg.convert('RGB')), cv2.COLOR_RGB2BGR)


def face_crop(image, face, size=512, margin=1.9):
    x1, y1, x2, y2 = face.bbox
    side = max(x2-x1, y2-y1)*margin
    cx, cy = (x1+x2)/2, (y1+y2)/2 - .05*side
    crop = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).crop(
        (int(cx-side/2), int(cy-side/2), int(cx+side/2), int(cy+side/2)))
    return crop.resize((size, size), Image.Resampling.LANCZOS)


def source_face(app, path):
    image = load_image(path)
    faces = [f for f in app.get(image) if f.det_score >= .65]
    if len(faces) != 1:
        raise ValueError('참고 이미지에서 얼굴을 한 개만 찾을 수 있어야 합니다. '
                         '얼굴이 크게 보이는 사진을 추가해 주세요. 감지된 얼굴: ' + str(len(faces)))
    face = faces[0]
    if min(face.bbox[2:] - face.bbox[:2]) < 32:
        raise ValueError('참고 얼굴이 너무 작습니다. 얼굴이 크게 보이는 사진을 추가해 주세요.')
    return image, face


def detail_reference(image, face):
    """Keep real source detail, crop inside the photo, and never enlarge it."""
    x1, y1, x2, y2 = map(float, face.bbox)
    height, width = image.shape[:2]
    side = int(np.ceil(max(x2-x1, y2-y1)*1.4))
    crop_width, crop_height = min(side, width), min(side, height)
    left = max(0, min(width-crop_width, int((x1+x2-crop_width)/2)))
    top = max(0, min(height-crop_height, int((y1+y2-crop_height)/2 - .08*side)))
    crop = Image.fromarray(cv2.cvtColor(
        image[top:top+crop_height, left:left+crop_width], cv2.COLOR_BGR2RGB))
    crop.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    return crop, [left, top, left+crop_width, top+crop_height]


def prepare(args):
    app, _ = engine(args.models)
    image, face = source_face(app, args.image)
    args.output.mkdir(parents=True, exist_ok=True)
    bounds = None
    if args.detail:
        reference, bounds = detail_reference(image, face)
    else:
        reference = face_crop(image, face)
    reference.save(args.output/'face-reference.png')
    report = dict(source=str(args.image), bbox=face.bbox.tolist(),
                  confidence=float(face.det_score), original_face_pixels=(face.bbox[2:]-face.bbox[:2]).tolist(),
                  reference_size=list(reference.size), crop_bounds=bounds, detail=args.detail,
                  note='Crop of the supplied image; detail mode never enlarges the source crop.')
    (args.output/'face-reference.json').write_text(json.dumps(report, indent=2))


def eligible(face):
    width, height = face.bbox[2:] - face.bbox[:2]
    eyes = np.linalg.norm(face.kps[0] - face.kps[1])
    # Reject ambiguous detections and extreme profiles. No face means no modification.
    return (face.det_score >= .65 and min(width, height) >= 22
            and eyes >= max(6, width*.15) and np.isfinite(face.kps).all())


def comparison(source, before, after, directory):
    canvas = Image.new('RGB', (960, 380), '#101827')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22, index=1)
    for index, (label, crop) in enumerate(zip(('입력 얼굴', 'H3 생성 얼굴', '얼굴 보정 후'),
                                             (source, before, after))):
        draw.text((index*320+20, 14), label, font=font, fill='white')
        canvas.paste(crop.resize((300, 300)), (index*320+10, 65))
    canvas.save(directory/'face-comparison.jpg', quality=95)


def correct(args):
    app, opts = engine(args.models)
    source_image, source = source_face(app, args.image)
    swapper = get_model(str(args.models/'inswapper_128.onnx'),
                        providers=['CPUExecutionProvider'], sess_options=opts)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError('얼굴 보정용 영상을 읽을 수 없습니다.')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(fps-24) > .001:
        raise ValueError('얼굴 보정 영상은 24 fps여야 합니다.')
    args.output.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', '24',
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '17', '-preset', 'fast',
               '-pix_fmt', 'yuv420p', '-threads', '4', '-movflags', '+faststart',
               str(args.output/'face-corrected-silent.mp4')]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    records = []
    best_quality = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces = [f for f in app.get(frame) if eligible(f)]
            record = dict(frame=len(records), corrected=False)
            # This pipeline is single-person. Ambiguous multi-person frames are left intact.
            if len(faces) == 1:
                target = faces[0]
                result = swapper.get(frame, target, source, paste_back=True)
                record.update(corrected=True, confidence=float(target.det_score),
                              bbox=target.bbox.tolist(),
                              before_similarity=float(np.dot(source.normed_embedding, target.normed_embedding)))
                app.models['recognition'].get(result, target)
                record['after_similarity'] = float(np.dot(source.normed_embedding, target.normed_embedding))
                quality = float(np.prod(target.bbox[2:]-target.bbox[:2])*target.det_score)
                if quality > best_quality:
                    comparison(face_crop(source_image, source), face_crop(frame, target),
                               face_crop(result, target), args.output)
                    best_quality = quality
            else:
                result = frame
                record['reason'] = 'No unambiguous visible face (back view, occlusion, profile or small face).'
            encoder.stdin.write(result.tobytes())
            records.append(record)
            if len(records) % 24 == 0:
                print('Face correction frames:', len(records), flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('얼굴 보정 영상 저장에 실패했습니다.')
    finally:
        cap.release()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
    changed = [r for r in records if r['corrected']]
    if not records:
        raise ValueError('처리 가능한 영상 프레임이 없습니다.')
    report = dict(frames=len(records), corrected_frames=len(changed), skipped_frames=len(records)-len(changed),
                  provider='CPUExecutionProvider', records=records,
                  mean_before_similarity=float(np.mean([r['before_similarity'] for r in changed])) if changed else None,
                  mean_after_similarity=float(np.mean([r['after_similarity'] for r in changed])) if changed else None,
                  note='Similarity is an internal comparison, not a guarantee of identity or visual quality.')
    (args.output/'face-report.json').write_text(json.dumps(report, indent=2))
    print('Corrected visible faces:', len(changed), '/', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'correct'])
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detail', action='store_true')
    args = parser.parse_args()
    if args.action == 'correct' and args.video is None:
        parser.error('--video is required for correct')
    (prepare if args.action == 'prepare' else correct)(args)
