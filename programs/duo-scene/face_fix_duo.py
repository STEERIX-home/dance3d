"""Bring ONE character's face closer to its reference photo, in a video with several characters.

Only faces that lie inside the chosen dancer's own identity mask are touched; the other
character and every frame without a clear face of that dancer stay exactly as generated.
Runs on the CPU with the face models of the earlier single-person program (InsightFace).
"""
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
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.model_zoo import get_model, model_zoo

IDENTITY = {'A': lambda m: (m[..., 0] > 200) & (m[..., 2] < 60), 'B': lambda m: (m[..., 2] > 200) & (m[..., 0] < 60)}


def eligible(face):
    width, height = face.bbox[2:] - face.bbox[:2]
    eyes = np.linalg.norm(face.kps[0] - face.kps[1])
    return bool(face.det_score >= .6 and min(width, height) >= 22 and eyes >= max(6, width * .15)
                and np.isfinite(face.kps).all())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--masks', required=True, type=Path, help='identity mask frames of the same camera path')
    parser.add_argument('--first', type=int, default=1)
    parser.add_argument('--label', required=True, choices=list(IDENTITY))
    parser.add_argument('--image', required=True, type=Path, help='reference photo of that character')
    parser.add_argument('--models', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()

    cv2.setNumThreads(4)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 6, 1
    # idle worker threads must sleep, not spin: spinning kept ~22 cores busy next to the GPU job
    opts.add_session_config_entry('session.intra_op.allow_spinning', '0')
    # InsightFace drops sess_options on the way to onnxruntime, so every model would start one
    # spinning thread per core: hand the options to its session class directly
    original_init = model_zoo.PickableInferenceSession.__init__

    def limited_init(self, model_path, **kwargs):
        kwargs.setdefault('sess_options', opts)
        original_init(self, model_path, **kwargs)

    model_zoo.PickableInferenceSession.__init__ = limited_init
    app = FaceAnalysis(name='buffalo_l', root=str(args.models), allowed_modules=['detection', 'recognition'],
                       providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    reference = cv2.imread(str(args.image))
    sources = [f for f in app.get(reference) if f.det_score >= .65]
    if len(sources) != 1:
        raise ValueError(f'The reference photo must show exactly one face, found {len(sources)}')
    source = sources[0]
    swapper = get_model(str(args.models / 'inswapper_128.onnx'), providers=['CPUExecutionProvider'])

    capture = cv2.VideoCapture(str(args.video))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    silent = args.output.with_suffix('.silent.mp4')
    encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
                                '-r', f'{fps:g}', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '15', '-preset', 'slow',
                                '-pix_fmt', 'yuv420p', str(silent)], stdin=subprocess.PIPE)
    records, index = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        mask = cv2.imread(str(args.masks / f'frame_{args.first + index:04d}.png'))
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        own = cv2.dilate(IDENTITY[args.label](mask).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        # a face lying on the OTHER dancer's body is never this character's face, however close
        # the two heads are
        for label, select in IDENTITY.items():
            if label != args.label:
                own &= ~select(mask)
        record = dict(frame=index + 1, corrected=False)
        result = frame
        for face in app.get(frame):
            x, y = np.clip(face.kps.mean(0).astype(int), 0, [width - 1, height - 1])
            if not own[y, x] or not eligible(face):
                continue
            before = float(np.dot(source.normed_embedding, face.normed_embedding))
            # work on a window around the face only: pasting back over the whole 1080p frame is slow
            side = int(max(384, 3 * max(face.bbox[2:] - face.bbox[:2])))
            x0 = int(np.clip(x - side // 2, 0, max(0, width - side)))
            y0 = int(np.clip(y - side // 2, 0, max(0, height - side)))
            window = result[y0:y0 + side, x0:x0 + side]
            local = Face(bbox=face.bbox - np.array([x0, y0, x0, y0], dtype=np.float32),
                         kps=face.kps - np.array([x0, y0], dtype=np.float32), det_score=face.det_score)
            result = result.copy()
            result[y0:y0 + side, x0:x0 + side] = swapper.get(window, local, source, paste_back=True)
            app.models['recognition'].get(result, face)
            record.update(corrected=True, face_px=float(min(face.bbox[2:] - face.bbox[:2])), similarity_before=before,
                          similarity_after=float(np.dot(source.normed_embedding, face.normed_embedding)))
            break  # one face per frame for this character
        encoder.stdin.write(result.tobytes())
        records.append(record)
        index += 1
    capture.release()
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('ffmpeg failed')
    # keep the original sound
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(silent), '-i', str(args.video), '-map', '0:v:0', '-map', '1:a:0?',
                    '-c', 'copy', '-movflags', '+faststart', str(args.output)], check=True)
    silent.unlink()
    changed = [r for r in records if r['corrected']]
    summary = dict(video=str(args.video), character=args.label, frames=len(records), corrected_frames=len(changed),
                   mean_similarity_before=float(np.mean([r['similarity_before'] for r in changed])) if changed else None,
                   mean_similarity_after=float(np.mean([r['similarity_after'] for r in changed])) if changed else None,
                   note='Similarity is the face model\'s own score, not a guarantee of likeness. Frames without a '
                        'clear face of this character (back view, profile, small or hidden face) are unchanged.',
                   records=records)
    args.report.write_text(json.dumps(summary))
    print(f'face corrected in {len(changed)}/{len(records)} frames; similarity '
          f'{summary["mean_similarity_before"]} -> {summary["mean_similarity_after"]}')


if __name__ == '__main__':
    main()
