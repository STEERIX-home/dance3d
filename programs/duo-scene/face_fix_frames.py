"""Bring the generated character's face closer to its reference photo, in the window pictures of
a solo run (solo_run.py), before the dancers are laid over the set.

The window shows one dancer only, so the face found on that dancer's matte is the character's
face. Frames without a clear face (back view, profile, small or hidden face) stay exactly as
generated. The correction makes a face a little softer, so it is used only as far as needed:
where the generated face already matches the photo well (--keep-above) it is left as it is,
where it matches poorly (--full-below) it is fully corrected, in between both are mixed. The
decision follows the match smoothed over half a second, so the face does not flicker between the
two. Runs on the CPU with the face models of the earlier single-person program (InsightFace).

Outputs in the run folder: faces/frame_####.png (all frames, corrected or not), faces.json
"""
import argparse
import json
import os
import shutil
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.model_zoo import get_model, model_zoo


def eligible(face):
    width, height = face.bbox[2:] - face.bbox[:2]
    eyes = np.linalg.norm(face.kps[0] - face.kps[1])
    return bool(face.det_score >= .6 and min(width, height) >= 22 and eyes >= max(6, width * .15)
                and np.isfinite(face.kps).all())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--image', required=True, type=Path, help='reference photo of the character')
    parser.add_argument('--models', required=True, type=Path)
    parser.add_argument('--keep-above', type=float, default=0.80)
    parser.add_argument('--full-below', type=float, default=0.65)
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

    record = json.loads((args.run / 'run.json').read_text())
    total = record['frames']
    out = args.run / 'faces'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    records, corrected = [], {}
    for index in range(total):
        name = f'frame_{index + 1:04d}.png'
        frame = cv2.imread(str(args.run / 'generated' / name))
        matte = cv2.imread(str(args.run / 'alpha' / name), cv2.IMREAD_GRAYSCALE)
        height, width = frame.shape[:2]
        own = cv2.dilate((matte > 127).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        entry = dict(frame=index + 1, corrected=False)
        result = frame
        faces = sorted(app.get(frame), key=lambda f: -float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        for face in faces:
            x, y = np.clip(face.kps.mean(0).astype(int), 0, [width - 1, height - 1])
            if not own[y, x] or not eligible(face):
                continue
            before = float(np.dot(source.normed_embedding, face.normed_embedding))
            side = int(max(384, 3 * max(face.bbox[2:] - face.bbox[:2])))
            x0 = int(np.clip(x - side // 2, 0, max(0, width - side)))
            y0 = int(np.clip(y - side // 2, 0, max(0, height - side)))
            window = result[y0:y0 + side, x0:x0 + side]
            local = Face(bbox=face.bbox - np.array([x0, y0, x0, y0], dtype=np.float32),
                         kps=face.kps - np.array([x0, y0], dtype=np.float32), det_score=face.det_score)
            result = result.copy()
            result[y0:y0 + side, x0:x0 + side] = swapper.get(window, local, source, paste_back=True)
            app.models['recognition'].get(result, face)
            entry.update(corrected=True, face_px=float(min(face.bbox[2:] - face.bbox[:2])), similarity_before=before,
                         similarity_after=float(np.dot(source.normed_embedding, face.normed_embedding)))
            corrected[index] = result
            break  # one face: the character's
        records.append(entry)
    # how much of the correction each frame gets: from the match before, smoothed over half a second
    fps = record['fps']
    match = np.array([r.get('similarity_before', np.nan) for r in records])
    reach = max(1, int(round(0.25 * fps)))
    for index, entry in enumerate(records):
        name = f'frame_{index + 1:04d}.png'
        frame = cv2.imread(str(args.run / 'generated' / name))
        if index in corrected:
            near = match[max(0, index - reach):index + reach + 1]
            smooth = float(np.nanmedian(near))
            share = float(np.clip((args.keep_above - smooth) / (args.keep_above - args.full_below), 0, 1))
            entry['share'] = round(share, 3)
            if share > 0:
                frame = cv2.addWeighted(corrected[index], share, frame, 1 - share, 0)
            entry['corrected'] = share > 0
        cv2.imwrite(str(out / name), frame)
    changed = [r for r in records if r['corrected']]
    faces = [r for r in records if 'similarity_before' in r]
    summary = dict(run=str(args.run), frames=len(records), frames_with_a_clear_face=len(faces), corrected_frames=len(changed),
                   fully_corrected_frames=sum(1 for r in changed if r['share'] >= 1),
                   match_as_generated=dict(median=float(np.median([r['similarity_before'] for r in faces])) if faces else None,
                                           p10=float(np.percentile([r['similarity_before'] for r in faces], 10)) if faces else None),
                   keep_above=args.keep_above, full_below=args.full_below,
                   face_px_median=float(np.median([r['face_px'] for r in changed])) if changed else None,
                   mean_similarity_before=float(np.mean([r['similarity_before'] for r in changed])) if changed else None,
                   mean_similarity_after=float(np.mean([r['similarity_after'] for r in changed])) if changed else None,
                   note='Similarity is the face model\'s own score, not a guarantee of likeness. Frames without a '
                        'clear face (back view, profile, small or hidden face) are unchanged.',
                   records=records)
    (args.run / 'faces.json').write_text(json.dumps(summary))
    print(f'clear face in {len(faces)}/{len(records)} frames, match as generated {summary["match_as_generated"]}; '
          f'corrected in {len(changed)} frames ({summary["fully_corrected_frames"]} fully)')


if __name__ == '__main__':
    main()
