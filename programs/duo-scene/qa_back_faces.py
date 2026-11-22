"""Faces where the camera sees the back of the head: for each shot of a challenge run, the frames in which the
face detector (InsightFace, the finishing step's detector, on the CPU) finds a face, and how sure it is.
In a shot filmed from behind (plan back_share >= 0.5) a found face means the model painted a face on the back of
the head; the front shots give the detector's usual rate for this character (a drawn face may not be found at all,
then the check cannot tell anything).

    qa_back_faces.py --run RUN_FOLDER --models FACE_MODELS [--output faces.json]     (python.face)
    qa_back_faces.py --files F1 F2 ... --models FACE_MODELS    (pair_run.py: one JSON line, the best face score of each
                                                               file or null)
"""
import argparse
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import cv2
import numpy as np


def detector(models):
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import model_zoo
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 4, 1
    opts.add_session_config_entry('session.intra_op.allow_spinning', '0')
    original_init = model_zoo.PickableInferenceSession.__init__

    def limited_init(self, model_path, **kwargs):
        kwargs.setdefault('sess_options', opts)
        original_init(self, model_path, **kwargs)

    model_zoo.PickableInferenceSession.__init__ = limited_init
    app = FaceAnalysis(name='buffalo_l', root=str(models), allowed_modules=['detection'], providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run')
    parser.add_argument('--files', nargs='+', help='pictures to look at (instead of --run)')
    parser.add_argument('--models', required=True, help='InsightFace models folder (models.face)')
    parser.add_argument('--every', type=int, default=2, help='look at every n-th frame')
    parser.add_argument('--output')
    args = parser.parse_args()
    app = detector(args.models)
    if args.files:
        scores = []
        for file in args.files:
            faces = app.get(cv2.imread(file))
            scores.append(round(float(max(f.det_score for f in faces)), 3) if faces else None)
        print(json.dumps(scores), flush=True)
        return
    work = Path(args.run) / 'outputs/work'
    plan = json.loads((work / 'plan.json').read_text())
    result = {}
    for shot in plan['shots']:
        run = json.loads((work / 'gen' / shot['name'] / 'run.json').read_text())
        first = run['first_frame']
        found = []
        numbers = list(range(shot['first'], shot['last'] + 1, args.every))
        for number in numbers:
            frame = cv2.imread(str(work / 'gen' / shot['name'] / 'generated' / f'frame_{number - first + 1:04d}.png'))
            if frame is None:
                continue
            faces = app.get(frame)
            if faces:
                best = max(faces, key=lambda f: f.det_score)
                found.append([number, round(float(best.det_score), 3)])
        result[shot['name']] = dict(view=shot['view'], back_share=shot['back_share'], frames_looked_at=len(numbers),
                                    frames_with_a_face=len(found), share=round(len(found) / max(1, len(numbers)), 3), faces=found)
        print(f'{shot["name"]:24s} back {shot["back_share"]:.0%}  faces in {len(found)}/{len(numbers)} frames'
              + (f' (best {max(s for _, s in found):.2f})' if found else ''), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
