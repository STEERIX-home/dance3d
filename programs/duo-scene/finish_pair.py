"""Finish a pair run (pair_run.py): full-size frames, the photo character's face held to its
photo, the original music.

The whole picture of every frame comes from the one generation run (characters, their shadows and
the room together); nothing is cut out or laid over another picture here. The frames are enlarged
from the generation size to the size of the 3D render (evenly, cutting the few pixels that do not
fit) and sharpened a little.

For a character whose picture shows one clear face (--face LABEL=picture; a picture in which the
face detector finds no face, like most drawings, is left alone) the face is compared with the
picture and corrected only as far as needed (the rule of face_fix_frames.py: left alone
where it matches well, fully corrected where it matches poorly, mixed in between, decided on the
match smoothed over half a second). Which face is that dancer's is decided from the 3D scene: the
face must lie at the head of that dancer's own 3D body as the camera sees it, and nearer to it
than to any other dancer's head. Runs on the CPU (InsightFace models of the single-person program).

Outputs: --output MP4 (H.264 + AAC); --compare [3D guide | result] MP4; faces.json in the run folder
"""
import argparse
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import cv2
import numpy as np

from composite import encoder

HEAD_METRES = 0.23  # height of an adult head


def quaternion_matrix(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def heads_in_picture(scene, inputs, size, zoom=1.0):
    """Per dancer and guide frame: where the head of the 3D body is in the picture (x, y) and how
    tall a head is there (pixels). {label: (frames, 3)}. zoom: how much larger than the 3D render
    the finished picture shows everything (about its centre)."""
    manifest = json.loads((scene / 'manifest.json').read_text())
    path = json.loads((scene / 'camera-path.json').read_text())
    focal = manifest['lens_mm'] / manifest['sensor_width_mm'] * size[0] * zoom
    out = {}
    for person in json.loads((inputs / 'scene_inputs.json').read_text())['people']:
        rows = []
        for row, head in zip(path, np.asarray(person['head'])):
            local = quaternion_matrix(*row['rotation_quaternion']).T @ (head - np.asarray(row['location']))
            depth = max(0.2, -local[2])  # a Blender camera looks along -z, x right, y up
            rows.append((size[0] / 2 + focal * local[0] / depth, size[1] / 2 - focal * local[1] / depth,
                         focal * HEAD_METRES * person['shown_size'] / depth))
        out[person['label']] = np.asarray(rows)
    return out


def eligible(face):
    width, height = face.bbox[2:] - face.bbox[:2]
    eyes = np.linalg.norm(face.kps[0] - face.kps[1])
    return bool(face.det_score >= .6 and min(width, height) >= 22 and eyes >= max(6, width * .15)
                and np.isfinite(face.kps).all())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path, help='output folder of pair_run.py')
    parser.add_argument('--inputs', required=True, type=Path, help='folder of export_scene_inputs.py (head positions)')
    parser.add_argument('--face', action='append', default=[], help='LABEL=reference photo of a real person')
    parser.add_argument('--models', type=Path, help='InsightFace models (needed with --face)')
    parser.add_argument('--keep-above', type=float, default=0.80)
    parser.add_argument('--full-below', type=float, default=0.65)
    parser.add_argument('--sharpen', type=float, default=0.4, help='unsharp mask amount after enlarging (0 = none)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--audio')
    parser.add_argument('--compare')
    parser.add_argument('--frames-folder', help='also write the finished frames here (PNG)')
    args = parser.parse_args()
    record = json.loads((args.run / 'run.json').read_text())
    scene = Path(record['scene'])
    manifest = json.loads((scene / 'manifest.json').read_text())
    size = tuple(manifest['resolution'])
    fps, total, first = record['fps'], record['frames'], record['first_frame']
    if args.audio and first != 1:
        raise NotImplementedError('audio for runs that do not start at frame 1')
    if cv2.imread(str(args.run / 'generated' / 'frame_0001.png')).shape[1] != record['width']:
        raise ValueError('The generated frames do not have the size written in run.json')
    cv2.setNumThreads(4)

    # The generation size has not exactly the shape of the render (768x1344 against 1080x1920): the guide
    # was scaled to cover it and cut at two edges, and the result is enlarged the same way, never stretched.
    made = (record['width'], record['height'])
    grow = max(size[0] / made[0], size[1] / made[1])
    zoom = grow * max(made[0] / size[0], made[1] / size[1])

    def enlarged(index):
        frame = cv2.imread(str(args.run / 'generated' / f'frame_{index + 1:04d}.png'))
        wide, tall = max(size[0], round(made[0] * grow)), max(size[1], round(made[1] * grow))
        frame = cv2.resize(frame, (wide, tall), interpolation=cv2.INTER_LANCZOS4)
        left, top = (wide - size[0]) // 2, (tall - size[1]) // 2
        frame = np.ascontiguousarray(frame[top:top + size[1], left:left + size[0]])
        if args.sharpen:
            frame = cv2.addWeighted(frame, 1 + args.sharpen, cv2.GaussianBlur(frame, (0, 0), 1.0), -args.sharpen, 0)
        return frame

    photos = dict(entry.split('=', 1) for entry in args.face)
    patches, records = {}, {label: [dict(frame=index + 1, corrected=False) for index in range(total)] for label in photos}
    if photos:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        from insightface.app.common import Face
        from insightface.model_zoo import get_model, model_zoo

        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = 6, 1
        # idle worker threads must sleep, not spin; InsightFace drops sess_options on the way to
        # onnxruntime, so hand them to its session class directly (see face_fix_frames.py)
        opts.add_session_config_entry('session.intra_op.allow_spinning', '0')
        original_init = model_zoo.PickableInferenceSession.__init__

        def limited_init(self, model_path, **kwargs):
            kwargs.setdefault('sess_options', opts)
            original_init(self, model_path, **kwargs)

        model_zoo.PickableInferenceSession.__init__ = limited_init
        app = FaceAnalysis(name='buffalo_l', root=str(args.models), allowed_modules=['detection', 'recognition'],
                           providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
        swapper = get_model(str(args.models / 'inswapper_128.onnx'), providers=['CPUExecutionProvider'])
        sources, skipped = {}, {}
        for label, photo in photos.items():
            found = [f for f in app.get(cv2.imread(photo)) if f.det_score >= .65]
            if len(found) == 1:
                sources[label] = found[0]
            else:  # a drawn face the detector does not read, or several people: this character's face stays as generated
                skipped[label] = len(found)
        heads = heads_in_picture(scene, args.inputs, size, zoom)
        for index in range(total):
            frame = enlarged(index)
            for label, source in sources.items():
                x, y, tall = heads[label][first - 1 + index]
                if not (0 <= x < size[0] and 0 <= y < size[1]):
                    continue
                side = int(np.clip(6 * tall, 320, 640))
                x0 = int(np.clip(x - side / 2, 0, size[0] - side))
                y0 = int(np.clip(y - side / 2, 0, size[1] - side))
                window = frame[y0:y0 + side, x0:x0 + side]
                best = None
                for face in app.get(window):
                    centre = face.kps.mean(0) + (x0, y0)
                    own = np.linalg.norm(centre - (x, y))
                    others = [np.linalg.norm(centre - heads[other][first - 1 + index][:2]) for other in heads if other != label]
                    if eligible(face) and own < 1.2 * tall and all(own < other for other in others) and (best is None or own < best[0]):
                        best = (own, face)
                if best is None:
                    continue
                face = best[1]
                before = float(np.dot(source.normed_embedding, face.normed_embedding))
                swapped = swapper.get(window, Face(bbox=face.bbox, kps=face.kps, det_score=face.det_score), source, paste_back=True)
                app.models['recognition'].get(swapped, face)
                records[label][index].update(face_px=float(min(face.bbox[2:] - face.bbox[:2])), similarity_before=before,
                                             similarity_after=float(np.dot(source.normed_embedding, face.normed_embedding)),
                                             off_the_3d_head_px=round(float(best[0]), 1))
                patches[label, index] = (x0, y0, swapped)
    # how much of the correction each frame gets: from the match before, smoothed over half a second
    reach = max(1, int(round(0.25 * fps)))
    for label in photos:
        match = np.array([r.get('similarity_before', np.nan) for r in records[label]])
        for index, entry in enumerate(records[label]):
            if (label, index) in patches:
                smooth = float(np.nanmedian(match[max(0, index - reach):index + reach + 1]))
                entry['share'] = round(float(np.clip((args.keep_above - smooth) / (args.keep_above - args.full_below), 0, 1)), 3)
                entry['corrected'] = entry['share'] > 0

    duration = total / fps
    out = encoder(args.output, size[0], size[1], fps, args.audio, duration)
    compare = encoder(args.compare, size[0] * 2, size[1], fps, args.audio, duration) if args.compare else None
    if args.frames_folder:
        Path(args.frames_folder).mkdir(parents=True, exist_ok=True)
    for index in range(total):
        frame = enlarged(index)
        for label in photos:
            entry = records[label][index]
            if entry['corrected']:
                x0, y0, swapped = patches.pop((label, index))
                part = frame[y0:y0 + swapped.shape[0], x0:x0 + swapped.shape[1]]
                frame[y0:y0 + swapped.shape[0], x0:x0 + swapped.shape[1]] = cv2.addWeighted(swapped, entry['share'], part,
                                                                                            1 - entry['share'], 0)
        out.stdin.write(frame.tobytes())
        if args.frames_folder:
            cv2.imwrite(str(Path(args.frames_folder) / f'frame_{index + 1:04d}.png'), frame)
        if compare:
            guide = cv2.imread(str(scene / record['guide'] / f'frame_{first + index:04d}.jpg'))
            compare.stdin.write(np.concatenate([guide, frame], axis=1).tobytes())
    for process in filter(None, (out, compare)):
        process.stdin.close()
        if process.wait():
            raise RuntimeError('ffmpeg failed')
    summary = dict(run=str(args.run), frames=total, size=list(size), enlarged_from=[record['width'], record['height']],
                   sharpen=args.sharpen, keep_above=args.keep_above, full_below=args.full_below, faces={},
                   note='Similarity is the face model\'s own score, not a guarantee of likeness. Frames without a clear '
                        'face (back view, profile, small or hidden face) are unchanged.')
    for label in photos:
        if photos and label in skipped:
            summary['faces'][label] = dict(photo=photos[label], corrected_frames=0,
                                           not_corrected=f'{skipped[label]} clear faces found in the picture (exactly one is needed)')
            print(f'{label}: face not corrected, {skipped[label]} clear faces found in the picture')
            continue
        seen = [r for r in records[label] if 'similarity_before' in r]
        changed = [r for r in seen if r['corrected']]
        summary['faces'][label] = dict(
            photo=photos[label], frames_with_a_clear_face=len(seen), corrected_frames=len(changed),
            fully_corrected_frames=sum(1 for r in changed if r['share'] >= 1),
            match_as_generated=dict(median=float(np.median([r['similarity_before'] for r in seen])),
                                    p10=float(np.percentile([r['similarity_before'] for r in seen], 10))) if seen else None,
            face_px_median=float(np.median([r['face_px'] for r in seen])) if seen else None,
            mean_similarity_before=float(np.mean([r['similarity_before'] for r in changed])) if changed else None,
            mean_similarity_after=float(np.mean([r['similarity_after'] for r in changed])) if changed else None,
            records=records[label])
        print(f'{label}: clear face in {len(seen)}/{total} frames, match as generated {summary["faces"][label]["match_as_generated"]}, '
              f'corrected in {len(changed)} frames ({summary["faces"][label]["fully_corrected_frames"]} fully), '
              f'after correction {summary["faces"][label]["mean_similarity_after"]}')
    (args.run / 'faces.json').write_text(json.dumps(summary))
    print(f'{total} frames, {duration:.2f} s -> {args.output}')


if __name__ == '__main__':
    main()
