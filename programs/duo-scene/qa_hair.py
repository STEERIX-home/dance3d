"""How much hair (or anything else the model adds to the head) shows outside the bald head of the 3D guide.

The video model must invent the back and sides of a character's head wherever the picture does not show them. On
11-22 SCAIL-2 gave reze.webp (short hair, small low bun) a long ponytail in both side shots, a big white bow and
twin braids from behind, while the front shots kept her short hair: a viewer sees a new hairstyle after each cut.

For one generated frame: the mannequin's head is the outline of the body mesh's points near the head joint, seen
through the frame's camera (camera-path.json); the dancer's whole body is the guide's mask. Hair outside the body
shows where the generated frame differs in structure from the guide (local SSIM of the brightness, so that a
colour shift of the room does not count). The measure is that area within reach of the head, in head areas:

    extra = area(changed & near the head & outside the body) / area(head)

Short hair adds a thin rim around the skull, a ponytail or braids that stick out add much more. The shots of a run
that face the character (front, front_close) show what the picture shows; a side or back shot whose median is far
above theirs has given the character hair the picture does not have (limit_from()). Measured on dance 02 (median
per shot): reze with her short hair and bun (H3 run 48c1) left/right/back 0.16-0.27, front 0.33-0.48, front close
0.19-0.22; SCAIL-2 with the ponytails 0.62 and 1.05 in the two left shots; with the limit of RATIO and MARGIN both
ponytail shots fail and no shot of the H3 run does. Not seen by this measure: hair that lies over the body (braids
hanging down the back in a back shot), a hair ribbon inside the head's outline.

The same 3D head serves two more checks of pair_run.py, neither of which looks at the colour of the clothes:
- facing(): where the face points (SMPL-X nose and ears) against the way to the camera. The camera sees the back of
  the head only where the face is turned more than 120 degrees away (AWAY); the body's direction is not enough, as a
  body seen from behind or from the side often turns its face to the camera and then shows a profile.
- head_likeness(): the dancer's head beside the guide's bald mannequin head. A head the model left as the mannequin
  differs from the guide by a few levels and keeps its structure (LEFT_DIFFERENCE, LEFT_SSIM).

    qa_hair.py --run RUN_FOLDER [--output hair.json]     (a challenge run: one line per shot)
"""
import argparse
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np

HEAD_RADIUS_M = 0.14      # mesh points this near the head joint make the head (blender_views.py frames close shots the same way)
REACH = 1.2               # how far from the head (in head heights) added hair is looked for
CHANGED = 0.35            # local SSIM below this: the frame shows something the guide does not
EDGE = 0.006              # share of the frame height kept free along the body's outline (the outline itself differs)
FACING = ('front', 'front_close')   # views whose frames show what the picture shows
# a side or back shot fails when its median is above both the facing shots' median times RATIO and that median plus MARGIN
RATIO, MARGIN = 1.4, 0.15
# SMPL-X (the body mesh of dance-joints): number of points, the tip of the nose, the right and the left ear
SMPLX_POINTS, NOSE, EARS = 10475, 9120, (616, 6)
AWAY = -0.5  # cosine of the face's direction and the way to the camera below which the camera sees the back of the head
# a head still drawn as the guide's mannequin: mean difference below LEFT_DIFFERENCE (0-255) and local SSIM above LEFT_SSIM
LEFT_DIFFERENCE, LEFT_SSIM = 25.0, 0.75


def read_pc2(path):
    with open(path, 'rb') as file:
        _, _, points, _, _, samples = struct.unpack('<12siiffi', file.read(32))
        return np.frombuffer(file.read(), dtype='<f4').reshape(samples, points, 3)


def cover(picture, width, height, interpolation):
    """Scale to cover width x height and cut the middle (as pair_run.py does with the guide)."""
    h, w = picture.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(picture, (int(round(w * scale)), int(round(h * scale))), interpolation=interpolation)
    y, x = (resized.shape[0] - height) // 2, (resized.shape[1] - width) // 2
    return resized[y:y + height, x:x + width]


def rotation_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Heads:
    """The mannequin's head outline in the frames of one camera (scene folder of blender_views.py)."""

    def __init__(self, scene, inputs):
        scene, inputs = Path(scene), Path(inputs)
        data = json.loads((inputs / 'scene_inputs.json').read_text())
        person = data['people'][0]
        self.cache = read_pc2(inputs / person['cache'])
        self.head = np.asarray(person['head'], dtype=np.float64)
        self.path = {row['frame']: row for row in json.loads((scene / 'camera-path.json').read_text())}
        manifest = json.loads((scene / 'manifest.json').read_text())
        self.render_w, self.render_h = manifest['resolution']
        self.tan_x = manifest['sensor_width_mm'] / 2 / manifest['lens_mm']
        self.tan_y = self.tan_x * self.render_h / self.render_w
        self.smplx = self.cache.shape[1] == SMPLX_POINTS  # facing() needs the SMPL-X points of the nose and the ears

    def facing(self, frame):
        """Cosine between the direction the face points (ears -> nose) and the way from the head to this frame's
        camera: 1 the face looks at the camera, -1 the camera sees the back of the head. None without SMPL-X points
        or a camera for the frame."""
        row = self.path.get(frame)
        if not self.smplx or row is None or not 1 <= frame <= len(self.cache):
            return None
        points = self.cache[frame - 1].astype(np.float64)
        ears = (points[EARS[0]] + points[EARS[1]]) / 2
        face, way = points[NOSE] - ears, np.asarray(row['location'], dtype=np.float64) - ears
        return float(face @ way / (np.linalg.norm(face) * np.linalg.norm(way) + 1e-9))

    def mask(self, frame, width, height):
        """Filled outline of the head in a width x height frame (covered like the guide), or None."""
        points = self.cache[frame - 1].astype(np.float64)
        points = points[np.linalg.norm(points - self.head[frame - 1], axis=1) <= HEAD_RADIUS_M]
        row = self.path.get(frame)
        if row is None or len(points) < 20:
            return None
        local = (points - np.asarray(row['location'])) @ rotation_matrix(row['rotation_quaternion'])
        depth = -local[:, 2]
        if (depth <= 0.05).any():
            return None
        u = (local[:, 0] / depth / self.tan_x + 1) / 2 * self.render_w
        v = (1 - local[:, 1] / depth / self.tan_y) / 2 * self.render_h
        scale = max(width / self.render_w, height / self.render_h)
        u = u * scale - (self.render_w * scale - width) / 2
        v = v * scale - (self.render_h * scale - height) / 2
        hull = cv2.convexHull(np.stack([u, v], 1).astype(np.float32)).astype(np.int32)
        out = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(out, hull, 1)
        return out > 0


def ssim_map(a, b, size=9):
    a, b = a.astype(np.float32), b.astype(np.float32)

    def blur(x):
        return cv2.GaussianBlur(x, (size, size), size / 4)
    ma, mb = blur(a), blur(b)
    va, vb, cab = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))


def head_likeness(generated, guide, head):
    """(mean difference 0-255, mean local SSIM of the brightness) of the generated frame and the guide inside the
    head's outline; None when the head is too small to tell."""
    if head is None:
        return None
    inner = cv2.erode(head.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if inner.sum() < 150:
        return None
    rows, columns = np.nonzero(inner.any(1))[0], np.nonzero(inner.any(0))[0]
    y0, y1 = max(0, rows[0] - 12), rows[-1] + 13
    x0, x1 = max(0, columns[0] - 12), columns[-1] + 13
    a, b, inside = generated[y0:y1, x0:x1], guide[y0:y1, x0:x1], inner[y0:y1, x0:x1]
    difference = float(np.abs(a.astype(int) - b.astype(int)).mean(-1)[inside].mean())
    similar = float(ssim_map(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))[inside].mean())
    return difference, similar


def head_left(likeness):
    """Is the head still the guide's mannequin head (head_likeness() values)?"""
    return likeness is not None and likeness[0] < LEFT_DIFFERENCE and likeness[1] > LEFT_SSIM


def extra_hair(generated, guide, body, head):
    """Added area outside the body within reach of the head, in head areas (None when the head is too small or cut)."""
    height = generated.shape[0]
    area = int(head.sum())
    if area < 150:
        return None
    rows = np.nonzero(head.any(1))[0]
    size = rows[-1] - rows[0] + 1
    if rows[0] <= 0:  # the head leaves the picture at the top: part of the hair cannot be seen
        return None
    near = cv2.dilate(head.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(REACH * size) + 1,) * 2)) > 0
    edge = max(1, int(round(EDGE * height)))
    outside = cv2.dilate(body.astype(np.uint8), np.ones((2 * edge + 1, 2 * edge + 1), np.uint8)) == 0
    changed = ssim_map(cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY), cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)) < CHANGED
    changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return float(attached(changed & near & outside, head, edge).sum()) / area


def attached(found, head, edge):
    """The parts of `found` that touch the head's rim (hair grows from the head; a redrawn wall stripe does not)."""
    rim = cv2.dilate(head.astype(np.uint8), np.ones((2 * edge + 7, 2 * edge + 7), np.uint8)) > 0
    count, labels = cv2.connectedComponents(found.astype(np.uint8), connectivity=8)
    if count <= 1:
        return found
    touching = np.unique(labels[rim & found])
    return np.isin(labels, touching[touching > 0])


def chunk_values(heads, frames, guides, masks, numbers, every=2):
    """[(guide frame number, extra hair)] for every n-th of these frames; frames, guides and identity masks (BGR,
    dancer A blue) all at the generation size (pair_run.py has them in memory)."""
    values = []
    for index in range(0, len(numbers), every):
        generated, guide, mask = frames[index], guides[index], masks[index]
        height, width = generated.shape[:2]
        head = heads.mask(numbers[index], width, height)
        if head is None:
            continue
        body = (mask[..., 0] > 200) & (mask[..., 2] < 60)
        value = extra_hair(generated, guide, body, head)
        if value is not None:
            values.append((numbers[index], round(value, 3)))
    return values


def frame_values(scene, inputs, generated_files, numbers, width, height, every=2):
    """chunk_values() for frames on disk (a finished run)."""
    scene = Path(scene)
    heads = Heads(scene, inputs)
    guide_folder = scene / ('guide' if (scene / 'guide').exists() else 'pair')
    frames, guides, masks, kept = [], [], [], []
    for file, number in list(zip(generated_files, numbers))[::every]:
        generated = cv2.imread(str(file))
        found = sorted(guide_folder.glob(f'frame_{number:04d}.*'))
        mask_picture = cv2.imread(str(scene / 'mask' / f'frame_{number:04d}.png'))
        if generated is None or not found or mask_picture is None:
            continue
        if generated.shape[:2] != (height, width):
            generated = cv2.resize(generated, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(generated)
        guides.append(cover(cv2.imread(str(found[0])), width, height, cv2.INTER_AREA))
        masks.append(cover(mask_picture, width, height, cv2.INTER_NEAREST))
        kept.append(number)
    return chunk_values(heads, frames, guides, masks, kept, every=1)


def limit_from(facing):
    """The most added hair (median of a shot) a side or back shot may have, from the frame values (numbers) of the
    run's facing shots; None without them."""
    if not facing:
        return None
    base = float(np.median(facing))
    return max(RATIO * base, base + MARGIN)


def check_shot(values, facing):
    """values: this shot's frame values; facing: the frame values of the run's facing shots made so far.
    Returns (failed, median, limit); no facing frames yet -> no limit."""
    if not values:
        return False, None, None
    median = float(np.median([v for _, v in values]))
    limit = limit_from([v for _, v in facing])
    return limit is not None and median > limit, median, limit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='a challenge run folder')
    parser.add_argument('--every', type=int, default=2)
    parser.add_argument('--output')
    args = parser.parse_args()
    work = Path(args.run) / 'outputs/work'
    plan = json.loads((work / 'plan.json').read_text())
    result, facing = {}, []
    for shot in plan['shots']:  # as challenge.py: a side or back shot is judged by the front shots made before it
        name = shot['name']
        run = json.loads((work / 'gen' / name / 'run.json').read_text())
        first = run['first_frame']
        numbers = [n for n in range(shot['first'], shot['last'] + 1)]
        files = [work / 'gen' / name / 'generated' / f'frame_{n - first + 1:04d}.png' for n in numbers]
        width, height = run['width'], run['height']
        values = frame_values(run['scene'], work / 'scene_inputs', files, numbers, width, height, args.every)
        failed, median, limit = check_shot(values, [] if shot['view'] in FACING else facing)
        if shot['view'] in FACING:
            facing += values
        result[name] = dict(view=shot['view'], median=None if median is None else round(median, 3),
                            p90=round(float(np.percentile([v for _, v in values], 90)), 3) if values else None,
                            limit=None if limit is None else round(limit, 3), failed=failed, values=values)
        print(f'{name:24s} {shot["view"]:12s} median {result[name]["median"]}  p90 {result[name]["p90"]}  '
              f'limit {result[name]["limit"]}  {"FAIL" if failed else ""}', flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
