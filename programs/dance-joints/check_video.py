"""Videos for checking the joint motion by eye.

  joints_front_side.mp4    the joints from the front and from the side above the floor (y = 0), at
                           one scale for the whole video, with the pelvis height; nothing of the
                           source picture is in it
  joints_over_source.mp4   the source video with the joints drawn where the 3D world puts them
                           (seen through the camera path), next to the same front and side views.
                           It shows the source video: for checking only, never to be shared

"Front" looks along the camera's mean viewing direction, so left and right are as in the source
picture; "side" looks from the camera's left (the dancer's right side when they face the camera),
farther from the camera to the right. The dancer's left side is drawn blue, the right side red;
what was not seen (the whole body bridged between seen frames, or legs out of the picture) grey.

    check_video.py --video source.mp4 --joints JOINTS_DIR --output DIR
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# SMPL-X body joints 0..21 and their parents
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
LEFT, RIGHT = {1, 4, 7, 10, 13, 16, 18, 20}, {2, 5, 8, 11, 14, 17, 19, 21}
LEG_JOINTS = {1, 2, 4, 5, 7, 8, 10, 11}  # a bone is drawn in the colour of its child joint
COLOURS = {'left': (255, 160, 40), 'right': (60, 60, 255), 'middle': (60, 230, 255), 'estimated': (150, 150, 150)}
PANEL = 960  # height of every panel (pixels)


def colour(joint, estimated, legs=False):
    if estimated or (legs and joint in LEG_JOINTS):
        return COLOURS['estimated']
    return COLOURS['left'] if joint in LEFT else COLOURS['right'] if joint in RIGHT else COLOURS['middle']


def draw_skeleton(picture, points, estimated, width=4, legs=False):
    """points (22, 2) pixel positions; picture is drawn on in place. legs: the legs were not seen."""
    for joint, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        a, b = points[joint], points[parent]
        if np.isfinite(a).all() and np.isfinite(b).all():
            cv2.line(picture, tuple(int(v) for v in a), tuple(int(v) for v in b), colour(joint, estimated, legs), width, cv2.LINE_AA)
    for joint, point in enumerate(points):
        if np.isfinite(point).all():
            cv2.circle(picture, tuple(int(v) for v in point), width + 1, colour(joint, estimated, legs), -1, cv2.LINE_AA)
    head = points[15]
    if np.isfinite(head).all():
        cv2.circle(picture, tuple(int(v) for v in head), 3 * width + 6, colour(15, estimated), width, cv2.LINE_AA)


def label(text, size=30):
    """A text picture (BGR on black, with its alpha) for a fixed Korean label."""
    from paths import korean_font  # Noto Sans CJK KR wherever the system keeps it
    font = korean_font(size)
    left, top, right, bottom = font.getbbox(text)
    picture = Image.new('L', (right - left + 8, bottom - top + 8), 0)
    ImageDraw.Draw(picture).text((4 - left, 4 - top), text, fill=255, font=font)
    return np.asarray(picture)


def paste(picture, alpha, x, y, value=(255, 255, 255)):
    h, w = alpha.shape
    region = picture[y:y + h, x:x + w]
    a = alpha[:region.shape[0], :region.shape[1], None] / 255.0
    region[:] = (region * (1 - a) + np.array(value) * a).astype(np.uint8)


class Views:
    """Front and side views of the world joints at one scale for the whole video."""

    def __init__(self, joints, R_cam, size):
        self.size = size  # (width, height) of one view
        right, forward = R_cam[:, :, 0].mean(0), R_cam[:, :, 2].mean(0)  # the camera's x and z in the world
        right[1] = forward[1] = 0
        # front: looking along the viewing direction, x of the picture = the camera's right; side:
        # looking from the camera's left (the dancer's right when facing it), x = away from the camera
        self.axes = [right / np.linalg.norm(right), forward / np.linalg.norm(forward)]
        body = joints[:, :22]
        spread = []
        for axis in self.axes:
            along = body @ axis
            spread.append((float(np.percentile(along.min(1), 1)), float(np.percentile(along.max(1), 99))))
        top = float(np.percentile(body[..., 1].max(1), 99)) + 0.25
        reach = max(high - low for low, high in spread) + 0.6
        self.scale = min(size[0] / reach, (size[1] - 90) / top)
        self.centres = [(low + high) / 2 for low, high in spread]
        self.floor = size[1] - 60

    def project(self, view, points):
        x = (points @ self.axes[view] - self.centres[view]) * self.scale + self.size[0] / 2
        y = self.floor - points[:, 1] * self.scale
        return np.stack([x, y], axis=1)

    def draw(self, view, points, estimated, legs, labels):
        picture = np.full((self.size[1], self.size[0], 3), 24, np.uint8)
        cv2.line(picture, (0, self.floor), (self.size[0], self.floor), (90, 90, 90), 2)
        metre = int(round(self.scale))
        for step in range(-6, 7):  # floor marks every metre around the middle
            x = int(self.size[0] / 2 + step * metre)
            if 0 <= x < self.size[0]:
                cv2.line(picture, (x, self.floor), (x, self.floor + 10), (90, 90, 90), 2)
        draw_skeleton(picture, self.project(view, points[:22]), estimated, legs=legs)
        paste(picture, labels[view], 14, 12)
        return picture


class Writer:
    def __init__(self, path, size, fps):
        self.process = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s',
                                         f'{size[0]}x{size[1]}', '-r', f'{fps:.6f}', '-i', '-', '-c:v', 'libx264', '-crf', '20',
                                         '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
                                        stdin=subprocess.PIPE)

    def write(self, picture):
        self.process.stdin.write(picture.tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait():
            raise RuntimeError('ffmpeg failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--joints', required=True, help='folder written by world_one.py')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(Path(args.joints) / 'results.npz') as data:
        joints, estimated, fps = data['joints_global'].astype(np.float64), data['estimated'] > 0, float(data['fps'])
        legs = data['legs_estimated'] if 'legs_estimated' in data.files else np.zeros(len(joints), dtype=bool)
    with np.load(Path(args.joints) / 'camera_world.npz') as data:
        R_cam, centre, K = data['R_c2w'].astype(np.float64), data['center'].astype(np.float64), data['K'].astype(np.float64)
    count = len(joints)
    capture = cv2.VideoCapture(args.video)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    shrink = PANEL / height
    source_size = (int(round(width * shrink / 2)) * 2, PANEL)
    view_size = (PANEL // 2 if height >= width else PANEL * 2 // 3, PANEL)
    views = Views(joints, R_cam, view_size)
    labels = [label('정면'), label('옆')]
    over = label('원본 위 관절')
    plain = Writer(out / 'joints_front_side.mp4', (2 * view_size[0], PANEL), fps)
    both = Writer(out / 'joints_over_source.mp4', (source_size[0] + 2 * view_size[0], PANEL), fps)
    pelvis_text = label('골반 높이', 24)
    for index in range(count):
        ok, frame = capture.read()
        if not ok:
            frame = np.zeros((height, width, 3), np.uint8)
        side_by_side = np.concatenate([views.draw(view, joints[index], estimated[index], legs[index], labels) for view in (0, 1)], axis=1)
        paste(side_by_side, pelvis_text, 14, PANEL - 44)
        cv2.putText(side_by_side, f'{joints[index, 0, 1]:.2f} m    {index / fps:6.2f} s', (14 + pelvis_text.shape[1] + 10, PANEL - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        plain.write(side_by_side)
        picture = cv2.resize(frame, source_size, interpolation=cv2.INTER_AREA)
        seen = (joints[index, :22] - centre[index]) @ R_cam[index]  # camera coordinates
        pixels = seen @ K.T
        pixels = pixels[:, :2] / np.maximum(pixels[:, 2:], 1e-6) * shrink
        draw_skeleton(picture, pixels, estimated[index], width=3, legs=legs[index])
        paste(picture, over, 14, 12)
        both.write(np.concatenate([picture, side_by_side], axis=1))
    capture.release()
    plain.close()
    both.close()
    print('check videos written to', out)


if __name__ == '__main__':
    main()
