"""The dancer's 3D joints (results.npz of the dance-joints program) drawn as a glowing figure on a floor grid.

World: metres, y up, floor at y = 0. The figure is seen through a simple pinhole camera that can stand anywhere.
"""
import functools
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from kit import AMBER, DIM, GREEN, GREY, SKY, WHITE

UP = np.array([0.0, 1.0, 0.0])
CENTRE = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
LEFT = [(0, 1), (1, 4), (4, 7), (9, 13), (13, 16), (16, 18), (18, 20)]
RIGHT = [(0, 2), (2, 5), (5, 8), (9, 14), (14, 17), (17, 19), (19, 21)]
LEFT_FOOT, RIGHT_FOOT = [(7, 62), (62, 60), (7, 60), (60, 61), (61, 62)], [(8, 65), (65, 63), (8, 63), (63, 64), (64, 65)]
LEFT_HAND = [[20, 37, 38, 39, 66], [20, 25, 26, 27, 67], [20, 28, 29, 30, 68], [20, 34, 35, 36, 69], [20, 31, 32, 33, 70]]
RIGHT_HAND = [[21, 52, 53, 54, 71], [21, 40, 41, 42, 72], [21, 43, 44, 45, 73], [21, 49, 50, 51, 74], [21, 46, 47, 48, 75]]
NOSE, RIGHT_EAR, LEFT_EAR, HEAD, NECK, PELVIS = 55, 58, 59, 15, 12, 0
TRAILS = ((20, SKY), (21, AMBER), (7, SKY), (8, AMBER))


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class Dance:
    """One dance: joints over time, with the dancer's usual place and facing."""

    def __init__(self, folder):
        data = np.load(Path(folder) / 'results.npz', allow_pickle=True)
        self.joints = data['joints_global'].astype(np.float64)
        self.fps = float(data['fps'])
        self.frames = len(self.joints)
        self.seconds = self.frames / self.fps
        self.names = [str(n) for n in data['joint_names']]
        self.estimated = data['estimated'].astype(bool) if 'estimated' in data else np.zeros(self.frames, bool)
        self.joints2d = data['joints_2d'].astype(np.float64)       # pixels in the source picture
        self.contact = data['foot_contact'].astype(bool)           # (left, right) foot on the floor
        box = data['bbx_xyxy']
        self.picture = (int(round(float(data['K_fullimg'][0, 0, 2]) * 2)), int(round(float(data['K_fullimg'][0, 1, 2]) * 2)))
        pelvis = self.joints[:, PELVIS]
        # where the camera looks: the pelvis, slowly (the figure may move in the picture, the camera does not chase every hop)
        self.anchor = gaussian_filter1d(pelvis, self.fps * 0.8, axis=0, mode='nearest')
        self.centre = np.median(pelvis, axis=0)
        lateral = self.joints[:, 1] - self.joints[:, 2]
        forward = np.cross(lateral, UP)
        forward[:, 1] = 0
        self.forward = unit(np.median(unit(forward), axis=0))  # the way the dancer mostly faces
        self.height = float(np.percentile(self.joints[:, HEAD, 1], 95) + 0.12)

    def pose(self, seconds):
        x = np.clip(seconds * self.fps, 0, self.frames - 1)
        a = int(np.floor(x))
        b = min(a + 1, self.frames - 1)
        w = x - a
        return self.joints[a] * (1 - w) + self.joints[b] * w

    def pose2d(self, seconds):
        x = np.clip(seconds * self.fps, 0, self.frames - 1)
        a = int(np.floor(x))
        b = min(a + 1, self.frames - 1)
        return self.joints2d[a] * (1 - (x - a)) + self.joints2d[b] * (x - a)

    def index(self, seconds):
        return int(np.clip(round(seconds * self.fps), 0, self.frames - 1))

    def anchor_at(self, seconds):
        return self.anchor[int(np.clip(round(seconds * self.fps), 0, self.frames - 1))]

    def track(self, joint, seconds, back=0.6, steps=14):
        """Where a joint was during the last `back` seconds (oldest first)."""
        times = seconds - back + back * np.arange(steps + 1) / steps
        return np.stack([self.pose(max(0.0, t))[joint] for t in times])


class Camera:
    def __init__(self, position, target, focal, centre, mirror=False):
        self.position = np.asarray(position, float)
        forward = unit(np.asarray(target, float) - self.position)
        right = unit(np.cross(forward, UP))
        self.axes = np.stack([right, np.cross(right, forward), forward])
        self.focal, self.centre, self.mirror = float(focal), centre, mirror

    def project(self, points):
        """Picture positions (N, 2) and depths (N,) of world points; depth <= 0.05 means behind the camera."""
        local = (np.asarray(points, float) - self.position) @ self.axes.T
        depth = local[:, 2]
        safe = np.maximum(depth, 0.05)
        x = self.focal * local[:, 0] / safe
        if self.mirror:
            x = -x
        return np.stack([self.centre[0] + x, self.centre[1] - self.focal * local[:, 1] / safe], axis=1), depth

    def segment(self, a, b):
        """A world segment cut at the near plane, as picture points; None if all of it is behind the camera."""
        pts = np.stack([a, b]).astype(float)
        local = (pts - self.position) @ self.axes.T
        near = 0.15
        if local[0, 2] < near and local[1, 2] < near:
            return None
        if local[0, 2] < near or local[1, 2] < near:
            i = 0 if local[0, 2] < near else 1
            w = (near - local[i, 2]) / (local[1 - i, 2] - local[i, 2])
            pts[i] = pts[i] + w * (pts[1 - i] - pts[i])
        return self.project(pts)[0]


def orbit_camera(dance, seconds, yaw_deg, distance, height, focal, centre, look_height=None, mirror=False):
    """A camera `distance` away from the dancer, `yaw_deg` round from the front (0 = in front, 90 = at the viewer's right)."""
    anchor = dance.anchor_at(seconds).copy()
    target = np.array([anchor[0], dance.height * 0.5 if look_height is None else look_height, anchor[2]])
    side = np.cross(UP, dance.forward)   # the dancer's left = the right of a viewer standing in front
    angle = np.radians(yaw_deg)
    direction = np.cos(angle) * dance.forward + np.sin(angle) * side
    position = np.array([anchor[0], 0.0, anchor[2]]) + direction * distance + UP * height
    return Camera(position, target, focal, centre, mirror)


def scaled(colour, factor):
    return tuple(int(min(255, c * factor)) for c in colour)


def draw_floor(layer, camera, around, offset=(0, 0), reach=2.6, step=0.5, colour=DIM):
    """A floor grid around a point, fading with distance."""
    ox, oy = offset
    cx, cz = np.round(around[0] / step) * step, np.round(around[2] / step) * step
    count = int(reach / step)
    for k in range(-count, count + 1):
        fade = max(0.0, 1 - abs(k) / (count + 1)) ** 1.5
        for a, b in ((np.array([cx + k * step, 0, cz - reach]), np.array([cx + k * step, 0, cz + reach])),
                     (np.array([cx - reach, 0, cz + k * step]), np.array([cx + reach, 0, cz + k * step]))):
            # in pieces, so that the far ends fade too
            parts = 8
            for p in range(parts):
                s0, s1 = p / parts, (p + 1) / parts
                edge = 1 - abs((s0 + s1) / 2 - 0.5) * 2
                seg = camera.segment(a + (b - a) * s0, a + (b - a) * s1)
                if seg is None:
                    continue
                tone = scaled(colour, 0.25 + 0.75 * fade * edge ** 0.8)
                cv2.line(layer, (int(seg[0, 0] - ox), int(seg[0, 1] - oy)), (int(seg[1, 0] - ox), int(seg[1, 1] - oy)), tone, 1, cv2.LINE_AA)


def draw_figure(layer, camera, pose, offset=(0, 0), thickness=1.0, tint=None, hands=True, head=True, strength=1.0):
    """The figure as bright limbs: left sky blue, right amber, spine green (or one tint for all)."""
    ox, oy = offset
    uv, depth = camera.project(pose)
    scale = camera.focal / np.maximum(depth, 0.3)  # pixels per metre at each joint
    items = []
    for bones, colour, wide in ((CENTRE, GREEN, 0.034), (LEFT, SKY, 0.03), (RIGHT, AMBER, 0.03), (LEFT_FOOT, SKY, 0.016), (RIGHT_FOOT, AMBER, 0.016)):
        for a, b in bones:
            items.append(((depth[a] + depth[b]) / 2, a, b, tint or colour, wide))
    if hands:
        for chains, colour in ((LEFT_HAND, SKY), (RIGHT_HAND, AMBER)):
            for chain in chains:
                for a, b in zip(chain[:-1], chain[1:]):
                    items.append(((depth[a] + depth[b]) / 2, a, b, tint or colour, 0.009))
    span = max(1e-6, depth.max() - depth.min())
    for mean, a, b, colour, wide in sorted(items, key=lambda item: -item[0]):
        if depth[a] < 0.1 or depth[b] < 0.1:
            continue
        near = 1 - (mean - depth.min()) / span          # 1 = nearest limb
        tone = scaled(colour, (0.55 + 0.45 * near) * strength)
        width = max(1, int(round(wide * (scale[a] + scale[b]) / 2 * thickness)))
        pa, pb = (int(uv[a, 0] - ox), int(uv[a, 1] - oy)), (int(uv[b, 0] - ox), int(uv[b, 1] - oy))
        cv2.line(layer, pa, pb, tone, width, cv2.LINE_AA)
        if wide >= 0.03:
            core = scaled(tuple(int(c + (255 - c) * 0.55) for c in colour), (0.6 + 0.4 * near) * strength)
            cv2.line(layer, pa, pb, core, max(1, width // 3), cv2.LINE_AA)
    for index in (1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21, 12):
        if depth[index] > 0.1:
            radius = max(2, int(round(0.022 * scale[index] * thickness)))
            cv2.circle(layer, (int(uv[index, 0] - ox), int(uv[index, 1] - oy)), radius, scaled(WHITE, strength), -1, cv2.LINE_AA)
    if head and depth[HEAD] > 0.1:
        middle = (pose[LEFT_EAR] + pose[RIGHT_EAR]) / 2
        centre_uv, centre_depth = camera.project(middle[None])
        radius = max(3, int(round(0.105 * camera.focal / max(0.3, centre_depth[0]) * thickness)))
        point = (int(centre_uv[0, 0] - ox), int(centre_uv[0, 1] - oy))
        cv2.circle(layer, point, radius, scaled(tint or GREEN, strength), max(1, int(round(0.03 * scale[HEAD] * thickness))), cv2.LINE_AA)
        nose_uv = camera.project((middle + (pose[NOSE] - middle) * 1.25)[None])[0][0]
        cv2.line(layer, point, (int(nose_uv[0] - ox), int(nose_uv[1] - oy)), scaled(WHITE, 0.9 * strength), max(1, radius // 6), cv2.LINE_AA)


def draw_trails(layer, camera, dance, seconds, offset=(0, 0), back=0.55, strength=1.0):
    ox, oy = offset
    for joint, colour in TRAILS:
        path = dance.track(joint, seconds, back)
        uv, depth = camera.project(path)
        for k in range(len(uv) - 1):
            if depth[k] < 0.1 or depth[k + 1] < 0.1:
                continue
            age = (k + 1) / (len(uv) - 1)
            cv2.line(layer, (int(uv[k, 0] - ox), int(uv[k, 1] - oy)), (int(uv[k + 1, 0] - ox), int(uv[k + 1, 1] - oy)),
                     scaled(colour, 0.5 * age ** 1.5 * strength), max(1, int(3 * age)), cv2.LINE_AA)


def shadow(canvas, camera, pose, region):
    """A soft dark-on-dark hint of the floor under the feet: a brighter disc on the grid."""
    x, y, w, h = region
    feet = (pose[7] + pose[8]) / 2
    spot = camera.project(np.array([[feet[0], 0.0, feet[2]]]))[0][0]
    ring = camera.project(np.array([[feet[0] + 0.45, 0.0, feet[2]], [feet[0], 0.0, feet[2] + 0.45]]))[0]
    rx = max(4, int(np.hypot(*(ring[0] - spot))))
    ry = max(2, int(np.hypot(*(ring[1] - spot)) * 0.5 + 0.5 * rx * 0.3))
    overlay = np.zeros((h, w, 3), np.uint8)
    cv2.ellipse(overlay, (int(spot[0] - x), int(spot[1] - y)), (rx, max(2, min(ry, rx))), 0, 0, 360, (26, 34, 30), -1, cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (0, 0), 12)
    part = canvas[y:y + h, x:x + w]
    part[:] = np.clip(part.astype(np.int16) + overlay, 0, 255).astype(np.uint8)


def figure_panel(canvas, dance, seconds, region, yaw_deg=0.0, distance=4.2, height=1.25, zoom=1.0, mirror=False, trails=True, floor=True,
                 tint=None, strength=1.0, hands=True, glow_strength=0.9, focus_height=None, shift=(0.0, 0.0)):
    """Draw the dance at `seconds` into region (x, y, w, h) of the canvas, seen from yaw_deg round the dancer."""
    from kit import glow
    x, y, w, h = [int(v) for v in region]
    focal = 1.9 * h * zoom
    centre = (x + w / 2 + shift[0] * w, y + h * 0.5 + shift[1] * h)
    camera = orbit_camera(dance, seconds, yaw_deg, distance, height, focal, centre, look_height=focus_height, mirror=mirror)
    pose = dance.pose(seconds)
    if floor:
        shadow(canvas, camera, pose, (x, y, w, h))

    def floor_only(layer, offset):
        draw_floor(layer, camera, dance.centre, offset)

    def figure_only(layer, offset):
        if trails:
            draw_trails(layer, camera, dance, seconds, offset, strength=strength)
        draw_figure(layer, camera, pose, offset, thickness=1.0, tint=tint, hands=hands, strength=strength)

    if floor:
        glow(canvas, floor_only, 0.0, region=(x, y, w, h), feather=max(8, int(min(w, h) * 0.14)))   # the grid fades towards the sides
    glow(canvas, figure_only, glow_strength, sigma=max(3, h / 110), region=(x, y, w, h))
    return camera


class Motion(Dance):
    """A dancer given as joints over time (not read from a results.npz)."""

    def __init__(self, joints, fps):
        self.joints = np.asarray(joints, np.float64)
        self.fps = float(fps)
        self.frames = len(self.joints)
        self.seconds = self.frames / self.fps
        pelvis = self.joints[:, PELVIS]
        self.anchor = gaussian_filter1d(pelvis, self.fps * 0.8, axis=0, mode='nearest')
        self.centre = np.median(pelvis, axis=0)
        forward = np.cross(self.joints[:, 1] - self.joints[:, 2], UP)
        forward[:, 1] = 0
        self.forward = unit(np.median(unit(forward), axis=0))
        self.height = float(np.percentile(self.joints[:, HEAD, 1], 95) + 0.12)


class Stage:
    """Several dancers sharing one floor: the camera looks at the middle of them."""

    def __init__(self, dancers):
        self.dancers = dancers
        self.fps = dancers[0].fps
        count = min(d.frames for d in dancers)
        self.anchor = np.mean([d.anchor[:count] for d in dancers], axis=0)
        self.centre = np.mean([d.centre for d in dancers], axis=0)
        self.forward = unit(np.mean([d.forward for d in dancers], axis=0))
        self.height = max(d.height for d in dancers)
        self.frames, self.seconds = count, count / self.fps

    def anchor_at(self, seconds):
        return self.anchor[int(np.clip(round(seconds * self.fps), 0, self.frames - 1))]


def stage_panel(canvas, stage, seconds, region, colours, yaw_deg=0.0, distance=5.2, height=1.5, zoom=1.0, glow_strength=0.9, strength=1.0,
                trails=True, hands=False):
    """Draw all dancers of a stage, each in its own colour, on one floor grid."""
    from kit import glow
    x, y, w, h = [int(v) for v in region]
    camera = orbit_camera(stage, seconds, yaw_deg, distance, height, 1.9 * h * zoom, (x + w / 2, y + h * 0.5))
    poses = [d.pose(seconds) for d in stage.dancers]
    for pose in poses:
        shadow(canvas, camera, pose, (x, y, w, h))

    def floor_only(layer, offset):
        draw_floor(layer, camera, stage.centre, offset, reach=3.2)

    def figures(layer, offset):
        order = sorted(range(len(poses)), key=lambda k: -camera.project(poses[k][[PELVIS]])[1][0])   # far dancer first
        for k in order:
            if trails:
                ox, oy = offset
                for joint in (20, 21):
                    path = stage.dancers[k].track(joint, seconds, 0.5)
                    uv, depth = camera.project(path)
                    for i in range(len(uv) - 1):
                        if depth[i] > 0.1 and depth[i + 1] > 0.1:
                            age = (i + 1) / (len(uv) - 1)
                            cv2.line(layer, (int(uv[i, 0] - ox), int(uv[i, 1] - oy)), (int(uv[i + 1, 0] - ox), int(uv[i + 1, 1] - oy)),
                                     scaled(colours[k], 0.5 * age ** 1.5 * strength), max(1, int(3 * age)), cv2.LINE_AA)
            draw_figure(layer, camera, poses[k], offset, tint=colours[k], hands=hands, strength=strength)

    glow(canvas, floor_only, 0.0, region=(x, y, w, h), feather=max(8, int(min(w, h) * 0.14)))
    glow(canvas, figures, glow_strength, sigma=max(3, h / 110), region=(x, y, w, h))
    return camera
