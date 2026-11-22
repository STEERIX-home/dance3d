"""Reuse an existing animated mesh scene and author a front-facing camera move.

Run with Blender's Python. Geometry and timing remain unchanged. Dolly moves
change the 3D camera distance at a fixed focal length, not a rendered crop.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector


# time, yaw, elevation, roll, requested dolly factor, explanatory phase
SHOTS = [
    (0.0,   0,  7,  0,   1.00, '정면 전신 · 시작'),
    (3.0, -10,  8, -1.5, 1.25, '정면에서 접근 · Dolly in'),
    (6.0, -24,  9, -2.5, 1.60, '왼쪽 아크 · 가까운 구도'),
    (9.0,  -6,  5,  1.0, 1.00, '다시 전신 · Dolly out'),
    (12.0, 22, 10,  2.5, 1.45, '오른쪽 아크 · 좌우 시차'),
    (15.0,  8, 13,  0,   1.18, '높이 변화 · 부드러운 하이앵글'),
    (18.0,-18,  6, -2.0, 1.55, '정면 접근 · 약한 롤'),
    (21.0, 12,  7,  2.0, 1.00, '전신으로 후퇴 · 동작 확인'),
    (24.0, 26,  9,  2.5, 1.50, '오른쪽에서 접근'),
    (26.5,  5,  5, -1.0, 1.80, '정면 가까운 구도 · 얼굴과 손'),
    (29.0833333333, 0, 7, 0, 1.00, '정면 전신 · 마무리'),
]


def smooth(values, radius):
    values = np.asarray(values)
    padded = np.pad(values, ((radius, radius),) + ((0, 0),)*(values.ndim-1), mode='edge')
    return np.stack([padded[i:i+2*radius+1].mean(axis=0) for i in range(len(values))])


def sample_shots(frames):
    times = np.linspace(0, SHOTS[-1][0], frames)
    knots = np.asarray([s[:5] for s in SHOTS], dtype=float)
    rows = []
    for t in times:
        index = min(len(knots)-2, max(0, int(np.searchsorted(knots[:, 0], t, side='right'))-1))
        a, b = knots[index:index+2]
        u = np.clip((t-a[0])/(b[0]-a[0]), 0, 1)
        u = u*u*u*(u*(u*6-15)+10)  # zero velocity/acceleration at shot accents
        rows.append((1-u)*a[1:]+u*b[1:])
    return np.asarray(rows)


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mesh_path = Path(args.mesh)
    with np.load(mesh_path, allow_pickle=False) as data:
        vertices = data['vertices'].astype(np.float32)
        faces = data['faces']
    assert vertices.shape == (699, 10475, 3), vertices.shape
    bpy.ops.wm.open_mainfile(filepath=args.source_blend)
    scene = bpy.context.scene
    body = bpy.data.objects['Motion guide']
    keys = body.data.shape_keys
    assert not keys.use_relative and len(keys.key_blocks) == len(vertices)
    buffer = np.empty(vertices.shape[1]*3, dtype=np.float32)
    errors = []
    for index, key in enumerate(keys.key_blocks):
        key.data.foreach_get('co', buffer)
        errors.append(float(np.abs(buffer.reshape(-1, 3)-vertices[index]).max()))
    assert max(errors) == 0, 'Source scene and saved motion differ'
    assert len(body.data.polygons) == len(faces)

    camera = scene.camera
    camera.animation_data_clear()
    camera.data.animation_data_clear()
    camera.data.lens = 50
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.rotation_mode = 'QUATERNION'
    scene.render.resolution_x = 768
    scene.render.resolution_y = 1344
    scene.render.resolution_percentage = 100
    scene.render.fps = 24
    scene.frame_start, scene.frame_end = 1, len(vertices)
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.cycles.device = 'CPU'  # preparing a scene never requires GPU compute
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = 'PNG'

    initial = vertices[0]
    center_x = np.median(initial[:, 0])
    pelvis_mask = ((np.abs(initial[:, 0]-center_x) < .12)
                   & (initial[:, 2] > .65) & (initial[:, 2] < .95))
    assert pelvis_mask.sum() > 30
    pelvis = smooth(vertices[:, pelvis_mask].mean(axis=1), 16)
    ground = float(vertices[:, :, 2].min())
    height = float(np.percentile(vertices[:, :, 2].max(axis=1)
                                 - vertices[:, :, 2].min(axis=1), 25))
    params = sample_shots(len(vertices))
    targets, rotations, directions = [], [], []
    tan_x = camera.data.sensor_width/(2*camera.data.lens)
    tan_y = tan_x*1344/768
    full_needs, upper_needs = [], []
    previous = None
    for index, (yaw, elevation, roll, zoom) in enumerate(params):
        yr, er = math.radians(yaw), math.radians(elevation)
        direction = Vector((math.sin(yr)*math.cos(er), -math.cos(yr)*math.cos(er), math.sin(er)))
        rotation = (-direction).to_track_quat('-Z', 'Y') @ Quaternion((0, 0, 1), math.radians(roll))
        if previous is not None and previous.dot(rotation) < 0:
            rotation.negate()
        previous = rotation.copy()
        base_z = ground+height*.52+.45*(pelvis[index, 2]-pelvis[0, 2])
        target = Vector((pelvis[index, 0], pelvis[index, 1], base_z+(zoom-1)*height*.30))
        inverse = np.asarray(rotation.to_matrix(), dtype=float).T
        local = (vertices[index]-np.asarray(target)) @ inverse.T
        required = np.maximum(np.abs(local[:, 0])/(tan_x*.91)+local[:, 2],
                              np.abs(local[:, 1])/(tan_y*.91)+local[:, 2])
        full_needs.append(float(required.max()))
        upper = vertices[index, :, 2] >= pelvis[index, 2]-.15
        upper_needs.append(float(required[upper].max()))
        targets.append(target)
        directions.append(direction)
        rotations.append(rotation)
    # Full frames use the complete body. Close frames protect head, torso and
    # hands while allowing an intentional lower-leg crop. An envelope anticipates
    # upcoming gestures, instead of making the camera jump to each pose.
    baseline = float(np.percentile(full_needs, 70))*1.07
    required = np.asarray([f if z <= 1.08 else u
                           for f, u, z in zip(full_needs, upper_needs, params[:, 3])])
    padded = np.pad(required, (16, 16), mode='edge')
    envelope = np.asarray([padded[i:i+33].max() for i in range(len(required))])
    envelope = smooth(envelope, 6)
    desired = baseline/params[:, 3]
    distance = .5*(desired+envelope+np.sqrt((desired-envelope)**2+.035**2))
    rows, full_bounds, upper_bounds = [], [], []
    for index, (p, target, direction, rotation, dist) in enumerate(zip(params, targets, directions, rotations, distance)):
        yaw, elevation, roll, zoom = map(float, p)
        frame = index+1
        camera.location = target+direction*float(dist)
        camera.rotation_quaternion = rotation
        camera.keyframe_insert('location', frame=frame)
        camera.keyframe_insert('rotation_quaternion', frame=frame)
        inverse = np.asarray(rotation.to_matrix(), dtype=float).T
        local = (vertices[index]-np.asarray(camera.location)) @ inverse.T
        assert (-local[:, 2]).min() > .1
        ndc = np.abs(local[:, :2]/(-local[:, 2:3]))/np.array([tan_x, tan_y])
        upper = vertices[index, :, 2] >= pelvis[index, 2]-.15
        full_bounds.append(float(ndc.max()))
        upper_bounds.append(float(ndc[upper].max()))
        assert upper_bounds[-1] < .93, (frame, upper_bounds[-1])
        if zoom <= 1.08:
            assert full_bounds[-1] < .93, (frame, full_bounds[-1])
        phase = min(len(SHOTS)-1, int(np.searchsorted([s[0] for s in SHOTS], index/24, side='right')))
        rows.append(dict(frame=frame,time_seconds=index/24,yaw=yaw,elevation=elevation,
            roll=roll,dolly_request=zoom,distance_metres=float(dist),lens_mm=50,
            target=list(target),location=list(camera.location),rotation_quaternion=list(rotation),
            phase=SHOTS[phase][5],upper_body_frame_extent=upper_bounds[-1],
            whole_body_frame_extent=full_bounds[-1]))
    for layer in camera.animation_data.action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for point in curve.keyframe_points:
                        point.interpolation = 'LINEAR'
    scene['camera_controls'] = json.dumps(dict(mode='front_dynamic',lens_mm=50,
        yaw_range=[-24,26],roll_range=[-2.5,2.5],motion_retimed=False))
    scene['camera_description'] = 'Front-facing dolly, limited arc, crane and subtle roll. One shared camera guide for all appearances.'
    camera.data.clip_end = 100
    scene.frame_set(1)
    report = dict(frames=len(vertices),fps=24,resolution=[768,1344],shape_key_max_error=max(errors),
        motion_retimed=False,source_mesh=str(mesh_path),source_scene=args.source_blend,
        source_mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        yaw_range=[float(params[:,0].min()),float(params[:,0].max())],
        distance_range=[float(distance.min()),float(distance.max())],
        roll_range=[float(params[:,2].min()),float(params[:,2].max())],lens_mm=50,
        max_upper_body_extent=max(upper_bounds),whole_body_visible_frames=sum(x<1 for x in full_bounds),
        intentional_close_framing=True,digital_zoom=False,renderer='Cycles CUDA 16 samples')
    (output/'camera-path.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (output/'camera-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (output/'shot-plan.json').write_text(json.dumps(SHOTS,ensure_ascii=False,indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'scene.blend'))
    print('CAMERA_VALIDATION',json.dumps(report,ensure_ascii=False),flush=True)


def render(args):
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Rendering requires a managed GPU lease')
    output = Path(args.output)
    bpy.ops.wm.open_mainfile(filepath=args.prepared)
    scene = bpy.context.scene
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'CUDA'
    preferences.get_devices()
    enabled = []
    for device in preferences.devices:
        device.use = device.type == 'CUDA'
        if device.use:
            enabled.append(device.name)
    if len(enabled) != 1:
        raise RuntimeError(f'Expected exactly one leased device, got {enabled}')
    scene.cycles.device = 'GPU'
    scene.render.use_persistent_data = True
    if args.preview_frames:
        directory = output/'previews'
        directory.mkdir(exist_ok=True)
        for frame in map(int,args.preview_frames.split(',')):
            scene.frame_set(frame)
            scene.render.filepath = str(directory/f'frame_{frame:04}.png')
            bpy.ops.render.render(write_still=True)
    else:
        directory = output/'frames'
        directory.mkdir(exist_ok=True)
        scene.render.filepath = str(directory/'frame_')
        bpy.ops.render.render(animation=True)
    print('RENDER_COMPLETED',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-blend')
    parser.add_argument('--mesh')
    parser.add_argument('--output',required=True)
    parser.add_argument('--prepared')
    parser.add_argument('--preview-frames')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.prepared:
        render(args)
    else:
        if not args.source_blend or not args.mesh:
            parser.error('--source-blend and --mesh are required for camera preparation')
        prepare(args)
