"""Run inside Blender. Save an editable animated scene, GLB preview and rendered frames."""
import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get('Principled BSDF')
    node.inputs['Base Color'].default_value = (*color, 1)
    node.inputs['Roughness'].default_value = 0.72
    return mat


def build(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.mesh, allow_pickle=False) as data:
        verts = np.asarray(data['vertices'], dtype=np.float32)
        faces = np.asarray(data['faces'], dtype=np.int32)
    if (verts.ndim != 3 or verts.shape[2] != 3 or not 2 <= len(verts) <= 1800
            or not np.isfinite(verts).all() or faces.ndim != 2 or faces.shape[1] != 3
            or faces.min() < 0 or faces.max() >= verts.shape[1]):
        raise ValueError('Invalid animated mesh')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    if args.cpu:
        scene.cycles.device = 'CPU'
    else:
        if not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('GPU rendering requires a managed GPU lease')
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        enabled = []
        for device in prefs.devices:
            device.use = device.type == 'CUDA'
            if device.use:
                enabled.append(device.name)
        if len(enabled) != 1:
            raise RuntimeError(f'Expected exactly one leased CUDA device, got {enabled}')
        scene.cycles.device = 'GPU'
        print('Blender leased CUDA device:', enabled, flush=True)
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = 24
    scene.frame_start, scene.frame_end = 1, len(verts)
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.view_transform = 'Standard'
    scene.world = bpy.data.worlds.new('Soft gray studio')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.45, 0.45, 0.45, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.7

    mesh = bpy.data.meshes.new('SMPL-X animated mesh')
    mesh.from_pydata(verts[0].tolist(), [], faces.tolist())
    mesh.update()
    body = bpy.data.objects.new('Motion guide', mesh)
    scene.collection.objects.link(body)
    body.data.materials.append(material('Gray mannequin', (0.42, 0.44, 0.47)))
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    body.shape_key_add(name='Frame 0001')
    keys = body.data.shape_keys
    keys.use_relative = False
    keys.key_blocks[0].interpolation = 'KEY_LINEAR'
    for index in range(1, len(verts)):
        key = body.shape_key_add(name=f'Frame {index + 1:04}')
        key.data.foreach_set('co', verts[index].ravel())
        key.interpolation = 'KEY_LINEAR'
    # Absolute shape-key times are assigned in increments of 10 by Blender.
    keys.eval_time = 0
    keys.keyframe_insert('eval_time', frame=1)
    keys.eval_time = (len(verts) - 1) * 10
    keys.keyframe_insert('eval_time', frame=len(verts))
    for layer in keys.animation_data.action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for key in curve.keyframe_points:
                        key.interpolation = 'LINEAR'

    low, high = verts.min(axis=(0, 1)), verts.max(axis=(0, 1))
    center = Vector(((low + high) / 2).tolist())
    bpy.ops.mesh.primitive_plane_add(size=max(20, float(np.max(high-low))*8), location=(center.x, center.y, float(low[2]) - 0.015))
    floor = bpy.context.object
    floor.name = 'Ground'
    floor.data.materials.append(material('Studio floor', (0.67, 0.67, 0.67)))
    for name, location, energy, size in [
        ('Key', (center.x-3, center.y-4, center.z+5), 900, 5),
        ('Fill', (center.x+4, center.y+2, center.z+4), 600, 4),
    ]:
        light = bpy.data.lights.new(name, 'AREA')
        light.energy, light.shape, light.size = energy, 'DISK', size
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (center - obj.location).to_track_quat('-Z', 'Y').to_euler()
    camera = bpy.data.objects.new('Guide camera', bpy.data.cameras.new('Guide camera'))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.lens = 50
    camera.data.sensor_fit = 'HORIZONTAL'
    elevation = math.radians(args.elevation)
    angles = np.linspace(args.yaw, args.yaw + args.orbit_degrees, len(verts))
    directions = [Vector((math.sin(math.radians(angle))*math.cos(elevation),
                         -math.cos(math.radians(angle))*math.cos(elevation), math.sin(elevation)))
                  for angle in angles]
    tan_x = camera.data.sensor_width / (2 * camera.data.lens)
    tan_y = tan_x * args.height / args.width
    distance = 0.0
    # Fit the entire moving body at every camera angle, without modifying the motion.
    for direction in directions:
        inverse = (-direction).to_track_quat('-Z', 'Y').to_matrix().transposed()
        for corner in itertools.product(*zip(low, high)):
            local = inverse @ (Vector(corner) - center)
            distance = max(distance, abs(local.x)/tan_x + local.z, abs(local.y)/tan_y + local.z)
    distance = max(distance * 1.15 / args.zoom, 1)
    camera.rotation_mode = 'QUATERNION'
    previous = None
    camera_path = []
    for frame, (angle, direction) in enumerate(zip(angles, directions), 1):
        rotation = (-direction).to_track_quat('-Z', 'Y')
        if previous is not None and previous.dot(rotation) < 0:
            rotation.negate()
        previous = rotation.copy()
        camera.location = center + direction * distance
        camera.rotation_quaternion = rotation
        camera.keyframe_insert('location', frame=frame)
        camera.keyframe_insert('rotation_quaternion', frame=frame)
        camera_path.append(dict(frame=frame, yaw=float(angle), location=list(camera.location)))
    for layer in camera.animation_data.action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for key in curve.keyframe_points:
                        key.interpolation = 'LINEAR'
    (out / 'camera-path.json').write_text(json.dumps(camera_path, indent=2))
    camera.data.clip_end = max(100, distance*5)
    scene.frame_set(1)
    scene.render.filepath = str(out / 'frames/frame_')
    (out / 'frames').mkdir(exist_ok=True)
    scene['motion_source'] = str(Path(args.mesh).resolve())
    scene['workflow'] = 'ExMR -> animated SMPL-X mesh -> Blender guide -> H3'
    scene['camera_controls'] = json.dumps(dict(yaw=args.yaw, elevation=args.elevation,
                                              zoom=args.zoom, orbit_degrees=args.orbit_degrees))
    for obj in scene.objects:
        obj.select_set(False)
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.export_scene.gltf(filepath=str(out / 'preview.glb'), export_format='GLB',
                              use_selection=True, export_animations=False, export_morph=False)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'scene.blend'))
    bpy.ops.render.render(animation=True)
    (out / 'render.json').write_text(json.dumps(dict(frames=len(verts), fps=24, width=args.width,
        height=args.height, yaw=args.yaw, elevation=args.elevation, zoom=args.zoom,
        orbit_degrees=args.orbit_degrees, motion_frames=len(verts), motion_retimed=False,
        renderer='Cycles CPU' if args.cpu else 'Cycles CUDA'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=1344)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--yaw', type=float, default=0)
    parser.add_argument('--elevation', type=float, default=8)
    parser.add_argument('--zoom', type=float, default=1)
    parser.add_argument('--orbit-degrees', type=float, choices=(0, 360), default=0)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--cpu', action='store_true')
    build(parser.parse_args(sys.argv[sys.argv.index('--') + 1:]))
