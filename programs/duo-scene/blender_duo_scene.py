"""Run inside Blender: two reconstructed dancers in one fixed 3D studio set, filmed by one camera.

    blender -b --factory-startup --python blender_duo_scene.py -- --inputs DIR --output DIR --camera orbit

The set (floor with tape marks, four different walls, ceiling lights, props) is built once from
the fixed description in this file and never changes between frames or camera paths; only the
camera moves. Both dancers and the set are always seen by the same camera.

Camera paths:  original  the reconstructed camera of the source video
               front     static, both dancers in full for the whole clip
               orbit     one full 360 degree turn around the dancers
               dynamic   front-side moves: approach, retreat, arc, height change, slight roll
Passes:        guide     set + grey bodies of all dancers
               plate     the set alone, with the shadows the dancers cast (the final background)
               solo      set + one grey body, once per dancer (solo_A, solo_B, ...): drives the
                         appearance model for that dancer alone, never hidden by anybody
               mask      who is where: one flat colour per dancer on white, no shading
               solomask  the same for one dancer at a time (mask_A, mask_B, ...): the whole
                         outline, also where another dancer stands in front
               pair      set + all dancers as shaded mannequins, each in its own colour (A blue,
                         B red): the guide for giving all dancers their characters in ONE run,
                         where a text prompt tells the model which colour becomes which character
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

MASK_COLORS = {'A': (0.0, 0.0, 1.0), 'B': (1.0, 0.0, 0.0)}  # the palette SCAIL-2 was trained with
PAIR_COLORS = {'A': (0.08, 0.22, 0.85), 'B': (0.85, 0.10, 0.08)}  # shaded mannequins told apart by colour name
ROOM_HEIGHT = 4.2
SET_VERSION = 'studio-1'
# Lights of the set. 'even' (the first one used): very soft light from everywhere, hardly any shadow
# on the floor - bodies look laid over the room. 'shadow' has one main light from the front left,
# so that every body throws a clear soft shadow on the floor and darkens the floor where it stands.
# top: watts and disk size (m) of the four ceiling lights; key: the main light (a tilted disk also
# lights the ceiling in front of it as a white patch, a spot does not; cone in degrees, radius =
# softness of its shadows in m); ambient: world strength; floor: roughness and specular level of
# the wooden floor (a shinier floor mirrors the main light as a white patch when the camera looks
# towards it)
LIGHTING = {'even': dict(top=(85, 2.6), key=dict(kind='AREA', watts=110, size=2.0), ambient=0.22, floor=(0.42, 0.5)),
            'shadow': dict(top=(40, 1.8), key=dict(kind='SPOT', watts=2600, cone=100, blend=0.6, radius=0.55),
                           ambient=0.10, floor=(0.7, 0.2))}
# How closely the following cameras (front, orbit, dynamic) frame the pair. 'wide' (the first one
# used) aims at the middle of the two pelvises; 'close' aims at the middle of the space the two
# bodies take up and leaves less room around them, so the dancers are larger in the picture.
# 'steady' keeps ONE distance for the whole clip (the one at which both bodies fit in 97 of 100 frames),
# so the dancers do not shrink and grow with every wide move; in the few widest moments a hand or a
# foot leaves the picture, as in a hand-filmed dance video. The other two change the distance all
# the time so that nothing ever leaves the picture, which shows as dancers pumping in size.
FRAMING = {'wide': dict(aim='pelvis', full=0.88, upper=0.90), 'close': dict(aim='extent', full=0.94, upper=0.95),
           'steady': dict(aim='extent', full=0.94, upper=0.95, fits_share=97)}
# time share, yaw, elevation, roll, dolly (1 = both dancers in full), like the single-person version
DYNAMIC_SHOTS = [(0.00, 0, 7, 0, 1.00), (0.12, -10, 8, -1.5, 1.20), (0.24, -24, 9, -2.5, 1.45),
                 (0.36, -6, 5, 1.0, 1.00), (0.48, 22, 10, 2.5, 1.35), (0.60, 8, 14, 0, 1.12),
                 (0.72, -18, 6, -2.0, 1.45), (0.84, 12, 7, 2.0, 1.00), (0.93, 26, 9, 2.5, 1.35),
                 (1.00, 0, 7, 0, 1.00)]


def read_pc2(path):
    with open(path, 'rb') as file:
        _, _, points, _, _, samples = struct.unpack('<12siiffi', file.read(32))
        return np.frombuffer(file.read(), dtype='<f4').reshape(samples, points, 3)


def material(name, color, roughness=0.6, emission=0.0, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes['Principled BSDF']
    node.inputs['Base Color'].default_value = (*color, 1)
    node.inputs['Roughness'].default_value = roughness
    node.inputs['Metallic'].default_value = metallic
    if emission:
        node.inputs['Emission Color'].default_value = (*color, 1)
        node.inputs['Emission Strength'].default_value = emission
    return mat


def flat_material(name, color):
    """Unshaded colour, for the identity mask."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*color, 1)
    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return mat


def wood_floor(roughness=0.42, specular=0.5):
    mat = bpy.data.materials.new('Oak floor')
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes['Principled BSDF']
    coords = nodes.new('ShaderNodeTexCoord')
    planks = nodes.new('ShaderNodeTexBrick')
    planks.offset, planks.squash, planks.squash_frequency = 0.5, 1.0, 1
    planks.inputs['Color1'].default_value = (0.62, 0.44, 0.27, 1)
    planks.inputs['Color2'].default_value = (0.70, 0.52, 0.33, 1)
    planks.inputs['Mortar'].default_value = (0.30, 0.20, 0.12, 1)
    planks.inputs['Scale'].default_value = 1.0
    planks.inputs['Mortar Size'].default_value = 0.004
    planks.inputs['Bias'].default_value = 0.0
    planks.inputs['Brick Width'].default_value = 1.4
    planks.inputs['Row Height'].default_value = 0.16
    grain = nodes.new('ShaderNodeTexNoise')
    grain.inputs['Scale'].default_value = 3.0
    grain.inputs['Detail'].default_value = 6.0
    stretch = nodes.new('ShaderNodeMapping')
    stretch.inputs['Scale'].default_value = (1.0, 14.0, 1.0)
    mix = nodes.new('ShaderNodeMixRGB')
    mix.blend_type, mix.inputs['Fac'].default_value = 'MULTIPLY', 0.35
    links.new(coords.outputs['Object'], planks.inputs['Vector'])
    links.new(coords.outputs['Object'], stretch.inputs['Vector'])
    links.new(stretch.outputs['Vector'], grain.inputs['Vector'])
    links.new(planks.outputs['Color'], mix.inputs['Color1'])
    links.new(grain.outputs['Fac'], mix.inputs['Color2'])
    links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Specular IOR Level'].default_value = specular
    return mat


class SetBuilder:
    """Builds the studio in its own frame: dancers around the origin, the audience side at -y."""

    def __init__(self, scene, half):
        self.scene, self.half, self.objects, self.spec = scene, half, [], []
        self.root = bpy.data.objects.new('Studio set', None)
        scene.collection.objects.link(self.root)

    def box(self, name, size, location, mat, rotation_z=0.0):
        bpy.ops.mesh.primitive_cube_add(size=1, location=location, rotation=(0, 0, rotation_z))
        obj = bpy.context.object
        obj.name, obj.scale = name, size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)  # metric texture coordinates
        return self._keep(obj, mat, ('box', name, size, location, rotation_z))

    def cylinder(self, name, radius, depth, location, mat, rotation=(0, 0, 0), vertices=32):
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location, rotation=rotation, vertices=vertices)
        obj = bpy.context.object
        obj.name = name
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        return self._keep(obj, mat, ('cylinder', name, radius, depth, location, rotation))

    def sphere(self, name, radius, location, mat):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, segments=32, ring_count=16)
        obj = bpy.context.object
        obj.name = name
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        return self._keep(obj, mat, ('sphere', name, radius, location))

    def _keep(self, obj, mat, spec):
        obj.data.materials.append(mat)
        obj.parent = self.root
        self.objects.append(obj)
        self.spec.append([spec[0], spec[1], *[list(v) if isinstance(v, (tuple, list)) else v for v in spec[2:]],
                          mat.name])
        return obj

    def build(self, lighting='even'):
        S, H = self.half, ROOM_HEIGHT
        lights = LIGHTING[lighting]
        plaster = material('Warm plaster', (0.80, 0.78, 0.73), 0.9)
        dark = material('Slate wall', (0.20, 0.24, 0.30), 0.85)
        wood_dark = material('Dark wood', (0.28, 0.18, 0.11), 0.6)
        wood_light = material('Light wood', (0.72, 0.56, 0.38), 0.55)
        white = material('Ceiling white', (0.88, 0.88, 0.86), 0.9)
        black = material('Black', (0.03, 0.03, 0.035), 0.5)
        steel = material('Steel', (0.62, 0.64, 0.66), 0.3, metallic=0.9)
        tape = material('Floor tape', (0.93, 0.93, 0.90), 0.7)
        glass = material('Window light', (0.72, 0.86, 1.0), 0.3, emission=0.9)
        lamp = material('Lamp light', (1.0, 0.86, 0.62), 0.3, emission=1.6)
        panel = material('Ceiling light', (1.0, 0.97, 0.92), 0.3, emission=1.1)

        self.box('Floor', (2 * S, 2 * S, 0.1), (0, 0, -0.05), wood_floor(*lights['floor']))
        self.box('Ceiling', (2 * S, 2 * S, 0.1), (0, 0, H + 0.05), white)
        # floor tape: a 4 m square with centre cross, so sliding feet and turning are easy to see
        for name, size, location in [('Tape front', (4.0, 0.03, 0.002), (0, -2, 0.001)), ('Tape back', (4.0, 0.03, 0.002), (0, 2, 0.001)),
                                     ('Tape left', (0.03, 4.0, 0.002), (-2, 0, 0.001)), ('Tape right', (0.03, 4.0, 0.002), (2, 0, 0.001)),
                                     ('Tape cross x', (0.6, 0.03, 0.002), (0, 0, 0.001)), ('Tape cross y', (0.03, 0.6, 0.002), (0, 0, 0.001))]:
            self.box(name, size, location, tape)

        # back wall (+y): plaster, three coloured panels, a barre
        self.box('Back wall', (2 * S, 0.2, H), (0, S + 0.1, H / 2), plaster)
        for index, (color, x) in enumerate([((0.05, 0.50, 0.52), -3.4), ((0.90, 0.45, 0.10), 0.0), ((0.45, 0.22, 0.62), 3.4)]):
            self.box(f'Back panel {index + 1}', (1.7, 0.06, 2.3), (x, S - 0.03, 2.25), material(f'Panel colour {index + 1}', color, 0.8))
        self.cylinder('Barre', 0.025, 2 * S - 2.0, (0, S - 0.28, 1.02), steel, rotation=(0, math.pi / 2, 0))
        for x in (-S + 1.4, -S / 3, S / 3, S - 1.4):
            self.cylinder(f'Barre post {x:.1f}', 0.02, 1.02, (x, S - 0.28, 0.51), steel)

        # right wall (+x): wooden slats and a door
        self.box('Right wall', (0.2, 2 * S, H), (S + 0.1, 0, H / 2), wood_dark)
        slat = 0.0
        y = -S + 0.2
        while y < S - 0.1:
            if not -0.2 < y - 2.6 < 1.3:  # leave room for the door
                self.box(f'Slat {slat:.0f}', (0.04, 0.12, H - 0.3), (S - 0.02, y, H / 2), wood_light)
                slat += 1
            y += 0.34
        self.box('Door', (0.06, 1.05, 2.15), (S - 0.03, 3.15, 1.075), material('Door green', (0.12, 0.33, 0.27), 0.5))
        self.sphere('Door handle', 0.04, (S - 0.09, 2.78, 1.05), steel)

        # left wall (-x): four tall windows
        self.box('Left wall', (0.2, 2 * S, H), (-S - 0.1, 0, H / 2), plaster)
        for index, y in enumerate(np.linspace(-S + 2.2, S - 2.2, 4)):
            self.box(f'Window {index + 1}', (0.05, 1.9, 2.7), (-S + 0.025, float(y), 2.0), glass)
            self.box(f'Window bar v{index + 1}', (0.07, 0.06, 2.7), (-S + 0.035, float(y), 2.0), black)
            self.box(f'Window bar h{index + 1}', (0.07, 1.9, 0.06), (-S + 0.035, float(y), 2.3), black)

        # front wall (-y, behind the default camera): dark wall, shelf with coloured boxes, a clock
        self.box('Front wall', (2 * S, 0.2, H), (0, -S - 0.1, H / 2), dark)
        self.box('Shelf', (5.0, 0.35, 0.05), (0, -S + 0.175, 1.5), wood_light)
        for index, color in enumerate([(0.85, 0.12, 0.10), (0.95, 0.75, 0.10), (0.10, 0.35, 0.80), (0.15, 0.60, 0.25), (0.92, 0.92, 0.90)]):
            size = 0.28 + 0.06 * (index % 3)
            self.box(f'Shelf box {index + 1}', (size, 0.26, size), (-2.0 + index * 1.0, -S + 0.18, 1.525 + size / 2), material(f'Box colour {index + 1}', color, 0.6))
        self.cylinder('Clock', 0.32, 0.05, (3.6, -S + 0.03, 2.9), white, rotation=(math.pi / 2, 0, 0))
        self.box('Clock hand', (0.03, 0.02, 0.24), (3.6, -S + 0.06, 3.0), black)

        # ceiling lights: 3 x 3 panels
        for i, x in enumerate((-S * 0.55, 0, S * 0.55)):
            for j, y in enumerate((-S * 0.55, 0, S * 0.55)):
                self.box(f'Ceiling light {i}{j}', (1.5, 0.35, 0.04), (x, y, H - 0.02), panel)

        # props near the walls, each one different, so the direction of view is always recognisable
        near = S - 0.9
        for side, x in (('left', -near + 0.2), ('right', near - 0.2)):
            self.box(f'Speaker {side}', (0.5, 0.45, 1.15), (x, near - 0.1, 0.575), black, rotation_z=0.5 if side == 'left' else -0.5)
            turn = 0.5 if side == 'left' else -0.5
            for k, z in enumerate((0.38, 0.86)):
                self.cylinder(f'Speaker cone {side} {k}', 0.17 - 0.05 * k, 0.03,
                              (x + 0.23 * math.sin(turn), near - 0.1 - 0.23 * math.cos(turn), z), steel,
                              rotation=(math.pi / 2, 0, turn))
        self.cylinder('Plant pot', 0.26, 0.5, (near, -near, 0.25), material('Terracotta', (0.66, 0.31, 0.18), 0.8))
        leaf = material('Leaf green', (0.10, 0.42, 0.16), 0.7)
        for k, (dx, dy, dz, r) in enumerate([(0, 0, 0.95, 0.42), (0.22, 0.1, 1.35, 0.3), (-0.2, -0.12, 1.3, 0.28), (0.02, 0.18, 1.7, 0.22)]):
            self.sphere(f'Plant leaves {k}', r, (near + dx, -near + dy, dz), leaf)
        self.box('Bench seat', (0.42, 1.9, 0.07), (-near - 0.2, 0.6, 0.45), wood_light)
        for y in (-0.2, 1.4):
            self.box(f'Bench leg {y}', (0.36, 0.06, 0.42), (-near - 0.2, y, 0.21), black)
        for k, (color, offset, turn) in enumerate([((0.80, 0.10, 0.10), (0, 0), 0.2), ((0.95, 0.72, 0.08), (0.05, -0.04), 0.55), ((0.10, 0.30, 0.75), (-0.03, 0.05), 0.0)]):
            self.box(f'Cube stack {k + 1}', (0.5, 0.5, 0.5), (-near + offset[0], -near + offset[1], 0.25 + 0.5 * k),
                     material(f'Cube colour {k + 1}', color, 0.55), rotation_z=turn)
        self.cylinder('Lamp pole', 0.02, 1.7, (near - 0.1, 1.2, 0.85), steel)
        self.sphere('Lamp shade', 0.17, (near - 0.1, 1.2, 1.8), lamp)

        # lights
        for index, (x, y) in enumerate([(-2.6, -2.6), (2.6, -2.6), (-2.6, 2.6), (2.6, 2.6)]):
            light = bpy.data.lights.new(f'Soft top {index + 1}', 'AREA')
            light.energy, light.shape, light.size = lights['top'][0], 'DISK', lights['top'][1]
            obj = bpy.data.objects.new(light.name, light)
            self.scene.collection.objects.link(obj)
            obj.location, obj.parent = (x, y, ROOM_HEIGHT - 0.25), self.root
        key = bpy.data.lights.new('Front key', lights['key']['kind'])
        key.energy = lights['key']['watts']
        if key.type == 'AREA':
            key.shape, key.size = 'DISK', lights['key']['size']
        else:
            key.spot_size, key.spot_blend = math.radians(lights['key']['cone']), lights['key']['blend']
            key.shadow_soft_size = lights['key']['radius']
        obj = bpy.data.objects.new(key.name, key)
        self.scene.collection.objects.link(obj)
        obj.location, obj.parent = (-2.2, -4.2, 3.3), self.root
        obj.rotation_euler = (Vector((0, 0, 1.2)) - obj.location).to_track_quat('-Z', 'Y').to_euler()
        return hashlib.sha256(json.dumps([SET_VERSION, round(self.half, 3), self.spec], sort_keys=True, default=float).encode()).hexdigest()


def fit_distance(points, target, rotation, tan_x, tan_y, margin=0.9):
    """Smallest camera distance along the view direction at which all points are inside the frame."""
    inverse = np.asarray(rotation.to_matrix(), dtype=float).T
    local = (points - np.asarray(target)) @ inverse.T  # camera axes: x right, y up, z towards the camera
    return float(np.maximum(np.abs(local[:, 0]) / (tan_x * margin) + local[:, 2],
                            np.abs(local[:, 1]) / (tan_y * margin) + local[:, 2]).max())


def view_rotation(direction, roll_deg=0.0):
    return (-direction).to_track_quat('-Z', 'Y') @ Quaternion((0, 0, 1), math.radians(roll_deg))


def direction_at(front_angle, yaw_deg, elevation_deg):
    """Unit vector from the dancers towards the camera: yaw 0 is the front (audience) side."""
    yaw, elevation = front_angle + math.radians(yaw_deg), math.radians(elevation_deg)
    return Vector((math.cos(yaw) * math.cos(elevation), math.sin(yaw) * math.cos(elevation), math.sin(elevation)))


def smootherstep(u):
    return u * u * u * (u * (u * 6 - 15) + 10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--camera', required=True, choices=['original', 'front', 'orbit', 'dynamic', 'views'])
    parser.add_argument('--width', type=int, default=1080)
    parser.add_argument('--height', type=int, default=1920)
    parser.add_argument('--passes', default='guide,mask')
    parser.add_argument('--people', default='AB', help='which dancers are in the picture')
    parser.add_argument('--frames', help='first:last (1-based, inclusive); default all')
    parser.add_argument('--stills', help='comma-separated frames: render only these')
    parser.add_argument('--lens', type=float, default=40.0)
    parser.add_argument('--lighting', default='even', choices=sorted(LIGHTING), help='lights of the set (see LIGHTING)')
    parser.add_argument('--framing', default='wide', choices=sorted(FRAMING), help='how closely the cameras frame the pair (see FRAMING)')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--reuse-path', action='store_true',
                        help='keep the camera path already saved in --output (camera-path.json) instead of fitting it '
                             'to the bodies again: for re-rendering some frames after a body was corrected')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    inputs, out = Path(args.inputs), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((inputs / 'scene_inputs.json').read_text())
    faces = np.load(inputs / 'faces.npy')
    frames, fps = data['frames'], data['fps']
    caches = {person['label']: read_pc2(inputs / person['cache']) for person in data['people']}
    labels = [label for label in caches if label in args.people]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = int(round(fps))
    scene.frame_start, scene.frame_end = 1, frames
    scene.view_settings.view_transform = 'Standard'
    scene.world = bpy.data.worlds.new('Studio ambient')
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes['Background']
    background.inputs['Color'].default_value = (0.55, 0.57, 0.60, 1)
    background.inputs['Strength'].default_value = LIGHTING[args.lighting]['ambient']

    # where the dancers are, and which side the source camera filmed them from
    all_points = np.concatenate([caches[label] for label in caches], axis=1)  # every dancer, also if hidden
    pelvis = np.mean([np.asarray(person['pelvis']) for person in data['people']], axis=0)  # (frames, 3)
    middle = pelvis.mean(0)
    source = np.asarray(data['source_camera']['location'])
    front = source.mean(0) - middle
    front_angle = math.atan2(front[1], front[0])

    tan_x = 18.0 / args.lens  # 36 mm wide sensor
    tan_y = tan_x * args.height / args.width
    target = Vector((float(middle[0]), float(middle[1]), 0.95))
    sample = all_points[::2, ::7].reshape(-1, 3)

    # the room must contain every camera path, whichever one is rendered now (also the source camera,
    # which may have stood further back than the following cameras ever go)
    orbit_elevation = 8.0
    orbit_radius = max(fit_distance(sample, target, view_rotation(direction_at(front_angle, yaw, orbit_elevation)), tan_x, tan_y)
                       for yaw in range(0, 360, 10)) * 1.04
    source_reach = float(np.linalg.norm(source[:, :2] - middle[:2], axis=1).max())
    half = max(7.0, math.ceil(orbit_radius + 2.0), math.ceil(source_reach + 1.0))
    builder = SetBuilder(scene, half)
    set_hash = builder.build(args.lighting)
    builder.root.location = (float(middle[0]), float(middle[1]), 0.0)
    builder.root.rotation_euler = (0, 0, front_angle + math.pi / 2)  # the set's -y side faces the source camera

    grey = material('Grey mannequin', (0.50, 0.51, 0.53), 0.72)
    bodies = {}
    for label, cache in caches.items():
        mesh = bpy.data.meshes.new(f'Dancer {label}')
        mesh.from_pydata(cache[0].tolist(), [], faces.tolist())
        mesh.update()
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        body = bpy.data.objects.new(f'Dancer {label}', mesh)
        scene.collection.objects.link(body)
        mesh.materials.append(grey)
        modifier = body.modifiers.new('Reconstructed motion', 'MESH_CACHE')
        modifier.cache_format, modifier.filepath = 'PC2', str((inputs / f'{label}.pc2').resolve())
        modifier.frame_start, modifier.forward_axis, modifier.up_axis = 1, 'POS_Y', 'POS_Z'
        modifier.interpolation = 'NONE'
        body.hide_render = label not in labels
        bodies[label] = body

    camera = bpy.data.objects.new('Shared camera', bpy.data.cameras.new('Shared camera'))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.rotation_mode = 'QUATERNION'
    camera.data.sensor_fit, camera.data.sensor_width, camera.data.lens = 'HORIZONTAL', 36.0, args.lens
    camera.data.clip_start, camera.data.clip_end = 0.05, 100
    path, previous = [], None

    def key(frame, location, rotation, extra):
        nonlocal previous
        if previous is not None and previous.dot(rotation) < 0:
            rotation = -rotation
        previous = rotation.copy()
        camera.location, camera.rotation_quaternion = location, rotation
        camera.keyframe_insert('location', frame=frame)
        camera.keyframe_insert('rotation_quaternion', frame=frame)
        path.append(dict(frame=frame, location=[round(v, 4) for v in location], rotation_quaternion=[round(v, 6) for v in rotation], **extra))

    saved = json.loads((out / 'camera-path.json').read_text()) if args.reuse_path else None
    if saved is not None and args.camera != 'views':
        if len(saved) != frames:
            raise RuntimeError('The saved camera path has another length than the motion')
        if args.camera == 'original':
            camera.data.lens = data['source_camera']['focal_px'] * 36.0 / data['source_camera']['width']
        for row in saved:
            extra = {k: v for k, v in row.items() if k not in ('frame', 'location', 'rotation_quaternion')}
            key(row['frame'], Vector(row['location']), Quaternion(row['rotation_quaternion']), extra)
    elif args.camera == 'original':
        focal = data['source_camera']['focal_px']
        camera.data.lens = focal * 36.0 / data['source_camera']['width']
        for index in range(frames):
            rotation = Matrix(data['source_camera']['rotation'][index]).to_quaternion()
            key(index + 1, Vector(data['source_camera']['location'][index]), rotation, {})
    elif args.camera == 'views':
        pass  # four fixed viewpoints of the same moments, set while rendering
    else:
        # per frame: yaw, elevation, roll, dolly (1 = both dancers from head to toe)
        if args.camera == 'front':
            rows = np.tile([0.0, 6.0, 0.0, 1.0], (frames, 1))
        elif args.camera == 'orbit':
            rows = np.stack([np.linspace(0, 360, frames), np.full(frames, orbit_elevation), np.zeros(frames), np.ones(frames)], axis=1)
        else:
            knots = np.asarray(DYNAMIC_SHOTS, dtype=float)
            rows = []
            for share in np.linspace(0, 1, frames):
                i = min(len(knots) - 2, max(0, int(np.searchsorted(knots[:, 0], share, side='right')) - 1))
                a, b = knots[i], knots[i + 1]
                u = smootherstep(float(np.clip((share - a[0]) / (b[0] - a[0]), 0, 1)))
                rows.append((1 - u) * a[1:] + u * b[1:])
            rows = np.asarray(rows)

        def smooth(values, radius):
            padded = np.pad(values, ((radius, radius),) + ((0, 0),) * (values.ndim - 1), mode='edge')
            return np.stack([padded[i:i + 2 * radius + 1].mean(axis=0) for i in range(len(values))])

        # the camera follows the pair: it aims at their smoothed middle and keeps the distance at
        # which both fit, looking ahead so that it has already made room when a big move comes
        framing = FRAMING[args.framing]
        centre = smooth(smooth(pelvis, 14), 10)
        if framing['aim'] == 'extent':
            extent = 0.5 * (all_points[..., :2].min(axis=1) + all_points[..., :2].max(axis=1))
            centre[:, :2] = smooth(smooth(extent, 14), 10)
        low = smooth(all_points[..., 2].min(axis=1), 12)
        high = smooth(np.percentile(all_points[..., 2], 99.5, axis=1), 12)
        full_need, upper_need, rotations, directions, aims = [], [], [], [], []
        for index, (yaw, elevation, roll, dolly) in enumerate(rows):
            direction = direction_at(front_angle, yaw, elevation)
            rotation = view_rotation(direction, roll)
            middle_height = 0.5 * (float(low[index]) + float(high[index]))
            if 'fits_share' in framing:
                dolly = 1.0  # no moving in and out: the dynamic path keeps its turns, height changes and roll
            aim = Vector((float(centre[index, 0]), float(centre[index, 1]), middle_height + (dolly - 1) * 0.55))
            points = all_points[index, ::4]
            full_need.append(fit_distance(points, aim, rotation, tan_x, tan_y, margin=framing['full']))
            upper_need.append(fit_distance(points[points[:, 2] > 0.75], aim, rotation, tan_x, tan_y, margin=framing['upper']))
            rotations.append(rotation), directions.append(direction), aims.append(aim)

        def envelope(needs):
            needs = np.asarray(needs)
            padded = np.pad(needs, (18, 18), mode='edge')
            return smooth(np.asarray([padded[i:i + 37].max() for i in range(frames)]), 9)

        if 'fits_share' in framing:
            rows[:, 3] = 1.0
            distance = np.full(frames, max(1.6, float(np.percentile(full_need, framing['fits_share']))))
        else:
            full_fit, upper_fit = envelope(full_need), envelope(upper_need)
            wanted = full_fit / rows[:, 3]
            distance = 0.5 * (wanted + upper_fit + np.sqrt((wanted - upper_fit) ** 2 + 0.05 ** 2))  # smooth maximum
            distance = np.maximum(distance, 1.6)
        for index in range(frames):
            key(index + 1, aims[index] + directions[index] * float(distance[index]), rotations[index],
                dict(yaw=round(float(rows[index, 0]), 3), elevation=round(float(rows[index, 1]), 3),
                     roll=round(float(rows[index, 2]), 3), dolly=round(float(rows[index, 3]), 3),
                     distance=round(float(distance[index]), 3)))
    views = {}
    if args.camera == 'views':
        for name, yaw, elevation in (('front', 0, 8), ('side', 90, 8), ('rear', 180, 8), ('high', 35, 45)):
            direction = direction_at(front_angle, yaw, elevation)
            rotation = view_rotation(direction)
            distance = fit_distance(sample, target, rotation, tan_x, tan_y) * 1.03
            views[name] = (target + direction * distance, rotation)
            path.append(dict(view=name, yaw=yaw, elevation=elevation, location=[round(v, 4) for v in views[name][0]],
                             rotation_quaternion=[round(v, 6) for v in rotation]))
    else:
        for layer in camera.animation_data.action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        for point in curve.keyframe_points:
                            point.interpolation = 'LINEAR'

    # the camera must stay inside the room
    inside = half - 0.35
    to_set = Matrix.Rotation(-(front_angle + math.pi / 2), 4, 'Z') @ Matrix.Translation((-float(middle[0]), -float(middle[1]), 0))
    worst = max(max(abs((to_set @ Vector(row['location'])).x), abs((to_set @ Vector(row['location'])).y)) for row in path)
    if args.camera != 'original' and worst > inside:
        raise RuntimeError(f'The camera leaves the room ({worst:.2f} m from the centre, walls at {half} m)')

    manifest = dict(camera=args.camera, framing=args.framing, frames=frames, fps=fps, resolution=[args.width, args.height],
                    lens_mm=round(camera.data.lens, 3), sensor_width_mm=36.0, people=labels,
                    units='metres', up_axis='z', set=dict(version=SET_VERSION, sha256=set_hash, half_size_m=half,
                    height_m=ROOM_HEIGHT, centre=[round(float(v), 4) for v in middle[:2]], rotation_z=round(front_angle + math.pi / 2, 6),
                    objects=len(builder.objects)),
                    lighting=dict(name=args.lighting, **LIGHTING[args.lighting], note='4 ceiling disk lights + 1 main '
                                  'light from the front left + world ambient; EEVEE Next, Standard view'),
                    camera_max_distance_from_centre_m=round(worst, 3), renderer='EEVEE Next',
                    motion_caches={person['label']: person['cache_sha256'] for person in data['people']})
    (out / 'camera-path.json').write_text(json.dumps(path))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'scene.blend'))
    print('SCENE', json.dumps(manifest), flush=True)
    if args.prepare_only:
        return
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Rendering requires a managed GPU lease')

    if args.stills:
        todo = [int(v) for v in args.stills.split(',')]
    elif args.frames:
        first, last = (int(v) for v in args.frames.split(':'))
        todo = list(range(first, last + 1))
    else:
        todo = list(range(1, frames + 1))

    def render(pass_name, extension):
        (out / pass_name).mkdir(exist_ok=True)
        for name, (location, rotation) in (views or {'': (None, None)}).items():
            if location is not None:
                camera.location, camera.rotation_quaternion = location, rotation
            for frame in todo:
                scene.frame_set(frame)
                prefix = f'{name}_' if name else ''
                scene.render.filepath = str(out / pass_name / f'{prefix}frame_{frame:04d}.{extension}')
                bpy.ops.render.render(write_still=True)

    def show(visible, shadow_only=()):
        for label, body in bodies.items():
            body.hide_render = label not in visible and label not in shadow_only
            body.visible_camera = label in visible

    passes = args.passes.split(',')
    scene.eevee.taa_render_samples = 24
    scene.eevee.use_raytracing = True
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 95
    if 'guide' in passes:
        show(labels)
        render('guide', 'jpg')
    if 'plate' in passes:
        show((), shadow_only=labels)
        render('plate', 'jpg')
    if 'pair' in passes:
        show(labels)
        for label, body in bodies.items():
            body.data.materials.clear()
            body.data.materials.append(material(f'Mannequin {label}', PAIR_COLORS[label], 0.6))
        render('pair', 'jpg')
        for body in bodies.values():
            body.data.materials.clear()
            body.data.materials.append(grey)
    if 'solo' in passes:
        for label in labels:
            show((label,))
            render(f'solo_{label}', 'jpg')
    show(labels)
    if 'mask' in passes or 'solomask' in passes:
        # flat identity colours on white: no lights, no anti-aliasing, no dithering
        for obj in builder.objects:
            obj.hide_render = True
        for obj in scene.objects:
            if obj.type == 'LIGHT':
                obj.hide_render = True
        for label, body in bodies.items():
            body.data.materials.clear()
            body.data.materials.append(flat_material(f'Identity {label}', MASK_COLORS[label]))
        background.inputs['Color'].default_value = (1, 1, 1, 1)
        background.inputs['Strength'].default_value = 1.0
        scene.eevee.taa_render_samples = 1
        scene.eevee.use_raytracing = False
        scene.render.filter_size = 0.01
        scene.render.dither_intensity = 0.0
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.compression = 60
        if 'mask' in passes:
            render('mask', 'png')
        if 'solomask' in passes:
            for label in labels:
                show((label,))
                render(f'mask_{label}', 'png')
            show(labels)
    print('RENDERED', len(todo), 'frames:', passes, flush=True)


if __name__ == '__main__':
    main()
