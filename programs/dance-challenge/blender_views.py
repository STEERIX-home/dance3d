"""Run inside Blender: the dancer in the fixed 3D studio of the two-dancer program, filmed by the
cameras of the cut plan (front, left, right, back; full body or from the thighs up).

    blender -b --factory-startup --python-exit-code 1 --python blender_views.py -- --inputs DIR --plan plan.json --output DIR

The studio, its lights and the mannequin colours are those of blender_duo_scene.py (imported from
there, so both programs show the same room). Every camera keeps one distance for the whole video
(the one at which the dancer fits in 97 of 100 frames) and follows the dancer's place smoothly; its
direction is fixed about the dancer, counted from the front (the side the source camera filmed from).
One folder per camera, as pair_run.py and finish_pair.py read it:
  <view>/pair/frame_####.jpg    set + the dancer as a shaded blue mannequin (what H3 follows)
  <view>/guide/frame_####.jpg   set + the dancer as a grey mannequin (what SCAIL-2 follows, with the mask)
  <view>/plate/frame_####.jpg   the set alone with the dancer's shadow
  <view>/mask/frame_####.png    the dancer in flat blue on white
  <view>/manifest.json, <view>/camera-path.json (one row per frame of the video; with --facts each row
                                also says where the picture cuts the body: the whole body above the lower
                                edge, both knees below it, the top of the head inside, the nearest body point)
Only the frames the plan needs are rendered.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.dont_write_bytecode = True  # Blender's own Python ignores PYTHONDONTWRITEBYTECODE: no __pycache__ in the programs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'duo-scene'))  # the two-dancer program's studio
from blender_duo_scene import (LIGHTING, MASK_COLORS, PAIR_COLORS, ROOM_HEIGHT, SET_VERSION, SetBuilder,  # noqa: E402
                               direction_at, fit_distance, flat_material, material, read_pc2, view_rotation)

# elevation of the camera (degrees), how tightly it frames, in how many of 100 frames the part that must
# fit is inside the picture, and how closely it follows the dancer (frames of the two averaging passes
# across, frames of the one up and down). 'full': the whole body. 'close': as a phone held close films a
# dancer, from the thighs up: the picture spans from 20 cm under the pelvis (upper thigh) to 8 cm over the
# top of the head, and across the head (the mesh within 14 cm of the head joint) must fit; arms and hands
# may leave the picture. In a portrait picture the height decides how close. The lower edge then stays on
# the thighs in every frame (the knees out of the picture) and the room left over goes above the head.
FRAMINGS = {'full': dict(elevation=6.0, margin=0.94, share=97, follow=(14, 10, 12)),
            'close': dict(elevation=3.0, margin=0.94, share=92, follow=(8, 6, 8), below_pelvis=0.20, headroom=0.08, head=0.14)}
LENS = 40.0
NEAREST = 0.15  # no part of the body ever comes nearer the camera than this (metres), in the frames it films


def smooth(values, radius):
    padded = np.pad(values, ((radius, radius),) + ((0, 0),) * (values.ndim - 1), mode='edge')
    return np.stack([padded[i:i + 2 * radius + 1].mean(axis=0) for i in range(len(values))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=1080)
    parser.add_argument('--height', type=int, default=1920)
    parser.add_argument('--lighting', default='shadow', choices=sorted(LIGHTING))
    parser.add_argument('--passes', default='pair,plate,mask')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--facts', help='motion_facts.json of joints_to_world.py (mesh points of the knees and the top of the head)')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    inputs, out = Path(args.inputs), Path(args.output)
    plan = json.loads(Path(args.plan).read_text())
    data = json.loads((inputs / 'scene_inputs.json').read_text())
    faces = np.load(inputs / 'faces.npy')
    frames, fps = data['frames'], data['fps']
    person = data['people'][0]
    label = person['label']
    cache = read_pc2(inputs / person['cache'])

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

    pelvis = np.asarray(person['pelvis'])
    middle = pelvis.mean(0)
    front_angle = float(plan['front_angle_rad'])
    tan_x = 18.0 / LENS
    tan_y = tan_x * args.height / args.width

    # the camera of each view, per frame: it aims at the dancer's smoothed place and keeps one distance
    head = np.asarray(person['head'])
    axis = 0.5 * (pelvis[:, :2] + head[:, :2])  # the body's middle, seen from above
    body_points = json.loads(Path(args.facts).read_text()).get('body_points') if args.facts else None
    if body_points:
        top = cache[:, body_points['head_top']].astype(np.float64)
        knees = np.stack([cache[:, body_points['left_knee']].mean(1), cache[:, body_points['right_knee']].mean(1)], 1).astype(np.float64)
    else:
        top, knees = head + np.array([0.0, 0.0, 0.12]), None  # the head joint sits about 12 cm under the top of the head
    everything = [cache[index, ::4].astype(np.float64) for index in range(frames)]
    cameras, records, facts = {}, {}, {}
    for view, spec in plan['views'].items():
        framing = FRAMINGS[spec['framing']]
        direction = direction_at(front_angle, spec['yaw_deg'], framing['elevation'])
        rotation = view_rotation(direction)
        across, again, upright = framing['follow']
        if spec['framing'] == 'full':
            shown = everything
            low = smooth(cache[..., 2].min(axis=1), upright)
            high = smooth(np.percentile(cache[..., 2], 99.5, axis=1), upright)
        else:  # mid-thigh to over the head must fit up and down, the head across
            low = smooth(pelvis[:, 2] - framing['below_pelvis'], upright)
            high = smooth(top[:, 2] + framing['headroom'], upright)
            shown = []
            for index in range(frames):
                points = cache[index, ::2].astype(np.float64)
                near_head = points[np.linalg.norm(points - head[index], axis=1) <= framing['head']]
                ends = np.array([[axis[index, 0], axis[index, 1], low[index]], [axis[index, 0], axis[index, 1], high[index]]])
                shown.append(np.concatenate([near_head, ends]))
        centre = smooth(smooth(np.asarray([0.5 * (p[:, :2].min(0) + p[:, :2].max(0)) for p in shown]), across), again)
        aims, needs, reach = [], [], []
        for index in range(frames):
            aim = Vector((float(centre[index, 0]), float(centre[index, 1]), 0.5 * (float(low[index]) + float(high[index]))))
            aims.append(aim)
            needs.append(fit_distance(shown[index], aim, rotation, tan_x, tan_y, margin=framing['margin']))
            reach.append(float(((everything[index] - np.asarray(aim)) @ np.asarray(direction)).max()))  # towards the camera
        # one distance for the frames this camera films: the dancer fits in `share` of them, and nothing comes nearer than NEAREST
        used = [frame - 1 for frame in plan['render'][view]]
        fit = float(np.percentile(np.asarray(needs)[used], framing['share']))
        toward = float(np.asarray(reach)[used].max())
        distance = max(0.5, fit, toward + NEAREST)
        if spec['framing'] == 'close':  # the lower edge stays on the thighs; the room left over goes above the head
            tilt, edge = math.radians(framing['elevation']), math.atan(tan_y * framing['margin'])
            drop = distance * (math.cos(tilt) * math.tan(tilt + edge) - math.sin(tilt))  # aim -> lower edge, at the dancer
            aims = [Vector((aim.x, aim.y, float(low[index]) + drop)) for index, aim in enumerate(aims)]
        locations = [aim + direction * distance for aim in aims]
        cameras[view] = (locations, rotation)

        # where the picture cuts the body, frame by frame
        turn = np.asarray(rotation.to_matrix(), dtype=float)  # columns: the camera's right, up and backward axes

        def picture(points, index):
            local = (np.atleast_2d(points) - np.asarray(locations[index])) @ turn
            depth = -local[:, 2]
            return local[:, 0] / depth / tan_x, local[:, 1] / depth / tan_y, depth

        rows = []
        for index in range(frames):
            _, y, depth = picture(everything[index], index)
            head_x, head_y, _ = picture(top[index], index)
            row = dict(feet_in=bool(y.min() >= -1.0), head_in=bool(abs(head_x[0]) <= 1.0 and head_y[0] <= 1.0),
                       nearest_m=round(float(depth.min()), 3))
            if knees is not None:
                row['knees_out'] = bool(picture(knees[index], index)[1].max() < -1.0)
            rows.append(row)
        facts[view] = rows
        records[view] = dict(direction=spec['direction'], yaw_deg=spec['yaw_deg'], framing=spec['framing'],
                             elevation_deg=framing['elevation'], distance_m=round(distance, 3), fit_distance_m=round(fit, 3),
                             body_towards_camera_m=round(toward, 3), fits_share=framing['share'], frames_rendered=len(used),
                             **{f'share_{key}': round(float(np.mean([rows[i][key] for i in used])), 3)
                                for key in rows[0] if key != 'nearest_m'},
                             nearest_m=round(min(rows[i]['nearest_m'] for i in used), 3))

    reach = max(float(np.linalg.norm(np.asarray(location)[:2] - middle[:2])) for locations, _ in cameras.values() for location in locations)
    half = max(7.0, math.ceil(reach + 2.0))
    builder = SetBuilder(scene, half)
    set_hash = builder.build(args.lighting)
    builder.root.location = (float(middle[0]), float(middle[1]), 0.0)
    builder.root.rotation_euler = (0, 0, front_angle + math.pi / 2)  # the set's -y side faces the front camera

    grey = material('Grey mannequin', (0.50, 0.51, 0.53), 0.72)
    mesh = bpy.data.meshes.new(f'Dancer {label}')
    mesh.from_pydata(cache[0].tolist(), [], faces.tolist())
    mesh.update()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    body = bpy.data.objects.new(f'Dancer {label}', mesh)
    scene.collection.objects.link(body)
    mesh.materials.append(grey)
    modifier = body.modifiers.new('Reconstructed motion', 'MESH_CACHE')
    modifier.cache_format, modifier.filepath = 'PC2', str((inputs / person['cache']).resolve())
    modifier.frame_start, modifier.forward_axis, modifier.up_axis = 1, 'POS_Y', 'POS_Z'
    modifier.interpolation = 'NONE'

    camera = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.rotation_mode = 'QUATERNION'
    camera.data.sensor_fit, camera.data.sensor_width, camera.data.lens = 'HORIZONTAL', 36.0, LENS
    camera.data.clip_start, camera.data.clip_end = 0.05, 100

    for view, (locations, rotation) in cameras.items():
        folder = out / view
        folder.mkdir(parents=True, exist_ok=True)
        path = [dict(frame=index + 1, location=[round(v, 4) for v in locations[index]],
                     rotation_quaternion=[round(v, 6) for v in rotation], view=view, **facts[view][index]) for index in range(frames)]
        (folder / 'camera-path.json').write_text(json.dumps(path))
        manifest = dict(camera=view, framing=records[view]['framing'], frames=frames, fps=fps, resolution=[args.width, args.height],
                        lens_mm=LENS, sensor_width_mm=36.0, people=[label], units='metres', up_axis='z', view=records[view],
                        set=dict(version=SET_VERSION, sha256=set_hash, half_size_m=half, height_m=ROOM_HEIGHT,
                                 centre=[round(float(v), 4) for v in middle[:2]], rotation_z=round(front_angle + math.pi / 2, 6),
                                 objects=len(builder.objects)),
                        lighting=dict(name=args.lighting, **LIGHTING[args.lighting]), renderer='EEVEE Next',
                        motion_caches={label: person['cache_sha256']})
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print('VIEWS', json.dumps(records), flush=True)
    if args.prepare_only:
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':  # the pipeline gave this step no GPU
        raise RuntimeError('Rendering needs the GPU of the run (CUDA_VISIBLE_DEVICES is empty)')

    def render(pass_name, extension):
        for view, (locations, rotation) in cameras.items():
            (out / view / pass_name).mkdir(exist_ok=True)
            camera.rotation_quaternion = rotation
            for frame in plan['render'][view]:
                scene.frame_set(frame)
                camera.location = locations[frame - 1]
                scene.render.filepath = str(out / view / pass_name / f'frame_{frame:04d}.{extension}')
                bpy.ops.render.render(write_still=True)

    passes = args.passes.split(',')
    scene.eevee.taa_render_samples = 24
    scene.eevee.use_raytracing = True
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 95
    if 'plate' in passes:  # the room with the dancer's shadow only
        body.hide_render, body.visible_camera = False, False
        render('plate', 'jpg')
        body.visible_camera = True
    if 'pair' in passes:
        body.data.materials.clear()
        body.data.materials.append(material(f'Mannequin {label}', PAIR_COLORS[label], 0.6))
        render('pair', 'jpg')
    if 'guide' in passes:
        body.data.materials.clear()
        body.data.materials.append(grey)
        render('guide', 'jpg')
    if 'mask' in passes:
        # flat identity colour on white: no lights, no anti-aliasing, no dithering
        for obj in builder.objects:
            obj.hide_render = True
        for obj in scene.objects:
            if obj.type == 'LIGHT':
                obj.hide_render = True
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
        render('mask', 'png')
    print('RENDERED', {view: len(plan['render'][view]) for view in cameras}, passes, flush=True)


if __name__ == '__main__':
    main()
