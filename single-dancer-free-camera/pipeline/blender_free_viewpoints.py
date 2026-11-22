"""Render one unchanged mesh pose from four actual 3D camera positions."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def main(args):
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('A GPU lease is required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=args.scene)
    scene = bpy.context.scene
    body = bpy.data.objects['Motion guide']
    keys = body.data.shape_keys
    keys.animation_data_clear()
    keys.eval_time = keys.key_blocks[args.frame-1].frame
    bpy.context.view_layer.update()
    with np.load(args.mesh, allow_pickle=False) as data:
        expected = data['vertices'][args.frame-1].astype(np.float32)
    evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    buffer = np.empty(len(mesh.vertices)*3, dtype=np.float32)
    mesh.vertices.foreach_get('co', buffer)
    vertices = buffer.reshape(-1, 3)
    error = float(np.abs(vertices-expected).max())
    assert error < 1e-5, error
    evaluated.to_mesh_clear()
    camera = scene.camera
    camera.animation_data_clear()
    camera.data.animation_data_clear()
    camera.rotation_mode = 'QUATERNION'
    camera.data.lens = 50
    camera.data.sensor_fit = 'HORIZONTAL'
    scene.render.resolution_x, scene.render.resolution_y = 432, 756
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.cycles.device = 'GPU'
    scene.render.use_persistent_data = True
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'CUDA'
    preferences.get_devices()
    enabled = []
    for device in preferences.devices:
        device.use = device.type == 'CUDA'
        if device.use:
            enabled.append(device.name)
    assert len(enabled) == 1, enabled
    target = Vector(((vertices.min(axis=0)+vertices.max(axis=0))/2))
    tan_x = camera.data.sensor_width/(2*camera.data.lens)
    tan_y = tan_x*756/432
    views = [('Front',0,8), ('Side',90,8), ('Rear',180,8), ('High angle',35,55)]
    rows = []
    for index,(label,yaw,elevation) in enumerate(views):
        yr,er = math.radians(yaw),math.radians(elevation)
        direction = Vector((math.sin(yr)*math.cos(er),-math.cos(yr)*math.cos(er),math.sin(er)))
        rotation = (-direction).to_track_quat('-Z','Y')
        inverse = np.asarray(rotation.to_matrix(), dtype=float).T
        local = (vertices-np.asarray(target)) @ inverse.T
        distance = float(np.maximum(np.abs(local[:,0])/(tan_x*.85)+local[:,2],
                                    np.abs(local[:,1])/(tan_y*.85)+local[:,2]).max())+.12
        camera.location = target+direction*distance
        camera.rotation_quaternion = rotation
        local_camera = (vertices-np.asarray(camera.location)) @ inverse.T
        ndc = np.abs(local_camera[:,:2]/(-local_camera[:,2:3]))/np.array([tan_x,tan_y])
        assert float(ndc.max()) < .9
        scene.render.filepath = str(output/f'view_{index}.png')
        bpy.ops.render.render(write_still=True)
        rows.append(dict(label=label,yaw=yaw,elevation=elevation,distance=distance,
            location=list(camera.location),target=list(target),lens_mm=50,
            frame=args.frame,source_time=(args.frame-1)/24,mesh_error=error,
            max_extent=float(ndc.max()),image=scene.render.filepath))
    (output/'viewpoints.json').write_text(json.dumps(rows,indent=2))
    print('FOUR_VIEWPOINTS_COMPLETE',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',required=True)
    parser.add_argument('--mesh',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--frame',type=int,default=73)
    main(parser.parse_args(sys.argv[sys.argv.index('--')+1:]))
