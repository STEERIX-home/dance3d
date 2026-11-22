"""Check: does the reconstructed front/back order of the two dancers agree with what the video shows?

Where one dancer covers the other in the picture, the tracked masks show who is in front: body
joints of the covered dancer fall inside the other dancer's mask. This is compared with which of
the two is nearer to the camera in the reconstruction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--mocap', required=True)
parser.add_argument('--tracks', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
mocap = Path(args.mocap)
info = json.loads((mocap / 'people.json').read_text())
with np.load(Path(args.tracks) / 'masks.npz') as data:
    shape = tuple(data['shape'])
    masks = {k: np.unpackbits(data[k], axis=-1)[..., :shape[-1]].astype(bool) for k in data.files if k != 'shape'}
scale_x, scale_y = shape[2] / info['width'], shape[1] / info['height']
people = {}
for entry in info['people']:
    result = torch.load(mocap / f'person_{entry["label"]}/results.pt', map_location='cpu', weights_only=False)
    people[entry['label']] = dict(points=result['joints_2d'][:, :22].numpy(), depth=result['joints_incam'][:, :22, 2].numpy(),
                                  reliable=np.array(entry['reliable']))
A, B = people['A'], people['B']
rows = []
for t in range(info['frames']):
    if not (A['reliable'][t] and B['reliable'][t]):
        continue
    hidden, depth = {}, {}
    for own, other, name in ((A, 'B', 'A'), (B, 'A', 'B')):
        x = np.clip((own['points'][t, :, 0] * scale_x).astype(int), 0, shape[2] - 1)
        y = np.clip((own['points'][t, :, 1] * scale_y).astype(int), 0, shape[1] - 1)
        inside = masks[other][t, y, x] & ~masks[name][t, y, x]
        hidden[name] = int(inside.sum())
        depth[name] = float(own['depth'][t].mean())
    if abs(hidden['A'] - hidden['B']) < 3:
        continue  # no clear covering in this frame
    seen_front = 'B' if hidden['A'] > hidden['B'] else 'A'
    built_front = 'A' if depth['A'] < depth['B'] else 'B'
    rows.append(dict(frame=t, seen_front=seen_front, built_front=built_front, depth_gap=abs(depth['A'] - depth['B'])))
agree = [r for r in rows if r['seen_front'] == r['built_front']]
summary = dict(frames_with_clear_covering=len(rows), agreeing=len(agree),
               share=len(agree) / max(1, len(rows)),
               disagreeing_frames=[r['frame'] for r in rows if r['seen_front'] != r['built_front']][:60],
               median_depth_gap_when_disagreeing=float(np.median([r['depth_gap'] for r in rows if r['seen_front'] != r['built_front']] or [0])))
Path(args.output).write_text(json.dumps(summary, indent=1))
print(json.dumps(summary))
