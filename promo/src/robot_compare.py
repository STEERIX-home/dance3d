"""Does the robot's measured head motion follow the dancer's joints? Compare signals over lags (experiment).

    robot_compare.py RESULTS.npz EYES.json MUSIC_START_S
"""
import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

data = np.load(sys.argv[1], allow_pickle=True)
eyes = json.load(open(sys.argv[2]))
start = float(sys.argv[3])
names = [str(n) for n in data['joint_names']]
J = data['joints_global'].astype(np.float64)
fps = float(data['fps'])
t_h = np.arange(len(J)) / fps
j = lambda name: J[:, names.index(name)]


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


up = np.array([0.0, 1.0, 0.0])
ears = j('left_ear') - j('right_ear')
head_roll = np.degrees(np.arcsin(unit(ears)[:, 1]))                       # + = the dancer's left ear higher
shoulders = j('left_shoulder') - j('right_shoulder')
hips = j('left_hip') - j('right_hip')
yaw = lambda v: np.degrees(np.unwrap(np.arctan2(v[:, 2], v[:, 0])))
head_yaw = yaw(ears) - yaw(shoulders)                                       # head turned against the shoulders
torso_yaw = yaw(shoulders)
torso_yaw -= np.median(torso_yaw)
face = j('nose') - (j('left_ear') + j('right_ear')) / 2
head_pitch = np.degrees(np.arcsin(unit(face)[:, 1]))                        # + = looking up
pelvis = j('pelvis')[:, 1]
neck_side = unit(j('head') - j('neck'))
lean = np.degrees(np.arcsin(np.einsum('ij,ij->i', unit(j('neck') - j('pelvis')), unit(shoulders))))  # body leaning sideways
arm = lambda side: np.degrees(np.arcsin(unit(j(f'{side}_wrist') - j(f'{side}_shoulder'))[:, 1]))
human = dict(head_roll=head_roll, head_yaw=head_yaw - np.median(head_yaw), head_pitch=head_pitch - np.median(head_pitch), torso_yaw=torso_yaw,
             pelvis_height=(pelvis - np.median(pelvis)) * 100, body_lean=lean - np.median(lean), left_arm=arm('left'), right_arm=arm('right'))

rows = [r for r in eyes['rows'] if r['found']]
t_r = np.array([r['time'] for r in rows])
rest = [r for r in rows if 9.0 < r['time'] < 12.5]
robot = dict(tilt=np.array([r['tilt_deg'] for r in rows]) - np.mean([r['tilt_deg'] for r in rest]),
             x=np.array([r['centre'][0] for r in rows]) - np.mean([r['centre'][0] for r in rest]),
             y=-(np.array([r['centre'][1] for r in rows]) - np.mean([r['centre'][1] for r in rest])),
             gap=np.array([r['gap'] for r in rows]) - np.mean([r['gap'] for r in rest]),
             size=np.array([np.sqrt(sum(r['area']) / 2) for r in rows]) - np.mean([np.sqrt(sum(r['area']) / 2) for r in rest]))
grid = np.arange(0.0, min(t_h[-1], t_r[-1] - start), 1 / 30)
print(f'dance {t_h[-1]:.2f} s at {fps:.2f}/s; robot samples {len(rows)}; compared over {grid[-1]:.1f} s')
for h_name, h in human.items():
    h_s = gaussian_filter1d(h, fps * 0.05)
    line = []
    for r_name, r in robot.items():
        best = (0, 0)
        for lag in np.arange(-1.5, 1.5001, 1 / 30):
            # robot time for dance time g: start + g + lag
            r_i = np.interp(start + grid + lag, t_r, r)
            # leave out stretches where the eyes were not found for more than 0.25 s
            near = np.abs(t_r[np.clip(np.searchsorted(t_r, start + grid + lag), 0, len(t_r) - 1)] - (start + grid + lag)) < 0.25
            a, b = np.interp(grid, t_h, h_s)[near], r_i[near]
            c = np.corrcoef(a - a.mean(), b - b.mean())[0, 1]
            if abs(c) > abs(best[0]):
                best = (c, lag)
        line.append(f'{r_name} {best[0]:+.2f}@{best[1]:+.2f}s')
    print(f'{h_name:14s} sd {np.std(h):6.1f} | ' + ' | '.join(line))
print('robot sd:', {k: round(float(np.std(v)), 1) for k, v in robot.items()})
