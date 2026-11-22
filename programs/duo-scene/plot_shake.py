"""Picture for the report: the quick sideways motion of one body band over time, in the source
video and in one or more reconstructions (files written by qa_fast_motion.py / qa_motion_fidelity.py).

    plot_shake.py --source traces_source.json --test NAME=file.json ... --band B/head --output shake.jpg
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from qa_fast_motion import quick_part



def series(data, band, cut, minus_camera):
    velocity = np.asarray(data['traces'][band], dtype=float)
    if minus_camera and 'camera' in data['traces']:
        velocity = velocity - np.nan_to_num(np.asarray(data['traces']['camera'], dtype=float))
    quick = quick_part(velocity, data['fps'], cut, noise=0.012)
    return data['start'] + np.arange(len(quick)) / data['fps'], quick[:, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--test', action='append', required=True, help='LABEL=file (file has a "test" entry or is a traces file)')
    parser.add_argument('--band', action='append', required=True, help='BAND=title, e.g. "B/head=갈색 머리 무용수의 머리"')
    parser.add_argument('--start', type=float, default=1.2)
    parser.add_argument('--stop', type=float, default=3.6)
    parser.add_argument('--cut', type=float, default=3.0)
    parser.add_argument('--raw', action='store_true', help='same camera on all sides: do not take the background motion away')
    parser.add_argument('--title', default='')
    parser.add_argument('--ylabel', default='좌우 위치 (픽셀)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from paths import ConfigError, font as font_file
    try:
        font = font_manager.FontProperties(fname=str(font_file('Regular')))
    except ConfigError:  # no Korean font: matplotlib's own (the Korean labels may show as boxes)
        font = None
    source = json.loads(Path(args.source).read_text())
    tests = []
    for entry in args.test:
        label, path = entry.split('=', 1)
        data = json.loads(Path(path).read_text())
        tests.append((label, data.get('test', data)))
    colours = ['#c0392b', '#1f77b4', '#2e8b57', '#8e44ad']
    figure, axes = plt.subplots(len(args.band), 1, figsize=(12, 3.4 * len(args.band)), sharex=True, squeeze=False)
    for axis, entry in zip(axes[:, 0], args.band):
        band, title = entry.split('=', 1)
        time, value = series(source, band, args.cut, not args.raw)
        keep = (time >= args.start) & (time <= args.stop)
        axis.plot(time[keep], value[keep], color='black', lw=2.4, label='원본 영상')
        for (label, data), colour in zip(tests, colours):
            if band not in data['traces']:
                continue
            time, value = series(data, band, args.cut, not args.raw)  # a video's own background motion is taken away
            keep = (time >= args.start) & (time <= args.stop)
            axis.plot(time[keep], value[keep], color=colour, lw=1.8, label=label)
        axis.set_title(title, fontproperties=font, fontsize=14, loc='left')
        axis.set_ylabel(args.ylabel, fontproperties=font, fontsize=12)
        axis.grid(alpha=0.3)
        axis.legend(prop=font, loc='upper right', ncol=4, frameon=False)
    axes[-1, 0].set_xlabel('시간 (초)', fontproperties=font, fontsize=12)
    if args.title:
        figure.suptitle(args.title, fontproperties=font, fontsize=15)
    figure.tight_layout()
    figure.savefig(args.output, dpi=110, pil_kwargs=dict(quality=90))


if __name__ == '__main__':
    main()
