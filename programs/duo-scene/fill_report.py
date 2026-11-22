"""Put the measured numbers into the written report text (placeholders @@NAME@@).

    fill_report.py --job JOB_DIR --text report_text.html --output report_text_filled.html

All numbers are read from the result files of the checks; nothing is typed in by hand.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from qa_fast_motion import quick_part

SHAKE = (2.0, 3.4)
NAMES = {'B/head': '갈색 머리 무용수(레제 역)의 머리', 'A/head': '검은 머리 무용수(아이 역)의 머리', 'B/chest': '갈색 머리 무용수의 가슴'}


def rows_of(path):
    data = json.loads(Path(path).read_text())
    rows = data['rows'] if isinstance(data, dict) else data
    return {(row['band'], tuple(row['window'])): row for row in rows}


def cell(rows, band, window=SHAKE):
    row = rows.get((band, window))
    if row is None or row['x']['correlation'] is None:
        return None
    return row['x']['ratio'], row['x']['correlation']


def timed(reference, test, band, window=SHAKE, cut=3.0):
    """Two videos' quick sideways motion of one band: size ratio, agreement as they are, best agreement
    when the test is shifted in time, and that shift in seconds (+: the test runs late)."""
    series = []
    for data in (reference, test):
        velocity = np.asarray(data['traces'][band], dtype=float) - np.nan_to_num(np.asarray(data['traces']['camera'], dtype=float))
        quick = quick_part(velocity, data['fps'], cut, noise=0.012)[:, 0]
        series.append((data['start'] + np.arange(len(quick)) / data['fps'], quick))
    grid = np.arange(window[0], window[1], 1 / 240)
    wanted = np.interp(grid, *series[0])
    found = {round(shift, 5): float(np.corrcoef(wanted, np.interp(grid + shift, *series[1]))[0, 1]) for shift in np.arange(-0.125, 0.12501, 1 / 240)}
    best = max(found, key=found.get)
    # test(t + shift) matches reference(t) best: a positive shift means the test shows things later
    return float(np.interp(grid, *series[1]).std() / wanted.std()), found[0.0], found[best], best, float(wanted.std())


def spread(values, digits=2):
    values = [v for v in values if v is not None]
    low, high = min(values), max(values)
    return f'{low:.{digits}f}' if round(low, digits) == round(high, digits) else f'{low:.{digits}f}~{high:.{digits}f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--text', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cameras', default='orbit,dynamic,original,front')
    args = parser.parse_args()
    diag, gen = args.job / 'outputs/work/diag', args.job / 'outputs/work/gen'
    cameras = [camera for camera in args.cameras.split(',')
               if (args.job / f'outputs/results/duo_{camera}.mp4').exists() and (gen / f'{camera}_B/matte.json').exists()]
    bodies = {name: rows_of(diag / f'same_{name}_3.0.json') for name in ('v1', 'v2')}
    source = json.loads((diag / 'traces_source.json').read_text())
    videos = {name: json.loads((diag / f'traces_{name}_original_result.json').read_text()) for name in ('v1', 'v2')}
    heads = ('B/head', 'A/head')
    values = {}
    values['SHAKE_BEFORE_RATIO'] = spread([cell(bodies['v1'], band)[0] for band in heads])
    values['SHAKE_RATIO'] = spread([cell(bodies['v2'], band)[0] for band in heads])
    values['SHAKE_BEFORE_AGREE'] = spread([cell(bodies['v1'], band)[1] for band in heads])
    values['SHAKE_AGREE'] = spread([cell(bodies['v2'], band)[1] for band in heads])
    lines = []
    for band, title in NAMES.items():
        before, after = cell(bodies['v1'], band), cell(bodies['v2'], band)
        if before is not None and after is not None:
            lines.append(f'<tr><td>{title} · 복원한 3D 몸</td><td>크기 {before[0]:.2f}배 · 일치도 {before[1]:.2f}</td>'
                         f'<td>크기 {after[0]:.2f}배 · 일치도 {after[1]:.2f}</td></tr>')

    def video_cell(found):
        ratio, plain, best, late, _ = found
        text = f'크기 {ratio:.2f}배 · 일치도 {plain:.2f}'
        if best - plain > 0.05:
            text += f' (결과가 {abs(late):.2f}초 {"늦은" if late > 0 else "앞선"} 것을 맞추면 {best:.2f})'
        return text

    for band, title in NAMES.items():
        if band in videos['v1']['traces'] and band in videos['v2']['traces']:
            lines.append(f'<tr><td>{title} · 완성 영상</td><td>{video_cell(timed(source, videos["v1"], band))}</td>'
                         f'<td>{video_cell(timed(source, videos["v2"], band))}</td></tr>')
    values['SHAKE_ROWS'] = '\n'.join(lines)
    # generated character against its own guide (head, shake window), all cameras and both dancers
    ratios, agreements, aligned = [], [], []
    for camera in cameras:
        for who in 'AB':
            path = gen / f'{camera}_{who}/motion_check.json'
            if not path.exists():
                continue
            check = json.loads(path.read_text())
            ratio, plain, best, _, size = timed(check['guide'], check['made'], f'{who}/head')
            if size >= 1.5:  # smaller shakes than this drown in the measuring noise
                ratios.append(ratio)
                agreements.append(plain)
                aligned.append(best)
    values['GEN_RATIO'], values['GEN_AGREE'], values['GEN_ALIGNED'] = spread(ratios), spread(agreements), spread(aligned)
    faces = [json.loads((gen / f'{camera}_A/faces.json').read_text()) for camera in cameras if (gen / f'{camera}_A/faces.json').exists()]
    values['FACE_MATCH'] = spread([face['match_as_generated']['median'] for face in faces]) + ' (카메라마다 다름. 얼굴이 정면으로 크게 보일수록 높음)'
    minutes = []
    for camera in cameras:
        seconds = sum(chunk['seconds'] for who in 'AB' for chunk in json.loads((gen / f'{camera}_{who}/run.json').read_text())['runs'])
        minutes.append(seconds / 60)
    values['MINUTES_PER_VIDEO'] = f'{np.median(minutes):.0f}'
    mattes = [json.loads((gen / f'{camera}_{who}/matte.json').read_text()) for camera in cameras for who in 'AB'
              if (gen / f'{camera}_{who}/matte.json').exists()]
    missing = sum(matte['frames_without_character'] for matte in mattes)
    total = sum(matte['frames'] for matte in mattes)
    values['MATTE_TEXT'] = (f'{total:,}장(카메라 {len(cameras)}개 × 2명) 중 캐릭터를 찾지 못한 장 {missing}장. '
                            f'3D 몸이 있던 자리를 캐릭터가 덮은 비율 가운데값 '
                            f'{spread([matte["guide_dancer_covered"]["median"] for matte in mattes])}')
    text = args.text.read_text()
    for name, value in values.items():
        text = text.replace(f'@@{name}@@', value)
    if '@@' in text:
        raise ValueError('Placeholders left: ' + ', '.join(sorted({part.split('@@')[0] for part in text.split('@@')[1::2]})))
    args.output.write_text(text)
    print(json.dumps({k: v for k, v in values.items() if k != 'SHAKE_ROWS'}, ensure_ascii=False, indent=1))
    print(values['SHAKE_ROWS'])


if __name__ == '__main__':
    main()
