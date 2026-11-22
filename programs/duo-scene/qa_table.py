"""Print one or more results of qa_motion_fidelity.py side by side (ratio / agreement, sideways motion)."""
import json
import sys

names = sys.argv[1::2]
tables = [{(row['band'], tuple(row['window'])): row for row in json.load(open(path))['rows']} for path in sys.argv[2::2]]
keys = sorted(set().union(*tables))
print(f"{'band':9s}{'window':>11s}{'source px':>10s}  " + '  '.join(f'{name:>20s}' for name in names))
for key in keys:
    cells, source = [], None
    for table in tables:
        row = table.get(key)
        if row is None:
            cells.append(f"{'-':>20s}")
            continue
        source = row['x']['reference_rms_px']
        agreement = row['x']['correlation']
        cells.append(f"{row['x']['ratio']:9.2f} /{'   - ' if agreement is None else format(agreement, '5.2f')}    ")
    print(f'{key[0]:9s}{key[1][0]:5.1f}-{key[1][1]:4.1f}s{source:9.1f}   ' + '  '.join(cells))
