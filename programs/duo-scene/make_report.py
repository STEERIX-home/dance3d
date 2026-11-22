"""Write the Korean result page (one self-contained HTML file) for the desktop result folder.

Numbers are read from the run records of the job; pictures are frames of the delivered videos and
of the 3D guide (no frame of the source video is embedded).
"""
import argparse
import base64
import json
import subprocess
from pathlib import Path

def strip(video, times, width=1500):
    """A row of frames of a video as a base64 JPEG."""
    inputs, filters = [], []
    for index, t in enumerate(times):
        inputs += ['-ss', f'{t:.2f}', '-i', str(video)]
        filters.append(f'[{index}:v]trim=end_frame=1,scale=-2:420[v{index}]')
    graph = ';'.join(filters) + ';' + ''.join(f'[v{i}]' for i in range(len(times))) + f'hstack={len(times)},scale={width}:-2'
    data = subprocess.run(['ffmpeg', '-v', 'error', *inputs, '-filter_complex', graph, '-frames:v', '1', '-q:v', '5',
                           '-f', 'image2pipe', '-c:v', 'mjpeg', '-'], capture_output=True, check=True).stdout
    return base64.b64encode(data).decode()


def image(path, width=1500):
    data = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', f'scale={width}:-2', '-frames:v', '1', '-q:v', '5',
                           '-f', 'image2pipe', '-c:v', 'mjpeg', '-'], capture_output=True, check=True).stdout
    return base64.b64encode(data).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, type=Path, help='result folder (videos already copied)')
    parser.add_argument('--deliveries', required=True, type=Path,
                        help='JSON list: file, title, description, method, runs (run.json paths), face (report path)')
    parser.add_argument('--text', required=True, type=Path, help='HTML fragment with the written sections')
    parser.add_argument('--picture', action='append', default=[],
                        help='NAME=file: embedded where the text says {{PICTURE_NAME}}')
    parser.add_argument('--views', type=Path, help='picture for {{VIEWS}} (default: 5_같은_순간_네_방향.jpg in the folder)')
    args = parser.parse_args()
    rows, figures = [], []
    for item in json.loads(args.deliveries.read_text()):
        video = args.folder / item['file']
        runs = [json.loads(Path(path).read_text()) for path in item['runs']]
        face = json.loads(Path(item['face']).read_text())
        if 'faces' in face:  # finish_pair.py: one entry per corrected character
            face['corrected_frames'] = sum(entry['corrected_frames'] for entry in face['faces'].values())
        seconds = sum(chunk['seconds'] for run in runs for chunk in run['runs'])
        rows.append(f'<tr><td><b>{item["file"]}</b></td><td>{item["title"]}<br><span class="dim">{item["description"]}</span></td>'
                    f'<td>{item["method"]}</td><td>{runs[0]["width"]}×{runs[0]["height"]}</td><td>{seconds / 60:.0f}분</td>'
                    f'<td>{face["corrected_frames"]} / {face["frames"]}장</td></tr>')
        duration = runs[0]['frames'] / runs[0]['fps']
        figures.append(f'<h3>{item["title"]}</h3><img src="data:image/jpeg;base64,'
                       f'{strip(video, [duration * share for share in (0.04, 0.2, 0.36, 0.52, 0.68, 0.84, 0.97)])}" alt="{item["title"]}">')
    table = ('<table><tr><th>파일</th><th>카메라</th><th>만든 방식</th><th>만든 크기</th><th>생성 시간</th>'
             '<th>아이 얼굴 보정</th></tr>' + ''.join(rows) + '</table>')
    views = args.views or args.folder / '5_같은_순간_네_방향.jpg'
    text = args.text.read_text()
    for entry in args.picture:
        name, path = entry.split('=', 1)
        text = text.replace('{{PICTURE_' + name + '}}', f'<img src="data:image/jpeg;base64,{image(path)}" alt="{name}">')
    text = text.replace('{{RESULT_TABLE}}', table).replace('{{RESULT_FIGURES}}', '\n'.join(figures))
    text = text.replace('{{VIEWS}}', f'<img src="data:image/jpeg;base64,{image(views)}" alt="같은 순간 네 방향">' if views.exists() else '')
    page = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>2인 캐릭터 3D 배경 결과</title>
<style>
:root {{ --bg:#f6f7f9; --card:#ffffff; --ink:#1b1f24; --dim:#5d6672; --line:#dfe3e8; --accent:#0b7a5a; --warn:#9a5b00; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171b; --card:#1d2126; --ink:#e7eaee; --dim:#9aa4b0; --line:#30363d; --accent:#3ecf9f; --warn:#e3a84a; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.7 "Noto Sans CJK KR","Noto Sans KR","Malgun Gothic",sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:24px 16px 80px; }}
h1 {{ font-size:28px; line-height:1.3; margin:8px 0 4px; }}
h2 {{ font-size:21px; margin:36px 0 10px; padding-top:14px; border-top:2px solid var(--line); }}
h3 {{ font-size:17px; margin:22px 0 6px; }}
p, li {{ max-width:860px; }}
.dim {{ color:var(--dim); font-size:14px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 18px; margin:12px 0; }}
.key {{ border-left:5px solid var(--accent); }}
.warn {{ border-left:5px solid var(--warn); }}
table {{ border-collapse:collapse; width:100%; margin:10px 0; font-size:15px; display:block; overflow-x:auto; }}
th, td {{ border:1px solid var(--line); padding:7px 10px; text-align:left; vertical-align:top; }}
th {{ background:var(--bg); white-space:nowrap; }}
img {{ max-width:100%; border-radius:8px; border:1px solid var(--line); display:block; margin:8px 0; }}
code {{ background:var(--bg); border:1px solid var(--line); border-radius:4px; padding:1px 5px; font-size:13px; word-break:break-all; }}
</style></head><body><main>
{text}
</main></body></html>'''
    (args.folder / '안내.html').write_text(page)
    print('report written:', args.folder / '안내.html', len(page) // 1024, 'KiB')


if __name__ == '__main__':
    main()
