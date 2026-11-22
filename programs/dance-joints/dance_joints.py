#!/usr/bin/env python3
"""A list of dance videos -> the 3D joint motion of the one dancer in each, as files that a motion
converter can read (the one-person program's results.npz, a BVH and an AMASS file).

One command from the list to the files; nothing is adjusted by hand in between and nothing in here
is specific to one video:

    dance_joints.py --list LIST.txt [--mirror auto|keep|flip] [--name NAME]
    dance_joints.py --job JOB_DIR        (go on with a run that was stopped or failed: the videos that
                                          are finished stay as they are, the others are made)

LIST.txt: one video per line, a title and then a YouTube link or a video file (the last word of the
line). Without a title the video's own title is used. Empty lines and lines starting with # are
skipped.

Per video, each step only once (a finished step is skipped when the run is continued):
  download   the video from YouTube (yt-dlp); a file is used as it is
  prepare    H.264 at its own frame rate up to 60, the long side at most 1920. A mirrored practice
             video (its title says "거울모드", "mirror mode", "mirrored", ...) is turned back
             left-right, so the joints are the dance as it is danced (--mirror)
  track      the dancer: the person clearly larger than anybody else, one identity all through (SAM 3)
  cuts       where the video cuts to another shot (find_cuts.py)
  flow       the background's motion in the picture, frame to frame
  geometry   field of view and floor of the video (MoGe)
  mocap      3D motion at the video's own frame rate (ExMR); the camera turn from the background only,
             never across a cut, never quicker than a camera filming a dancer turns and never a
             turn the background did not move with
  refine     quick motion that the motion capture smooths away, given back from the 2D keypoints
  world      "up" from the floor seen in the picture, gaps bridged; results.npz, BVH, AMASS
  check      the joints over the source and from the front and the side, for checking by eye
Downloads run ahead of the motion steps, cuts and flow (CPU) run beside the floor step (GPU), and
the last two steps of a video run beside the next video's motion steps. Every finished video goes into the results
folder (results_dir of the config) at once; a Korean summary follows at the end. A video that fails is reported and
the others go on.

Around the steps (programs/duo-scene/run_support.py): one run folder with every file the run writes (a dated folder
under runs_dir, or a managed job of the config's gpu_helper), one GPU lease for the whole run when the config names a
gpu_helper (renewed while it runs, always released). Files that show the source videos stay in the run folder and in
the results folder's 검증용_원본포함_공유금지 folder. Where the tools and models are: the config file (DANCE3D_CONFIG or
config.local.json, see config.example.json).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DUO = ROOT.parent / 'duo-scene'  # the motion steps are the two-dancer program's own
sys.path.insert(0, str(DUO))
import paths  # noqa: E402
from paths import ConfigError  # noqa: E402
from run_support import Failed, Lease, given_run, new_run, register, stamp  # noqa: E402

OWNER = 'dance-joints'
VIDEO_FORMAT = 'bv*[height<=1920][width<=1920]+ba/b[height<=1920][width<=1920]/bv*+ba/b'
# degrees per second: a quicker "camera turn" is an edit (a background sliding or jumping behind the
# dancer). Measured 11-14: real videos up to 160 (a quick hand-held swing), edit transitions 400-530
MAX_TURN_RATE = 180
# a title that says the video was turned left-right for learners to follow
MIRRORED = re.compile(r'거울\s*모드|미러\s*모드|좌우\s*반전|mirror\s*mode|mirrored|mirror\s*ver', re.IGNORECASE)
# what a step's own error text means for the user
HINTS = [('No frame shows one person clearly larger', '영상에서 한 사람이 다른 사람보다 또렷이 크게 나오는 장면을 찾지 못했습니다. '
          '춤추는 사람이 화면에 크게 나오는 영상을 써 주세요.'),
         ('Anchor frame shows', '영상에서 사람을 끝까지 따라가지 못했습니다.'),
         ('reliably seen for less than a second', '사람이 또렷이 보이는 장면이 1초도 되지 않습니다.'),
         ('Video unavailable', '유튜브에서 볼 수 없는 영상입니다(지워졌거나 비공개).'),
         ('Private video', '비공개 영상이라 받을 수 없습니다.'),
         ('Sign in to confirm', '유튜브가 로그인 확인을 요구해서 받지 못했습니다. 잠시 뒤 「멈춘 작업 이어서 하기」로 다시 해 보세요.'),
         ('members-only', '회원 전용 영상이라 받을 수 없습니다.'),
         ('is not a valid URL', '링크를 읽지 못했습니다. 줄 끝에 유튜브 링크가 있는지 확인해 주세요.'),
         ('영상이 너무 짧습니다', ''),
         ('out of memory', 'GPU 메모리가 모자랍니다. 다른 GPU 작업이 끝난 뒤 「멈춘 작업 이어서 하기」로 다시 해 보세요.')]


def yt_dlp():
    """The yt-dlp program: the config's yt_dlp, else the one on PATH."""
    configured = paths.location('yt_dlp', required=False)
    if configured is not None:
        return configured
    found = shutil.which('yt-dlp')
    if found is None:
        raise Failed('yt-dlp 이 없습니다. 설치: yt-dlp (설정 yt_dlp 또는 PATH) / yt-dlp is missing: install it (config key yt_dlp, '
                     'or on PATH).')
    return Path(found)


def is_link(text):
    return text.startswith(('http://', 'https://', 'www.', 'youtu.be/', 'youtube.com/', 'm.youtube.com/'))


def read_list(text):
    """The user's list -> [{number, title, link}] and the lines that could not be read."""
    videos, problems = [], []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        link, title = parts[-1], ' '.join(parts[:-1]).strip()
        if not (is_link(link) or Path(link).expanduser().is_file()):
            problems.append(f'{line_number}번째 줄: 끝에 유튜브 링크나 영상 파일이 없습니다 ({line[:60]})')
            continue
        if not is_link(link):
            link = str(Path(link).expanduser().resolve())
        elif not link.startswith('http'):
            link = 'https://' + link
        videos.append(dict(number=len(videos) + 1, title=title, link=link))
    return videos, problems


def safe_name(text, limit=60):
    """A folder name: no characters that a file system or a file manager would trip over."""
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', '_', text).strip(' ._')
    return text[:limit].rstrip(' ._') or 'video'


class Batch:
    def __init__(self, job, options):
        self.job, self.options = job, options
        self.videos = job / 'outputs/videos'
        self.results = job / 'outputs/results'
        for folder in (self.videos, self.results, job / 'logs', job / 'cache/tmp'):
            folder.mkdir(parents=True, exist_ok=True)
        self.env = {}
        self.children = set()  # running steps, so that a stop request can end them
        self.stopping = False
        self.started = time.time()
        self.lock = threading.Lock()
        self.state = {video['number']: dict(state='waiting', step='') for video in options['videos']}

    def stop_children(self):
        """A stop was asked for: end every running step (each is its own process group)."""
        self.stopping = True
        with self.lock:
            children = list(self.children)
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    # ---------- bookkeeping ----------
    def progress(self, state, title='', error=None):
        with self.lock:
            videos = [dict(number=video['number'], title=video['title'], **self.state[video['number']])
                      for video in self.options['videos']]
            text = dict(state=state, step=title, started=self.started, updated=time.time(), job=str(self.job), error=error,
                        videos=videos, done=sum(v['state'] == 'done' for v in videos),
                        failed=sum(v['state'] == 'failed' for v in videos), total=len(videos))
            (self.job / 'outputs/progress.json').write_text(json.dumps(text, ensure_ascii=False))
        print(f'[{stamp()}] {state}: {title}' + (f' - {error}' if error else ''), flush=True)

    def mark(self, video, state, step='', error=None):
        with self.lock:
            self.state[video['number']] = dict(state=state, step=step, error=error)
        self.progress('running', f'{video["number"]:02d} {video["title"]} - {step or state}')

    def stage(self, video, name, title, command, deps=(), gpu=False, expect=(), own_packages=False):
        """Run one step of one video unless it is already done with the same command and inputs.
        expect: files the step must have written (a tool can end without an error code and still have failed).
        own_packages: the step's interpreter uses only its own packages, not the motion tools' extra ones."""
        folder = self.videos / f'{video["number"]:02d}'
        stages, logs = folder / 'stages', folder / 'logs'
        stages.mkdir(parents=True, exist_ok=True)
        logs.mkdir(exist_ok=True)
        marker = stages / f'{name}.json'
        stamps = []
        for dep in deps:
            dep_marker = stages / f'{dep}.json'
            stamps.append(json.loads(dep_marker.read_text())['finished'] if dep_marker.exists() else None)
        # a step whose own program changed is done again (and with it every step after it)
        programs = [hashlib.sha256(Path(part).read_bytes()).hexdigest() for part in command
                    if isinstance(part, Path) and part.suffix == '.py' and part.is_file()]
        signature = hashlib.sha256(json.dumps([[str(part) for part in command], stamps, programs]).encode()).hexdigest()
        if marker.exists() and json.loads(marker.read_text()).get('signature') == signature and all(Path(f).exists() for f in expect):
            return
        self.mark(video, 'running', title)
        env = dict(self.env, PWD=str(DUO))
        if not gpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        if own_packages:
            env.pop('PYTHONPATH', None)
        marker.unlink(missing_ok=True)
        started = time.time()
        log = logs / f'{name}.log'
        if self.stopping:
            raise Failed('사용자가 중지했습니다.')
        with log.open('w') as handle:
            done = subprocess.Popen([str(part) for part in command], cwd=DUO, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            with self.lock:
                self.children.add(done)
            done.wait()
            with self.lock:
                self.children.discard(done)
        if self.stopping:
            raise Failed('사용자가 중지했습니다.')
        if done.returncode:
            text = log.read_text(errors='replace')
            tail = [line for line in text.splitlines() if line.strip() and 'Warning' not in line and 'WARNING' not in line][-4:]
            hint = next((hint for key, hint in HINTS if key in text and hint), '')
            raise Failed(f'{title} 단계가 실패했습니다. {hint}\n(기록: {log})\n' + '\n'.join(tail)[-600:])
        missing = [str(f) for f in expect if not Path(f).exists()]
        if missing:
            raise Failed(f'{title} 단계가 끝났지만 결과 파일이 없습니다: {missing[0]}\n(기록: {log})')
        marker.write_text(json.dumps(dict(signature=signature, finished=time.time(), seconds=round(time.time() - started, 1))))

    # ---------- one video ----------
    def fetch(self, video):
        """Download (or take the file) and prepare. Returns what the later steps need to know."""
        from paths import EXMR_PYTHON

        folder = self.videos / f'{video["number"]:02d}'
        given = folder / 'input'
        given.mkdir(parents=True, exist_ok=True)
        info = {}
        if is_link(video['link']):
            self.stage(video, 'download', '내려받기',
                       [yt_dlp(), '--no-playlist', '--no-progress', '--no-mtime', '--cache-dir', self.job / 'cache/yt-dlp',
                        '-f', VIDEO_FORMAT, '--merge-output-format', 'mp4', '--write-info-json',
                        '-o', given / 'download.%(ext)s', video['link']], own_packages=True)
            found = [p for p in given.glob('download.*') if p.suffix not in ('.json', '.part', '.ytdl')]
            if not found:
                raise Failed(f'내려받기가 끝났지만 영상 파일이 없습니다 ({given}).')
            path = found[0]
            info = json.loads((given / 'download.info.json').read_text())
            info = {key: info.get(key) for key in ('id', 'title', 'uploader', 'duration', 'fps', 'width', 'height', 'webpage_url',
                                                   'upload_date')}
        else:
            path = Path(video['link'])
        title = video['title'] or info.get('title') or path.stem
        mirror = self.options['mirror']
        said = MIRRORED.search(' '.join(filter(None, [video['title'], info.get('title')])))
        flip = mirror == 'flip' or (mirror == 'auto' and said is not None)
        self.stage(video, 'prepare', '영상 준비', [EXMR_PYTHON, '-B', ROOT / 'prepare_video.py', '--video', path,
                                                 '--output', folder / 'work/input'] + (['--flip'] if flip else []),
                   ['download'] if is_link(video['link']) else [], expect=[folder / 'work/input/source.mp4'])
        inputs = json.loads((folder / 'work/input/inputs.json').read_text())
        about = dict(number=video['number'], title=title, link=video['link'], youtube=info or None, flipped=flip,
                     flipped_because=(f'제목에 "{said.group(0)}"' if said and mirror == 'auto' else
                                      '「좌우 되돌리기: 항상」' if mirror == 'flip' else None),
                     fps=inputs['fps'], frames=inputs['frames'], seconds=inputs['duration'],
                     width=inputs['width'], height=inputs['height'])
        (folder / 'about.json').write_text(json.dumps(about, ensure_ascii=False, indent=1))
        self.mark(video, 'waiting', '영상 준비 끝')
        return about

    def capture(self, video, about):
        """The GPU steps: who the dancer is, the floor, the motion, the quick motion."""
        from paths import BODY_MODELS, DETECTOR, EXMR_PYTHON, MOGE, SAM3

        work = self.videos / f'{video["number"]:02d}/work'
        py, source = [EXMR_PYTHON, '-B'], work / 'input/source.mp4'
        self.stage(video, 'track', '사람 찾기', [*py, ROOT / 'track_one.py', '--video', source, '--output', work / 'tracks',
                                             '--sam3', SAM3, '--detector', DETECTOR], ['prepare'], gpu=True,
                   expect=[work / 'tracks/masks.npz'])
        beside = [self.cpu.submit(self.stage, video, 'cuts', '장면 바뀜 찾기',
                                  ['nice', '-n', '5', *py, ROOT / 'find_cuts.py', '--video', source, '--masks', work / 'tracks/masks.npz',
                                   '--output', work / 'cuts.json'], ['track'], expect=[work / 'cuts.json']),
                  self.cpu.submit(self.stage, video, 'flow', '배경 움직임 재기',
                                  ['nice', '-n', '5', *py, DUO / 'qa_fast_motion.py', 'video', '--video', source, '--masks',
                                   work / 'tracks/masks.npz', '--mask-fps', about['fps'], '--output', work / 'flow.json'], ['track'],
                                  expect=[work / 'flow.json'])]
        self.stage(video, 'geometry', '바닥 재기', [*py, DUO / 'scene_geometry.py', '--video', source, '--masks', work / 'tracks/masks.npz',
                                                '--model', MOGE, '--output', work / 'geometry', '--step', max(1, round(about['fps'] / 15))],
                   ['track'], gpu=True, expect=[work / 'geometry/geometry.json'])
        for future in beside:
            future.result()
        self.stage(video, 'mocap', '3D 동작 뽑기', [*py, DUO / 'mocap_people.py', '--video', source, '--tracks', work / 'tracks',
                                                '--geometry', work / 'geometry/geometry.json', '--checkpoints', BODY_MODELS,
                                                '--output', work / 'mocap', '--cuts', work / 'cuts.json', '--max-turn-rate', MAX_TURN_RATE,
                                                '--camera-flow', work / 'flow.json'],
                   ['track', 'cuts', 'flow', 'geometry'], gpu=True, expect=[work / 'mocap/people.json'])
        self.stage(video, 'refine', '빠른 움직임 되돌리기', [*py, DUO / 'refine_motion.py', '--mocap', work / 'mocap', '--tracks',
                                                      work / 'tracks', '--output', work / 'mocap_refined'], ['mocap'], gpu=True,
                   expect=[work / 'mocap_refined/people.json'])

    def finish(self, video, about):
        """The CPU steps beside the next video's GPU steps, then the files into the results folder."""
        from paths import EXMR_PYTHON

        work = self.videos / f'{video["number"]:02d}/work'
        py, source = ['nice', '-n', '10', EXMR_PYTHON, '-B'], work / 'input/source.mp4'
        self.stage(video, 'world', '바닥 기준으로 세우기', [*py, ROOT / 'world_one.py', '--mocap', work / 'mocap_refined', '--geometry',
                                                     work / 'geometry/geometry.json', '--camera-flow', work / 'flow.json',
                                                     '--cuts', work / 'cuts.json', '--camera-held', work / 'mocap/camera_held.json',
                                                     '--output', work / 'joints'] + (['--flipped'] if about['flipped'] else []),
                   ['refine', 'flow', 'geometry', 'cuts'], expect=[work / 'joints/results.npz', work / 'joints/report.json'])
        self.stage(video, 'check', '확인 영상 만들기', [*py, ROOT / 'check_video.py', '--video', source, '--joints', work / 'joints',
                                                   '--output', work / 'check'], ['world'],
                   expect=[work / 'check/joints_front_side.mp4', work / 'check/joints_over_source.mp4'])
        self.publish_video(video, about)
        self.mark(video, 'done', '끝')

    # ---------- results ----------
    def batch_folder(self):
        return paths.RESULTS / self.job.name

    def video_folder_name(self, about):
        return f'{about["number"]:02d}_{safe_name(about["title"])}'

    def publish_video(self, video, about):
        from joints_summary import video_notes

        work = self.videos / f'{video["number"]:02d}/work'
        report = json.loads((work / 'joints/report.json').read_text())
        target = self.batch_folder() / self.video_folder_name(about)
        check = self.batch_folder() / '검증용_원본포함_공유금지'
        target.mkdir(parents=True, exist_ok=True)
        check.mkdir(parents=True, exist_ok=True)
        for name in ('results.npz', 'motion.bvh', 'motion_amass.npz'):
            shutil.copy2(work / 'joints' / name, target / name)
        shutil.copy2(work / 'check/joints_front_side.mp4', target / '관절_정면과_옆.mp4')
        # anything that shows the source video stays in the check folder
        shutil.copy2(work / 'check/joints_over_source.mp4', check / f'{self.video_folder_name(about)}_원본_위_관절_원본포함.mp4')
        notes = video_notes(about, report)
        # run: where this result was made (the challenge program finds the source video's sound there)
        (target / '정보.json').write_text(json.dumps(dict(about, notes=notes, report=report, run=str(self.job)), ensure_ascii=False,
                                                     indent=1))
        (self.videos / f'{video["number"]:02d}' / 'published.json').write_text(json.dumps(dict(folder=str(target), finished=time.time())))

    def finished(self, video):
        """This video went through every step and into the results folder (in this run or an earlier one;
        runs before 2026-11-14 20:40 left no published.json, only the last step's marker and the files)."""
        folder = self.videos / f'{video["number"]:02d}'
        if (folder / 'published.json').exists():
            return True
        if not (folder / 'stages/check.json').exists() or not (folder / 'about.json').exists():
            return False
        about = json.loads((folder / 'about.json').read_text())
        return (self.batch_folder() / self.video_folder_name(about) / 'results.npz').exists()

    def write_summary(self, problems):
        from joints_summary import write_summary

        abouts, reports = {}, {}
        for video in self.options['videos']:
            folder = self.videos / f'{video["number"]:02d}'
            if (folder / 'about.json').exists():
                abouts[video['number']] = json.loads((folder / 'about.json').read_text())
            if (folder / 'work/joints/report.json').exists() and self.state[video['number']]['state'] == 'done':
                reports[video['number']] = json.loads((folder / 'work/joints/report.json').read_text())
        target = self.batch_folder()
        target.mkdir(parents=True, exist_ok=True)
        record = self.job / 'outputs/runs.json'  # minutes of every run of this job
        runs = json.loads(record.read_text()) if record.exists() else []
        runs.append(round((time.time() - self.started) / 60, 2))
        record.write_text(json.dumps(runs))
        write_summary(target, self.job, self.options, self.state, abouts, reports, problems, runs)
        shutil.copy2(ROOT / '사용법.txt', paths.RESULTS / '프로그램 사용법.txt')
        # one file to take to another computer: the joint files and the notes, nothing that shows a source video
        with zipfile.ZipFile(target / '관절_파일_모음.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
            for name in ('결과_안내.txt', '파일_설명.txt'):
                bundle.write(target / name, f'{target.name}/{name}')
            for folder in sorted(p for p in target.iterdir() if p.is_dir() and p.name != '검증용_원본포함_공유금지'):
                for file in sorted(folder.iterdir()):
                    bundle.write(file, f'{target.name}/{folder.name}/{file.name}')
        sums = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(target.rglob('*')) if p.is_file() and p.name != 'sha256.json'}
        (target / 'sha256.json').write_text(json.dumps(sums, ensure_ascii=False, indent=1))
        return target

    # ---------- the run ----------
    def run(self):
        videos = [video for video in self.options['videos'] if not self.finished(video)]
        for video in self.options['videos']:
            if self.finished(video):  # a continued run keeps what an earlier one finished; its notes are written anew
                about = json.loads((self.videos / f'{video["number"]:02d}' / 'about.json').read_text())
                self.publish_video(video, about)
                self.mark(video, 'done', '끝 (앞선 실행에서)')
        ahead, beside = ThreadPoolExecutor(max_workers=1), ThreadPoolExecutor(max_workers=1)
        self.cpu = ThreadPoolExecutor(max_workers=2)  # the CPU steps beside a video's floor step
        fetched = [ahead.submit(self.fetch, video) for video in videos]  # downloads run ahead, in order
        finishing = []
        for video, future in zip(videos, fetched):
            if self.stopping:
                break
            try:
                about = future.result()
                self.capture(video, about)
                finishing.append((video, beside.submit(self.finish, video, about)))
            except Failed as error:
                self.mark(video, 'failed', '멈춤', str(error))
            except Exception as error:  # an unexpected fault of one video must not end the others
                self.mark(video, 'failed', '멈춤', f'{type(error).__name__}: {error}')
        for video, future in finishing:
            try:
                future.result()
            except Failed as error:
                self.mark(video, 'failed', '멈춤', str(error))
            except Exception as error:
                self.mark(video, 'failed', '멈춤', f'{type(error).__name__}: {error}')
        ahead.shutdown(cancel_futures=True)
        beside.shutdown()
        self.cpu.shutdown()
        if self.stopping:
            raise Failed('사용자가 중지했습니다. 「멈춘 작업 이어서 하기」로 끝난 단계는 건너뛰고 이어 갈 수 있습니다.')
        self.progress('running', '결과 정리')
        return self.write_summary(self.options.get('problems', []))


def main():
    parser = argparse.ArgumentParser(description='춤 영상 목록 -> 영상마다 춤추는 한 사람의 3D 관절 움직임 파일')
    parser.add_argument('--list', help='한 줄에 영상 하나: 제목과 유튜브 링크(또는 영상 파일)')
    parser.add_argument('--mirror', default='auto', choices=['auto', 'keep', 'flip'],
                        help='좌우 되돌리기: auto = 제목에 거울모드가 있으면 되돌림(기본), keep = 화면 그대로, flip = 모두 되돌림')
    parser.add_argument('--name', default='dance-joints', help='작업 이름')
    parser.add_argument('--job', help='이어서 할 작업 폴더')
    args = parser.parse_args()

    try:
        if args.job:
            job = given_run(args.job)  # always the same spelling, or finished steps would not be recognised
            options = json.loads((job / 'outputs/options.json').read_text())
        else:
            if not args.list or not Path(args.list).is_file():
                parser.error('--list 로 목록 파일을 주세요')
            videos, problems = read_list(Path(args.list).read_text(encoding='utf-8'))
            if not videos:
                parser.error('목록에서 영상을 하나도 읽지 못했습니다: ' + ' / '.join(problems))
            options = dict(videos=videos, problems=problems, mirror=args.mirror, name=args.name,
                           list=Path(args.list).read_text(encoding='utf-8'))
        paths.require('python.motion', 'exmr', 'smplx', 'models.motion', 'models.sam3', 'models.moge', 'models.detector',
                      'runs_dir', 'results_dir')
        if not args.job:
            job = new_run(OWNER, args.name)
    except (ConfigError, Failed) as error:
        print(error, file=sys.stderr, flush=True)
        raise SystemExit(2)
    # the shared locations of the motion tools (duo-scene/paths.py) are worked out for this run
    os.environ['DUO_JOB'] = str(job)

    batch = Batch(job, options)
    (job / 'outputs/options.json').write_text(json.dumps(options, ensure_ascii=False, indent=1))
    print(f'[{stamp()}] job {job}: {len(options["videos"])} videos', flush=True)
    signal.signal(signal.SIGTERM, lambda signum, frame: batch.stop_children())
    signal.signal(signal.SIGINT, lambda signum, frame: batch.stop_children())
    lease = None
    try:
        lease = Lease(OWNER, job.name)
        batch.env = paths.cache_environment()
        batch.env.update(lease.environment())
        target = batch.run()
        failed = sum(state['state'] == 'failed' for state in batch.state.values())
        batch.progress('done', f'끝났습니다. 결과: {target}' + (f' (영상 {failed}개는 실패, 결과_안내.txt 참고)' if failed else ''))
    except (Failed, ConfigError) as error:
        batch.progress('failed', '멈췄습니다', str(error))
        raise SystemExit(1)
    except BaseException as error:
        batch.progress('failed', '멈췄습니다', f'{type(error).__name__}: {error}')
        raise
    finally:
        if lease:
            lease.release()
        register(job)


if __name__ == '__main__':
    main()
