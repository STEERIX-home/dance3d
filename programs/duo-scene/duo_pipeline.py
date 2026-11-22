#!/usr/bin/env python3
"""Two dancers of a video -> two characters dancing in one fixed 3D room, filmed by the source
video's own camera work and, if asked for, by free cameras.

One command from the user's files to the finished videos; nothing is adjusted by hand in between
and nothing in here is specific to one video or one character:

    duo_pipeline.py --video DANCE.mp4 --character LEFT.png --character RIGHT.png
                    [--swap] [--height 1=130] [--outfit 2="a red hoodie and jeans"]
                    [--cameras original,orbit,dynamic,front] [--base official|singularity]
                    [--start S] [--seconds S] [--name NAME]
    duo_pipeline.py --job JOB_DIR        (go on with a run that was stopped or failed)

The first picture goes to the dancer who stands on the left when the two are most clearly apart
(the tracker's anchor frame), the second to the one on the right; --swap exchanges them.

What it does, each step only once (a finished step is skipped when the run is continued):
  inputs      the files are checked and normalised (prepare_inputs.py)
  track       both dancers get a fixed identity through the video (SAM 3)
  geometry    field of view and floor of the source video (MoGe)
  mocap       3D motion of each dancer at the video's own frame rate (ExMR), quick motion restored
  world       both dancers in one room, bodies kept apart and on the floor; "up" is the floor seen
              in the picture, so a rolling hand-held camera stays a rolling camera and the dancers
              stand upright (align_people.py)
  references  each character picture prepared (cut out unless its surroundings are plain)
  scene       the fixed studio with both dancers as coloured mannequins, one render per camera.
              "original" is the source video's camera, frame by frame (position, direction, roll,
              field of view), so it shows what the source shows and nothing the source did not
              film; the free cameras keep one distance per video
  describe    one sentence about each character, written from its picture by a small
              picture-reading model; with it the instruction text for the video model is built
  generate    H3 paints both characters and the room together in one run per camera; a chunk in
              which the room is lost, a head sits on the wrong body, a mannequin is left or the
              characters dance behind the guide is made again by itself
  finish      full size, faces held to their pictures where a picture shows one, the music
  checks      motion against the 3D guide and the source, who is who, chunk seams
  publish     videos and a result page into the results folder (results_dir of the config)

Around the steps (run_support.py): one run folder with every file the run writes (a dated folder under runs_dir,
or a managed job of the config's gpu_helper), one GPU lease for the whole run when the config names a gpu_helper
(renewed while it runs, always released). Where the tools and models are: the config file (paths.py).
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import paths  # noqa: E402
from paths import ConfigError  # noqa: E402
from run_support import Failed, Lease, given_run, new_run, register, stamp  # noqa: E402

OWNER = 'duo-dance'
COMFY_URL = f'http://127.0.0.1:{os.environ.get("DANCE3D_COMFY_PORT", "8189")}'
# in the order they are made and listed: the source's own camera work first
CAMERAS = {'original': ('1_원본과_같은_카메라', '원본과 같은 카메라'), 'orbit': ('2_360도_회전', '360도 회전'),
           'dynamic': ('3_다이내믹_카메라', '다이내믹 카메라'), 'front': ('4_정면', '정면')}
LABELS = 'AB'
# what a step's own error text means for the user
HINTS = [('No frame shows exactly', '영상에서 두 사람이 서로 떨어져 함께 보이는 장면을 찾지 못했습니다. 두 사람의 몸이 같이 보이는 영상을 써 주세요.'),
         ('Anchor frame shows', '영상에서 두 사람을 끝까지 따라가지 못했습니다. 두 사람이 계속 화면에 있는 구간만 골라 다시 해 보세요.'),
         ('No person found in', '캐릭터 그림에서 사람이나 캐릭터를 찾지 못했습니다. 캐릭터 한 명이 또렷이 나온 그림을 써 주세요.'),
         ('out of memory', 'GPU 메모리가 모자랍니다. 다른 GPU 작업이 끝난 뒤 「멈춘 작업 이어서 하기」로 다시 해 보세요.')]


class Pipeline:
    def __init__(self, job, options, dry=False):
        self.job, self.options, self.dry = job, options, dry
        self.work, self.results, self.logs = job / 'outputs/work', job / 'outputs/results', job / 'logs'
        self.input = job / 'outputs/input'
        if not dry:
            for folder in (self.work / 'stages', self.results, self.logs, self.input, job / 'cache/tmp'):
                folder.mkdir(parents=True, exist_ok=True)
        self.env = {}
        self.comfy = None
        self.children = set()  # running steps, so that a stop request can end them
        self.stopping = False
        self.started = time.time()
        self.lock = threading.Lock()

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
        if self.dry:
            print(f'[{state}] {title}', flush=True)
            return
        with self.lock:
            done = len(list((self.work / 'stages').glob('*.json')))
            cameras = self.options['cameras']
            total = 9 + len(LABELS) + 6 * len(cameras) + (2 if 'original' in cameras else 0) + 1
            text = dict(state=state, step=title, steps_done=done, steps_total=total, started=self.started, updated=time.time(),
                        job=str(self.job), error=error)
            (self.job / 'outputs/progress.json').write_text(json.dumps(text, ensure_ascii=False))
        print(f'[{stamp()}] {state}: {title}' + (f' - {error}' if error else ''), flush=True)

    def stage(self, name, title, command, deps=(), gpu=False, quiet=False, expect=(), own_packages=False, also='', before=None):
        """Run one step unless it is already done with the same command and the same inputs.
        expect: files the step must have written (a tool can end without an error code and still have failed).
        own_packages: the step's interpreter uses only its own packages, not the motion tools' extra ones.
        also: text the step depends on besides its command (the content of a file it reads).
        before: called first if the step is going to run (what it needs running, e.g. the generation server)."""
        marker = self.work / 'stages' / f'{name}.json'
        stamps = []
        for dep in deps:
            dep_marker = self.work / 'stages' / f'{dep}.json'
            stamps.append(json.loads(dep_marker.read_text())['finished'] if dep_marker.exists() else None)
        signature = hashlib.sha256(json.dumps([[str(part) for part in command], stamps] + ([also] if also else [])).encode()).hexdigest()
        if marker.exists() and json.loads(marker.read_text()).get('signature') == signature and all(Path(f).exists() for f in expect):
            return False
        if before:
            before()
        if not quiet:
            self.progress('running', title)
        if self.dry:
            print('   ', ' '.join(str(part) for part in command)[:400], flush=True)
            return True
        env = dict(self.env, PWD=str(ROOT))
        if not gpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        if own_packages:
            env.pop('PYTHONPATH', None)
            env['HF_HUB_OFFLINE'] = env['TRANSFORMERS_OFFLINE'] = '1'
        marker.unlink(missing_ok=True)
        started = time.time()
        log = self.logs / f'{name}.log'
        if self.stopping:
            raise Failed('사용자가 중지했습니다.')
        with log.open('w') as handle:
            done = subprocess.Popen([str(part) for part in command], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
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
            tail = [line for line in text.splitlines() if line.strip() and 'Warning' not in line][-6:]
            hint = next((hint for key, hint in HINTS if key in text and hint), '')
            raise Failed(f'{title} 단계가 실패했습니다. {hint}\n(기록: {log})\n' + '\n'.join(tail)[-900:])
        missing = [str(f) for f in expect if not Path(f).exists()]
        if missing:
            raise Failed(f'{title} 단계가 끝났지만 결과 파일이 없습니다: {missing[0]}\n(기록: {log})')
        marker.write_text(json.dumps(dict(signature=signature, finished=time.time(), seconds=round(time.time() - started, 1))))
        return True

    # ---------- the private ComfyUI ----------
    def comfy_up(self):
        try:
            urllib.request.urlopen(COMFY_URL + '/system_stats', timeout=3)
            return True
        except (urllib.error.URLError, OSError):
            return False

    def start_comfy(self):
        if self.dry or self.comfy:
            return
        if self.comfy_up():
            raise Failed('생성 서버 자리(127.0.0.1:8189)를 다른 실행이 쓰고 있습니다. 그 실행이 끝난 뒤 다시 시작해 주세요.')
        self.progress('running', '생성 서버 시작')
        log = (self.logs / 'comfy_server.log').open('a')
        self.comfy = subprocess.Popen([sys.executable, '-B', str(ROOT / 'comfy_serve.py')], cwd=ROOT, env=self.env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(240):
            if self.comfy.poll() is not None:
                raise Failed(f'생성 서버가 시작하지 못했습니다 (기록: {self.logs / "comfy_server.log"}).')
            if self.comfy_up():
                return
            time.sleep(1)
        raise Failed('생성 서버가 4분 안에 준비되지 않았습니다.')

    def stop_comfy(self):
        if self.comfy and self.comfy.poll() is None:
            os.killpg(self.comfy.pid, signal.SIGTERM)
            try:
                self.comfy.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(self.comfy.pid, signal.SIGKILL)
                self.comfy.wait()
        self.comfy = None

    # ---------- the run ----------
    def run(self):
        from paths import (BASE_MODELS, BLENDER, BODY_MODELS, COMFY_PYTHON, DESCRIBER, DETECTOR, EXMR_PYTHON, FACE_MODELS,
                           FACE_PYTHON, MOGE, SAM3)
        o, work, py = self.options, self.work, [EXMR_PYTHON, '-B']
        cameras = o['cameras']
        video, inputs_file = self.input / 'source.mp4', self.input / 'inputs.json'

        command = [*py, 'prepare_inputs.py', '--video', o['video'], '--output', self.input, '--start', o['start']]
        for picture in (reversed(o['characters']) if o['swap'] else o['characters']):
            command += ['--character', picture]
        if o['seconds']:
            command += ['--seconds', o['seconds']]
        self.stage('inputs', '입력 파일 검사', command)
        if self.dry and not inputs_file.exists():
            source = dict(fps=60.0, duration=20.0, has_audio=True, portrait=True)
        else:
            source = json.loads(inputs_file.read_text())['video']
        fps = source['fps']
        size = (1080, 1920) if source['portrait'] else (1920, 1080)
        made = (768, 1344) if source['portrait'] else (1344, 768)

        self.stage('track', '두 사람 추적', [*py, 'track_people.py', '--video', video, '--output', work / 'tracks', '--sam3', SAM3,
                                        '--detector', DETECTOR, '--people', len(LABELS)], ['inputs'], gpu=True)
        self.stage('geometry', '원본 카메라와 바닥 재기', [*py, 'scene_geometry.py', '--video', video, '--masks', work / 'tracks/masks.npz',
                                                 '--model', MOGE, '--output', work / 'geometry', '--step', max(1, round(fps / 15))],
                   ['track'], gpu=True)
        self.stage('mocap', '3D 동작 뽑기', [*py, 'mocap_people.py', '--video', video, '--tracks', work / 'tracks', '--geometry',
                                         work / 'geometry/geometry.json', '--checkpoints', BODY_MODELS, '--output', work / 'mocap',
                                         '--keep-frames'], ['track', 'geometry'], gpu=True, expect=[work / 'mocap/people.json'])
        self.stage('source_motion', '원본의 빠른 움직임 재기', ['nice', '-n', '5', *py, 'qa_fast_motion.py', 'video', '--video', video, '--masks',
                                                      work / 'tracks/masks.npz', '--mask-fps', fps, '--output',
                                                      work / 'diag/traces_source.json'], ['track'])
        self.stage('refine', '빠른 움직임 되돌리기', [*py, 'refine_motion.py', '--mocap', work / 'mocap', '--tracks', work / 'tracks',
                                              '--output', work / 'mocap_refined'], ['mocap'], gpu=True)
        self.stage('align', '두 사람을 한 공간에 놓기', [*py, 'align_people.py', '--mocap', work / 'mocap_refined', '--output', work / 'world',
                                                '--camera-seconds', 0.1, '--camera-flow', work / 'diag/traces_source.json',
                                                '--geometry', work / 'geometry/geometry.json'],
                   ['refine', 'source_motion', 'geometry'])
        # quick motion is exported a quarter larger: the video model gives back about 0.82 of it (export_scene_inputs.py)
        command = [*py, 'export_scene_inputs.py', '--world', work / 'world', '--output', work / 'scene_inputs', '--fps', 24,
                   '--mocap', work / 'mocap_refined', '--quick-gain', 1.25]
        for label, centimetres in sorted(o['heights'].items()):
            command += ['--height', f'{label}={centimetres:g}']
        self.stage('export', '3D 몸 내보내기', command, ['align'])
        for label in LABELS:
            self.stage(f'reference_{label}', f'캐릭터 그림 준비 ({label})',
                       [*py, 'make_solo_reference.py', '--image', self.input / f'character_{label}.png', '--sam3', SAM3,
                        '--output', work / f'ref_{label}', '--background', 'auto'], ['inputs'], gpu=True)
        for camera in cameras:
            self.stage(f'scene_{camera}', f'3D 화면 그리기 ({CAMERAS[camera][1]})',
                       # Blender looks for a script by $PWD, not by the working directory, and ends with code 0 when the
                       # script fails unless told otherwise
                       [BLENDER, '-b', '--factory-startup', '--python-exit-code', 1, '--python', ROOT / 'blender_duo_scene.py', '--',
                        '--inputs', work / 'scene_inputs',
                        '--output', work / 'scene' / camera, '--camera', camera, '--lighting', 'shadow', '--framing', 'steady',
                        '--width', size[0], '--height', size[1], '--passes', 'pair,plate,mask,solomask'], ['export'], gpu=True,
                       expect=[work / 'scene' / camera / 'manifest.json', work / 'scene' / camera / 'pair/frame_0001.jpg',
                               work / 'scene' / camera / 'mask/frame_0001.png'])

        # what each character looks like, written from its picture by a small picture-reading model
        descriptions = {}
        if DESCRIBER.is_dir():
            command = [COMFY_PYTHON, '-B', 'describe_characters.py', '--model', DESCRIBER, '--output', work / 'describe']
            for label in LABELS:  # the pictures as supplied: a cut-out on plain grey reads as "a 3D render"
                command += ['--picture', f'{label}={self.input / f"character_{label}.png"}']
            self.stage('describe', '캐릭터 생김새 적기', command, ['inputs'], gpu=True, own_packages=True,
                       expect=[work / f'describe/description_{label}.txt' for label in LABELS])
            if not self.dry:
                descriptions = {LABELS.index(label) + 1: (work / f'describe/description_{label}.txt').read_text().strip()
                                for label in LABELS}

        # the instruction text: the same form for every video and character (pair_prompt.py)
        prompt = work / 'prompts/pair.txt'
        from pair_prompt import build_prompt
        text = build_prompt(len(LABELS), True, {LABELS.index(label) + 1: cm for label, cm in o['heights'].items()}, descriptions,
                            {LABELS.index(label) + 1: outfit for label, outfit in o.get('outfits', {}).items()})
        if not self.dry and (not prompt.exists() or prompt.read_text() != text):
            prompt.parent.mkdir(parents=True, exist_ok=True)
            prompt.write_text(text)

        base = [] if o['base'] == 'official' else ['--unet', BASE_MODELS[o['base']]]
        pictures = []
        for kind in ('full', 'closeup'):
            for label in LABELS:
                pictures += ['--picture', work / f'ref_{label}/picture_{kind}.png']
        finishing = ThreadPoolExecutor(max_workers=1)  # CPU work of one camera runs beside the next camera's generation
        waiting = []

        def finish(camera):
            run = work / 'gen' / camera
            command = ['nice', '-n', '10', FACE_PYTHON, '-B', 'finish_pair.py', '--run', run, '--inputs', work / 'scene_inputs',
                       '--models', FACE_MODELS, '--output', self.results / f'duo_{camera}.mp4',
                       '--compare', self.results / f'duo_{camera}_guide_and_result.mp4']
            for label in LABELS:
                command += ['--face', f'{label}={self.input / f"character_{label}.png"}']
            if source['has_audio']:
                command += ['--audio', video]
            self.stage(f'finish_{camera}', f'마무리 ({CAMERAS[camera][1]})', command, [f'generate_{camera}'], quiet=True,
                       expect=[self.results / f'duo_{camera}.mp4'])
            checks = ['nice', '-n', '10', *py]
            self.stage(f'check_motion_{camera}', '확인: 동작', [*checks, 'qa_pair_motion.py', '--run', run, '--output', run / 'motion_check.json'],
                       [f'generate_{camera}'], quiet=True)
            self.stage(f'check_identity_{camera}', '확인: 누가 누구인지',
                       [*checks, 'qa_pair_identity.py', '--run', run, '--output', run / 'identity_check.json'],
                       [f'generate_{camera}'], quiet=True)
            self.stage(f'check_seams_{camera}', '확인: 이음매', [*checks, 'qa_pair_seams.py', '--run', run, '--output', run / 'seams.json'],
                       [f'generate_{camera}'], quiet=True)
            if not self.dry:  # the finished video goes to the results folder at once; the rest follows at the end
                target = self.target()
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.results / f'duo_{camera}.mp4', target / (CAMERAS[camera][0] + '.mp4'))
            if camera == 'original':
                traces = work / 'diag/traces_result_original.json'
                self.stage('check_source', '확인: 원본과 비교', [*checks, 'qa_fast_motion.py', 'video', '--video',
                                                              self.results / 'duo_original.mp4', '--scene', work / 'scene/original',
                                                              '--scene-fps', 24, '--output', traces], ['finish_original'], quiet=True)
                self.stage('check_source_compare', '확인: 원본과 비교',
                           [*checks, 'qa_fast_motion.py', 'compare', '--reference', work / 'diag/traces_source.json', '--test', traces,
                            '--cut', 3.0, '--output', work / 'diag/original_vs_source.json'],
                           ['check_source'], quiet=True)

        # the checks every chunk must pass belong to this step: when they change, a job that is continued goes through
        # its chunks again (those already made are read back and only judged; the ones that now fail are made again)
        checks = '\nchecks ' + hashlib.sha256((ROOT / 'pair_run.py').read_bytes() + (ROOT / 'qa_pair_identity.py').read_bytes()).hexdigest()[:16]
        try:
            for camera in cameras:  # the server is started when a camera's step really runs (also when it is redone in a continued job)
                self.stage(f'generate_{camera}', f'캐릭터 입히기 ({CAMERAS[camera][1]})',
                           [*py, 'pair_run.py', '--scene', work / 'scene' / camera, *pictures, '--prompt', prompt, '--output',
                            work / 'gen' / camera, '--tag', camera, '--width', made[0], '--height', made[1], *base],
                           [f'scene_{camera}', *(f'reference_{label}' for label in LABELS)], gpu=True, also=text + checks,
                           expect=[work / 'gen' / camera / 'run.json'], before=self.start_comfy)
                if not self.dry:
                    waiting.append(finishing.submit(finish, camera))
                else:
                    finish(camera)
        finally:
            self.stop_comfy()
        self.progress('running', '마무리와 확인')
        for future in waiting:
            future.result()
        finishing.shutdown()
        self.stage('summary', '결과 정리', [*py, 'make_summary.py', '--job', self.job],
                   [f'finish_{camera}' for camera in cameras] + [f'check_identity_{camera}' for camera in cameras],
                   expect=[self.results / '결과_요약.html'])
        return self.publish()

    def target(self):
        return paths.RESULTS / self.job.name

    def publish(self):
        target = self.target()
        if self.dry:
            return target
        self.progress('running', '결과 폴더에 저장')
        check = target / '검증용_원본포함_공유금지'
        check.mkdir(parents=True, exist_ok=True)
        sums = {}

        def copy(source, name, folder=target):
            if Path(source).exists():
                shutil.copy2(source, folder / name)
                sums[str((folder / name).relative_to(target))] = hashlib.sha256((folder / name).read_bytes()).hexdigest()

        for camera in self.options['cameras']:
            copy(self.results / f'duo_{camera}.mp4', CAMERAS[camera][0] + '.mp4')
        first = CAMERAS[self.options['cameras'][0]][1].replace(' ', '_')
        copy(self.results / 'guide_and_result_small.mp4', f'3D화면과_결과_{first}.mp4')
        copy(self.results / '결과_요약.html', '결과_요약.html')
        copy(self.results / 'summary.json', '결과_요약.json')
        # anything that shows the source video stays in the check folder
        copy(self.results / 'source_and_result.mp4', '원본과_결과_나란히_원본포함.mp4', check)
        copy(self.work / 'tracks/preview.mp4', '사람_배정_확인_원본포함.mp4', check)
        (target / 'sha256.json').write_text(json.dumps(sums, ensure_ascii=False, indent=1))
        return target


def main():
    parser = argparse.ArgumentParser(description='두 사람 춤 영상 + 캐릭터 그림 두 장 -> 고정 3D 방에서 춤추는 두 캐릭터 영상')
    parser.add_argument('--video', help='두 사람이 춤추는 영상')
    parser.add_argument('--character', action='append', default=[], help='캐릭터 그림. 첫째는 왼쪽 사람, 둘째는 오른쪽 사람에게 입힌다')
    parser.add_argument('--swap', action='store_true', help='두 그림의 배정을 서로 바꾼다')
    parser.add_argument('--height', action='append', default=[], help='번호=cm: 그 캐릭터의 키 (예: 1=130). 없으면 원본 사람의 크기')
    parser.add_argument('--outfit', action='append', default=[],
                        help='번호=옷: 그 캐릭터에게 그림의 옷 대신 입힐 옷 (예: 2="a red hoodie and blue jeans"). 없으면 그림의 옷')
    parser.add_argument('--cameras', default='original',
                        help='만들 카메라: ' + ', '.join(CAMERAS) + ' (기본: original = 원본 영상의 카메라 그대로)')
    parser.add_argument('--base', default='official', choices=['official', 'singularity'], help='H3 본체')
    parser.add_argument('--start', type=float, default=0.0, help='영상에서 쓸 구간의 시작(초)')
    parser.add_argument('--seconds', type=float, help='쓸 길이(초). 없으면 끝까지')
    parser.add_argument('--name', default='duo-dance', help='작업 이름')
    parser.add_argument('--job', help='이어서 할 작업 폴더')
    parser.add_argument('--dry-run', action='store_true', help='단계와 명령만 보여 주고 실행하지 않는다')
    args = parser.parse_args()

    keys = ('python.motion', 'python.comfy', 'python.face', 'blender', 'comfyui', 'exmr', 'smplx', 'models.comfy', 'models.h3',
            'models.sam3', 'models.moge', 'models.detector', 'models.motion', 'models.face', 'runs_dir', 'results_dir')
    if args.job:
        try:
            job = given_run(args.job)  # always the same spelling, or finished steps would not be recognised
            paths.require(*keys)
        except (ConfigError, Failed) as error:
            print(error, file=sys.stderr, flush=True)
            raise SystemExit(2)
        options = json.loads((job / 'outputs/options.json').read_text())
    else:
        if not args.video or len(args.character) != len(LABELS):
            parser.error(f'--video 하나와 --character {len(LABELS)}개가 필요합니다')
        asked = [camera.strip() for camera in args.cameras.split(',') if camera.strip()]
        unknown = [camera for camera in asked if camera not in CAMERAS]
        if unknown or not asked:
            parser.error(f'모르는 카메라: {unknown}')
        cameras = [camera for camera in CAMERAS if camera in asked]
        for path in (args.video, *args.character):
            if not Path(path).is_file():
                parser.error(f'파일이 없습니다: {path}')
        heights = {}
        for entry in args.height:
            number, centimetres = entry.split('=', 1)
            if number not in ('1', '2') or not 50 <= float(centimetres) <= 250:
                parser.error('--height 는 1=130 처럼 번호(1 또는 2)와 50~250 사이의 cm')
            # picture number -> dancer label, after --swap
            heights[LABELS[(int(number) - 1) ^ int(args.swap)]] = float(centimetres)
        outfits = {}
        for entry in args.outfit:
            number, outfit = entry.split('=', 1)
            if number not in ('1', '2') or not outfit.strip():
                parser.error('--outfit 는 2="a red hoodie and blue jeans" 처럼 번호(1 또는 2)와 옷')
            outfits[LABELS[(int(number) - 1) ^ int(args.swap)]] = outfit.strip()
        options = dict(video=str(Path(args.video).resolve()), characters=[str(Path(p).resolve()) for p in args.character],
                       swap=args.swap, heights=heights, outfits=outfits, cameras=cameras, base=args.base, start=args.start,
                       seconds=args.seconds, name=args.name)
        try:
            paths.require(*keys)
            job = paths.RUNS / '00000000-000000-dry-run' if args.dry_run else new_run(OWNER, args.name)
        except (ConfigError, Failed) as error:
            print(error, file=sys.stderr, flush=True)
            raise SystemExit(2)
    # the shared locations (paths.py) are worked out for this run
    os.environ['DUO_JOB'] = str(job)

    if args.dry_run:
        Pipeline(job, options, dry=True).run()
        return

    pipeline = Pipeline(job, options)
    (job / 'outputs/options.json').write_text(json.dumps(options, ensure_ascii=False, indent=1))
    print(f'[{stamp()}] job {job}', flush=True)
    signal.signal(signal.SIGTERM, lambda signum, frame: pipeline.stop_children())
    signal.signal(signal.SIGINT, lambda signum, frame: pipeline.stop_children())
    lease = None
    try:
        lease = Lease(OWNER, job.name)
        pipeline.env = paths.cache_environment()
        pipeline.env.update(lease.environment())
        target = pipeline.run()
        pipeline.progress('done', f'끝났습니다. 결과: {target}')
    except (Failed, ConfigError) as error:
        pipeline.progress('failed', '멈췄습니다', str(error))
        raise SystemExit(1)
    except BaseException as error:
        pipeline.progress('failed', '멈췄습니다', f'{type(error).__name__}: {error}')
        raise
    finally:
        pipeline.stop_comfy()
        if lease:
            lease.release()
        register(job)


if __name__ == '__main__':
    main()
