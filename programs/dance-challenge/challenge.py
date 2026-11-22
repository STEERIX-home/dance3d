#!/usr/bin/env python3
"""One dancer's joints (dance-joints result) + one character picture -> a dance challenge video:
the character dances in the 3D studio, filmed from the front, the left, the right and the back,
cut on the beat of the music, with effects, title and camera names.

One command from the user's files to the finished video; nothing is adjusted by hand in between
and nothing in here is specific to one dance or one character:

    challenge.py --joints RESULT_FOLDER --character PICTURE [--music FILE] [--title TEXT]
                 [--backend scail|h3] [--vram auto|8|12|16|24]
                 [--effects normal|strong|none] [--no-labels] [--seconds S] [--name NAME]
    challenge.py --job RUN_FOLDER     (go on with a run that was stopped or failed)

RESULT_FOLDER is one video's folder of the dance-joints program (<results_dir>/<run>/NN_title, or
<run>/outputs/videos/NN). The music is the sound of that dance video unless --music gives another file.
Where the installed tools and models are comes from the config file (DANCE3D_CONFIG or config.local.json,
see config.example.json and programs/duo-scene/paths.py).

Video model (--backend):
  scail   SCAIL-2 (zai-org, MIT licence), the default. Its model file, picture size and memory settings
          follow the graphics card's memory (--vram, default vram_gb of the config: "auto" = the card's own):
          the table in programs/duo-scene/scail_tiers.py. A size smaller than the card makes the run keep
          the rest of the card free, so it behaves as on a card of that size.
  h3      MiniMax H3 (the earlier backend; its licence excludes Korea, the US, the EU and the UK).

What it does, each step only once (a finished step is skipped when the run is continued):
  inputs      the files are checked and copied into the run (prepare_challenge.py)
  world       the joints are handed over as the dancer of the 3D studio (joints_to_world.py,
              export_scene_inputs.py of the two-dancer program)
  beats       tempo, beats and bar starts of the music (beats.py)
  plan        the edit: which camera when, where the cuts are, which effect where (cut_plan.py)
  tier        scail: the model file and settings for the card (scail_tiers.py), every file checked
  reference   the character picture cut out (SAM 3); one sentence about the character written from it
  scene       the studio with the dancer as a mannequin, from each camera of the plan (blender_views.py)
  shots       the video model paints the character for each shot (pair_run.py, which checks every chunk
              and makes a failed one again), then full size and the face (finish_pair.py)
  edit        the shots one after another with the effects, the title, the names and the music (edit_video.py)
  checks      every cut on a beat, every shot checked, the video's length and sound (check_challenge.py)
  publish     the video and a result note into the results folder (results_dir of the config)

Around the steps (run_support.py): one run folder with every file the run writes, one GPU lease for the
whole run when the config names a gpu_helper (renewed while it runs, always released), and the graphics
memory the run's processes really hold, sampled every second (outputs/results/gpu_memory.json).
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
DUO = ROOT.parent / 'duo-scene'
sys.path.insert(0, str(DUO))
import paths  # noqa: E402
import scail_tiers  # noqa: E402
from paths import ConfigError  # noqa: E402
from run_support import Failed, Lease, MemoryWatch, free_cache, given_run, gpu_memory_gib, new_run, register, stamp  # noqa: E402

OWNER = 'dance-challenge'
PORT = os.environ.get('DANCE3D_COMFY_PORT', '8189')
COMFY_URL = f'http://127.0.0.1:{PORT}'
H3_MADE = (768, 1344)  # what H3 makes; enlarged to the 1080x1920 render
HINTS = [('No person found in', '캐릭터 그림에서 사람이나 캐릭터를 찾지 못했습니다. 캐릭터 한 명이 또렷이 나온 그림을 써 주세요. / '
                                'No person or character was found in the picture: use a picture that clearly shows one character.'),
         ('does not give the stored joints', '관절 결과의 형식이 맞지 않습니다. 「춤 관절 정보」 프로그램이 만든 결과 폴더를 골라 주세요. / '
                                             'The joints result has the wrong form: choose a result folder of the dance-joints program.'),
         ('out of memory', 'GPU 메모리가 모자랍니다. 다른 GPU 작업이 끝난 뒤 이어서 하거나 (--job) 더 작은 --vram 등급으로 해 보세요. / '
                           'Not enough graphics memory: go on later with --job when other GPU work has ended, or use a smaller --vram.'),
         ('OutOfMemoryError', 'GPU 메모리가 모자랍니다. 더 작은 --vram 등급으로 해 보세요. / Not enough graphics memory: use a smaller --vram.')]
BACKEND_KEYS = {'scail': ('comfy_custom_nodes',), 'h3': ('models.h3',)}
COMMON_KEYS = ('python.motion', 'python.comfy', 'python.face', 'blender', 'comfyui', 'exmr', 'smplx', 'models.comfy', 'models.sam3',
               'models.face', 'runs_dir', 'results_dir')


def shot_prompt(base, shot):
    """H3: the instruction text of one shot: the common text and what this camera sees."""
    extra = []
    if shot['framing'] == 'close':
        extra.append('The camera is close: the character is seen from the thighs up, exactly as the mannequin is framed.')
    if shot['back_share'] >= 0.5:
        extra.append('In this shot the camera sees the character mostly from behind: show the back of the head, the hair and the '
                     'back of the outfit where the mannequin shows its back; never a face on the back of the head.')
    elif shot['direction'] in ('left', 'right') and shot['front_share'] < 0.5:
        extra.append('In this shot the camera sees the character mostly from the side: show the character in profile where the '
                     'mannequin is seen from the side.')
    return ' '.join([base] + extra)


def scail_prompt(description, shot):
    """SCAIL-2: what the finished shot shows, as one short plain caption (the form of SCAIL-2's own examples):
    the sentence the picture-reading model wrote about the character, the dance, and what this camera sees.
    Who becomes the character is fixed by the masks, not by the text."""
    who = (description or 'The character of the reference picture').strip().rstrip('.')
    text = [f'{who}.', 'The character is dancing energetically in a bright dance studio, with the same face, hairstyle, body '
            'and complete outfit as in the reference picture for the whole video.']
    if shot['back_share'] >= 0.5:
        text.append('Seen from behind: the back of the head with the hair and the back of the outfit; the face is not visible.')
    elif shot['direction'] in ('left', 'right') and shot['front_share'] < 0.5:
        text.append('Seen from the side, in profile.')
    if shot['framing'] == 'close':
        text.append('Close shot from the thighs up.')
    text.append('Natural light of the room, soft shadow on the floor, sharp focus, stable colours.')
    return ' '.join(text)


class Pipeline:
    def __init__(self, job, options):
        self.job, self.options = job, options
        self.backend = options.get('backend', 'h3')
        self.work, self.results, self.logs = job / 'outputs/work', job / 'outputs/results', job / 'logs'
        self.input = job / 'outputs/input'
        for folder in (self.work / 'stages', self.results, self.logs, self.input, job / 'cache/tmp'):
            folder.mkdir(parents=True, exist_ok=True)
        self.env = {}
        self.comfy = None
        self.comfy_options = []
        self.children = set()
        self.stopping = False
        self.started = time.time()
        self.lock = threading.Lock()
        self.total = 0
        self.watch = None
        self.gpus = []

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

    def progress(self, state, title='', error=None):
        with self.lock:
            done = len(list((self.work / 'stages').glob('*.json')))
            text = dict(state=state, step=title, steps_done=done, steps_total=max(self.total, done), started=self.started,
                        updated=time.time(), job=str(self.job), error=error)
            (self.job / 'outputs/progress.json').write_text(json.dumps(text, ensure_ascii=False))
        print(f'[{stamp()}] {state}: {title}' + (f' - {error}' if error else ''), flush=True)

    def stage(self, name, title, command, deps=(), gpu=False, quiet=False, expect=(), own_packages=False, also=''):
        """Run one step unless it is already done with the same command and the same inputs (see duo_pipeline.py)."""
        marker = self.work / 'stages' / f'{name}.json'
        stamps = []
        for dep in deps:
            dep_marker = self.work / 'stages' / f'{dep}.json'
            stamps.append(json.loads(dep_marker.read_text())['finished'] if dep_marker.exists() else None)
        # a step whose own program was changed is done again (its files are part of what it depends on)
        scripts = [hashlib.sha256(Path(part).read_bytes()).hexdigest() for part in command
                   if str(part).endswith('.py') and Path(part).parent in (ROOT, DUO) and Path(part).is_file()]
        signature = hashlib.sha256(json.dumps([[str(part) for part in command], stamps, scripts] + ([also] if also else [])).encode()).hexdigest()
        if marker.exists() and json.loads(marker.read_text()).get('signature') == signature and all(Path(f).exists() for f in expect):
            return False
        if not quiet:
            self.progress('running', title)
        if self.watch is not None and gpu:
            self.watch.step = name
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
            raise Failed('사용자가 중지했습니다. / Stopped by the user.')
        with log.open('w') as handle:
            done = subprocess.Popen([str(part) for part in command], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            with self.lock:
                self.children.add(done)
            done.wait()
            with self.lock:
                self.children.discard(done)
        if self.stopping:
            raise Failed('사용자가 중지했습니다. / Stopped by the user.')
        if done.returncode:
            text = log.read_text(errors='replace')
            tail = [line for line in text.splitlines() if line.strip() and 'Warning' not in line][-6:]
            hint = next((hint for key, hint in HINTS if key in text), '')
            if not hint:  # a step that refused the user's files says why in Korean (no program error to show then)
                refusal = next((text.rsplit('ValueError: ', 1)[-1].strip() for line in reversed(tail) if 'ValueError' in line and
                                any('가' <= ch <= '힣' for ch in line)), '')
                if refusal:
                    raise Failed(f'{title} 단계에서 멈췄습니다. {refusal}\n(기록: {log})')
            raise Failed(f'{title} 단계가 실패했습니다. {hint}\n(기록: {log})\n' + '\n'.join(tail)[-900:])
        missing = [str(f) for f in expect if not Path(f).exists()]
        if missing:
            raise Failed(f'{title} 단계가 끝났지만 결과 파일이 없습니다: {missing[0]}\n(기록: {log})')
        marker.write_text(json.dumps(dict(signature=signature, finished=time.time(), seconds=round(time.time() - started, 1))))
        return True

    # ---------- the private ComfyUI of the two-dancer program, inside this run ----------
    def comfy_up(self):
        try:
            urllib.request.urlopen(COMFY_URL + '/system_stats', timeout=3)
            return True
        except (urllib.error.URLError, OSError):
            return False

    def start_comfy(self):
        if self.comfy:
            return
        if self.comfy_up():
            raise Failed(f'생성 서버 자리(127.0.0.1:{PORT})를 다른 실행이 쓰고 있습니다. 그 실행이 끝난 뒤 다시 시작해 주세요. / '
                         f'Another run uses the port of the video model server (127.0.0.1:{PORT}); start again when it has ended '
                         '(or set DANCE3D_COMFY_PORT).')
        self.progress('running', '생성 서버 시작')
        if self.watch is not None:
            self.watch.step = 'comfy_start'
        free_cache()
        log = (self.logs / 'comfy_server.log').open('a')
        self.comfy = subprocess.Popen([sys.executable, '-B', str(DUO / 'comfy_serve.py'), *self.comfy_options], cwd=DUO, env=self.env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(240):
            if self.comfy.poll() is not None:
                raise Failed(f'생성 서버가 시작하지 못했습니다 (기록: {self.logs / "comfy_server.log"}). / The video model server did not start.')
            if self.comfy_up():
                return
            time.sleep(1)
        raise Failed('생성 서버가 4분 안에 준비되지 않았습니다. / The video model server was not ready within 4 minutes.')

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
    def choose_tier(self):
        """scail: the tier row for this card and this run's --vram, every model file checked (work/tier.json)."""
        card = gpu_memory_gib(self.gpus)
        try:
            tier = scail_tiers.choose(self.options.get('vram', 'auto'), card)
        except ValueError as error:
            raise Failed(str(error))
        models = paths.COMFY_MODELS
        missing = [f'{folder}/{name}' for folder, name in scail_tiers.model_files(tier) if not (models / folder / name).is_file()]
        if missing:
            raise Failed(f'SCAIL-2 모델 파일이 없습니다: {", ".join(missing)} (모델 폴더 {models}). 설치: SCAIL-2 모델 / '
                         f'SCAIL-2 model files are missing in models.comfy: {", ".join(missing)} (install step: SCAIL-2 models).')
        if tier['loader'] == 'gguf' and not any(folder.name.lower().startswith('comfyui-gguf') for folder in paths.custom_nodes()):
            raise Failed('GGUF 모델 파일에는 ComfyUI-GGUF 추가 노드가 필요합니다 (설정 comfy_custom_nodes). 설치: ComfyUI-GGUF / '
                         'GGUF model files need the ComfyUI-GGUF custom node in comfy_custom_nodes (install step: ComfyUI-GGUF).')
        measured = tier.pop('measured', None)  # a note on the row: changing it must not remake the shots of a resumed run
        text = json.dumps(tier, indent=1)
        file = self.work / 'tier.json'
        if not file.exists() or file.read_text() != text:
            file.write_text(text)
        print(f'[{stamp()}] SCAIL-2 tier {tier["gb"]} GB for {tier["behaves_as_gb"]:g} GB (card {tier["card_gib"]} GiB): {tier["unet"]}, '
              f'{tier["size"][0]}x{tier["size"][1]}' + (f', ComfyUI as on a {tier["cap_gb"]:g} GB card' if tier['cap_gb'] else '')
              + ('' if measured else ' (untested tier: no run with it has been measured yet)'), flush=True)
        return tier, file

    def run(self):
        from paths import BLENDER, COMFY_PYTHON, DESCRIBER, EXMR_PYTHON, FACE_MODELS, FACE_PYTHON, SAM3
        o, work, py = self.options, self.work, [EXMR_PYTHON, '-B']
        scail = self.backend == 'scail'
        command = [*py, ROOT / 'prepare_challenge.py', '--joints', o['joints'], '--character', o['character'], '--output', self.input]
        if o.get('music'):
            command += ['--music', o['music']]
        if o.get('title') is not None:
            command += ['--title', o['title']]
        self.total = 10
        self.stage('inputs', '입력 파일 검사', command, expect=[self.input / 'inputs.json'])
        inputs = json.loads((self.input / 'inputs.json').read_text())
        music = self.input / inputs['music']['file'] if inputs['music']['file'] else None

        self.stage('world', '관절을 3D 무대로 옮기기', [*py, ROOT / 'joints_to_world.py', '--results', self.input / 'results.npz',
                                                 '--info', self.input / 'info.json', '--output', work / 'world'], ['inputs'])
        # quick motion is exported a quarter larger: the video model gives back about 0.82 of it (export_scene_inputs.py)
        self.stage('export', '3D 몸 내보내기', [*py, DUO / 'export_scene_inputs.py', '--world', work / 'world', '--output',
                                            work / 'scene_inputs', '--fps', 24, '--quick-gain', 1.25], ['world'],
                   expect=[work / 'scene_inputs/scene_inputs.json'])
        seconds = json.loads((work / 'scene_inputs/scene_inputs.json').read_text())['duration_seconds']
        command = [*py, ROOT / 'beats.py', '--seconds', f'{seconds:.4f}', '--output', work / 'beats.json']
        if music:
            command += ['--music', music]
        self.stage('beats', '음악 박자 찾기', command, ['inputs', 'export'], expect=[work / 'beats.json'])
        self.stage('plan', '컷 계획 세우기', [*py, ROOT / 'cut_plan.py', '--scene-inputs', work / 'scene_inputs/scene_inputs.json',
                                         '--facts', work / 'world/motion_facts.json', '--beats', work / 'beats.json',
                                         '--output', work / 'plan.json', '--grid', 'scail' if scail else 'h3',
                                         *(['--seconds', o['seconds']] if o.get('seconds') else [])],
                   ['beats', 'export'], expect=[work / 'plan.json'])
        plan = json.loads((work / 'plan.json').read_text())
        shots = plan['shots']
        self.total = 10 + 2 * len(shots)

        if scail:
            tier, tier_file = self.choose_tier()
            made = tuple(tier['size'])
            self.comfy_options = scail_tiers.comfy_options(tier)
            tier_text = tier_file.read_text()
            reference = work / 'ref_scail'
            # the character cut out on black with its identity mask, and two enlarged views (head and shoulders, down to
            # mid-thigh) as extra references; the head crop of the picture itself for the describing step
            self.stage('reference', '캐릭터 그림 준비', [*py, DUO / 'make_reference.py', '--character', f'A={self.input / "character_A.png"}',
                                                  '--sam3', SAM3, '--output', reference, '--closeups'], ['inputs'], gpu=True,
                       expect=[reference / 'reference.png', reference / 'reference_mask.png', reference / 'reference_extra_1.png',
                               reference / 'reference_extra_2.png', reference / 'closeup_A.png'])
        else:
            made, tier_text = H3_MADE, ''
            self.stage('reference', '캐릭터 그림 준비', [*py, DUO / 'make_solo_reference.py', '--image', self.input / 'character_A.png',
                                                  '--sam3', SAM3, '--output', work / 'ref_A', '--background', 'auto'], ['inputs'], gpu=True,
                       expect=[work / 'ref_A/picture_full.png', work / 'ref_A/picture_closeup.png'])
        descriptions = {}
        if DESCRIBER is not None and DESCRIBER.is_dir():
            command = [COMFY_PYTHON, '-B', DUO / 'describe_characters.py', '--model', DESCRIBER,
                       '--picture', f'A={self.input / "character_A.png"}', '--output', work / 'describe']
            if scail:  # on a card smaller than the model, part of it waits in the computer's memory
                command += ['--gpu-gb', f'{tier["describer_gb"]:g}']
                # the hairstyle read from a close-up of the head: from the whole figure alone a small bun became "a ponytail"
                # and SCAIL-2 drew one wherever the picture does not show the back of the head
                command += ['--closeup', f'A={reference / "closeup_A.png"}']
            self.stage('describe', '캐릭터 생김새 적기', command, ['inputs', 'reference'] if scail else ['inputs'], gpu=True,
                       own_packages=True, expect=[work / 'describe/description_A.txt'])
            descriptions = {1: (work / 'describe/description_A.txt').read_text().strip()}
        guide_pass = 'guide' if scail else 'pair'
        self.stage('scene', '3D 화면 그리기 (네 방향)', [BLENDER, '-b', '--factory-startup', '--python-exit-code', 1, '--python',
                                                  ROOT / 'blender_views.py', '--', '--inputs', work / 'scene_inputs', '--plan',
                                                  work / 'plan.json', '--output', work / 'scene', '--lighting', 'shadow',
                                                  '--passes', f'{guide_pass},plate,mask', '--facts', work / 'world/motion_facts.json'],
                   ['plan', 'export', 'world'], gpu=True,
                   expect=[work / 'scene' / view / 'manifest.json' for view in plan['views']])

        if scail:
            model_arguments = ['--backend', 'scail', '--reference', reference, '--tier', tier_file]
            face_picture = self.input / 'character_A.png'
        else:
            from pair_prompt import build_prompt
            base = build_prompt(1, True, {}, descriptions, {})
            model_arguments = ['--backend', 'h3', '--picture', work / 'ref_A/picture_full.png', '--picture', work / 'ref_A/picture_closeup.png']
            face_picture = self.input / 'character_A.png'
        finishing = ThreadPoolExecutor(max_workers=1)  # the CPU work of one shot runs beside the next shot's generation
        waiting = []

        def finish(shot):
            run = work / 'gen' / shot['name']
            self.stage(f'finish_{shot["name"]}', f'마무리 ({shot["index"]}/{len(shots)})',
                       ['nice', '-n', '10', FACE_PYTHON, '-B', DUO / 'finish_pair.py', '--run', run, '--inputs', work / 'scene_inputs',
                        '--face', f'A={face_picture}', '--models', FACE_MODELS,
                        '--output', work / 'finished' / f'{shot["name"]}.mp4', '--frames-folder', work / 'finished' / shot['name']],
                       [f'generate_{shot["name"]}'], quiet=True, expect=[work / 'finished' / shot['name'] / 'frame_0001.png'])

        try:
            for shot in shots:
                text = scail_prompt(descriptions.get(1), shot) if scail else shot_prompt(base, shot)
                prompt = work / 'prompts' / f'{shot["name"]}.txt'
                if not prompt.exists() or prompt.read_text() != text:
                    prompt.parent.mkdir(parents=True, exist_ok=True)
                    prompt.write_text(text)
                if not (work / 'stages' / f'generate_{shot["name"]}.json').exists():
                    self.start_comfy()
                start, frames = shot['window']
                away = ','.join(f'{a}-{b}' for a, b in shot.get('away_frames', []))
                hair = []
                if scail:  # added hair around the head: measured in every shot, limited in side and back shots by the
                    # front shots made so far (they show what the picture shows)
                    hair = ['--inputs', work / 'scene_inputs']
                    if shot['view'] not in ('front', 'front_close'):
                        for earlier in shots[:shots.index(shot)]:
                            if earlier['view'] in ('front', 'front_close'):
                                hair += ['--hair-from', work / 'gen' / earlier['name'] / 'run.json']
                self.stage(f'generate_{shot["name"]}', f'캐릭터 입히기 ({shot["index"]}/{len(shots)}: {shot["label"]}'
                           + (', 가까이' if shot['framing'] == 'close' else '') + ')',
                           [*py, DUO / 'pair_run.py', '--scene', work / 'scene' / shot['view'], *model_arguments, '--prompt', prompt,
                            '--output', work / 'gen' / shot['name'], '--tag', shot['name'], '--width', made[0], '--height', made[1],
                            '--first', start, '--frames', frames, *(['--faces-away', away] if scail and away else []), *hair],
                           ['scene', 'reference'], gpu=True, also=text + tier_text, expect=[work / 'gen' / shot['name'] / 'run.json'])
                waiting.append(finishing.submit(finish, shot))
        finally:
            self.stop_comfy()
            if self.watch is not None:
                self.watch.step = 'after_generation'
        self.progress('running', '장면 마무리')
        for future in waiting:
            future.result()
        finishing.shutdown()

        title = inputs['title']
        command = [*py, ROOT / 'edit_video.py', '--plan', work / 'plan.json', '--finished', work / 'finished', '--output',
                   self.results / 'challenge.mp4', '--title', title, '--effects', o['effects'], '--compare',
                   self.results / 'compare.mp4', '--scene', work / 'scene', '--guide-pass', guide_pass]
        if music:
            command += ['--music', music]
        if not o.get('labels', True):
            command.append('--no-labels')
        self.stage('edit', '컷 편집과 효과', command, [f'finish_{shot["name"]}' for shot in shots], expect=[self.results / 'challenge.mp4'])
        self.stage('checks', '결과 확인', ['nice', '-n', '10', *py, ROOT / 'check_challenge.py', '--job', self.job], ['edit'],
                   expect=[self.results / 'checks.json'])
        self.stage('summary', '결과 정리', [*py, ROOT / 'challenge_summary.py', '--job', self.job], ['checks'],
                   expect=[self.results / '결과_안내.txt'])
        return self.publish(title)

    def publish(self, title):
        target = paths.RESULTS / self.job.name
        self.progress('running', '결과 폴더에 저장')
        target.mkdir(parents=True, exist_ok=True)
        name = '_'.join(title.split()) if title else '댄스'
        sums = {}
        for source, file in ((self.results / 'challenge.mp4', f'{name}_챌린지.mp4'),
                             (self.results / 'compare.mp4', '3D화면_만든장면_완성_나란히.mp4'),
                             (self.results / '결과_안내.txt', '결과_안내.txt'), (self.results / 'summary.json', '결과_요약.json')):
            if source.exists():
                shutil.copy2(source, target / file)
                sums[file] = hashlib.sha256((target / file).read_bytes()).hexdigest()
        (target / 'sha256.json').write_text(json.dumps(sums, ensure_ascii=False, indent=1))
        return target


def main():
    parser = argparse.ArgumentParser(description='한 사람의 관절 결과 + 캐릭터 그림 -> 정면·왼쪽·오른쪽·뒤를 박자에 맞춰 자른 댄스 챌린지 영상 / '
                                                 'one dancer\'s joints + one character picture -> a dance challenge video')
    parser.add_argument('--joints', help='「춤 관절 정보 만들기」 결과 안의 영상 폴더 (예: .../02_L을가져가) / one video\'s folder of the '
                                          'dance-joints results')
    parser.add_argument('--character', help='캐릭터 그림 / the character picture')
    parser.add_argument('--music', help='음악 파일. 없으면 관절을 뽑은 춤 영상의 소리 / music file (default: the dance video\'s sound)')
    parser.add_argument('--title', help='영상 첫머리에 넣을 제목. 없으면 관절 결과의 제목, 빈 글자면 넣지 않음')
    parser.add_argument('--backend', default='scail', choices=['scail', 'h3'],
                        help='영상 모델: scail (SCAIL-2, MIT, 기본) 또는 h3 (MiniMax H3; 한국·미국·EU·영국 사용 불가 약관)')
    parser.add_argument('--vram', choices=['auto', '8', '12', '16', '24'],
                        help='scail: 그래픽 카드 메모리 등급 (GB). 기본: 설정 vram_gb ("auto" = 카드의 메모리). 카드보다 작게 주면 '
                             '나머지를 비워 두고 그 크기의 카드처럼 돕니다 / SCAIL-2 memory tier in GB (default: vram_gb of the config)')
    parser.add_argument('--effects', default='normal', choices=['normal', 'strong', 'none'], help='효과의 세기 (none: 컷만)')
    parser.add_argument('--no-labels', action='store_true', help='장면마다 정면·왼쪽·오른쪽·뒤 이름을 넣지 않음')
    parser.add_argument('--seconds', type=float, help='춤의 앞부분 이만큼(초)만 만들기 (빨리 해 보기) / make only the first S seconds')
    parser.add_argument('--name', default='dance-challenge', help='작업 이름 / run name')
    parser.add_argument('--job', help='이어서 할 작업 폴더 / a run folder to go on with')
    args = parser.parse_args()

    try:
        if args.job:
            job = given_run(args.job)  # the same spelling, or finished steps are not recognised
            options = json.loads((job / 'outputs/options.json').read_text())
            options.setdefault('backend', 'h3')  # runs made before the SCAIL-2 backend
        else:
            if not args.joints or not args.character:
                parser.error('--joints 와 --character 가 필요합니다 / --joints and --character are needed')
            for path in (args.joints, args.character) + ((args.music,) if args.music else ()):
                if not Path(path).expanduser().exists():
                    parser.error(f'파일이나 폴더가 없습니다 / no such file or folder: {path}')
            vram = args.vram or paths.vram_setting()
            options = dict(joints=str(Path(args.joints).expanduser().resolve()), character=str(Path(args.character).expanduser().resolve()),
                           music=str(Path(args.music).expanduser().resolve()) if args.music else None, title=args.title,
                           effects=args.effects, labels=not args.no_labels, name=args.name, backend=args.backend,
                           seconds=f'{args.seconds:g}' if args.seconds else None,
                           vram=vram if vram == 'auto' else f'{float(vram):g}')
        paths.require(*COMMON_KEYS, *BACKEND_KEYS[options['backend']])
        if not args.job:
            job = new_run(OWNER, options['name'])
    except (ConfigError, Failed) as error:
        print(error, file=sys.stderr, flush=True)
        raise SystemExit(2)
    os.environ['DUO_JOB'] = str(job)  # the run folder of every step (paths.JOB)
    pipeline = Pipeline(job, options)
    (job / 'outputs/options.json').write_text(json.dumps(options, ensure_ascii=False, indent=1))
    print(f'[{stamp()}] run {job} (backend {options["backend"]}' + (f', vram {options.get("vram")}' if options['backend'] == 'scail' else '')
          + ')', flush=True)
    if options['backend'] == 'h3':
        print(f'[{stamp()}] note: the MiniMax H3 licence excludes use in Korea, the US, the EU and the UK', flush=True)
    signal.signal(signal.SIGTERM, lambda signum, frame: pipeline.stop_children())
    signal.signal(signal.SIGINT, lambda signum, frame: pipeline.stop_children())
    lease = None
    try:
        lease = Lease(OWNER, job.name)
        pipeline.gpus = lease.gpus
        pipeline.env = paths.cache_environment()
        pipeline.env.update(lease.environment())
        with MemoryWatch(job / 'outputs/results/gpu_memory.json') as watch:
            pipeline.watch = watch
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
