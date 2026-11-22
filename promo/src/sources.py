"""The material of the dance promo films: challenge runs, the two-dancer run, the robot footage, the brand pieces."""
import functools
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from kit import BRAND, FPS, GREEN, WORK, Frames, bgr
from skeleton import Dance, unit

JOINTS = Path('/home/steerix-train-server/바탕화면/춤 관절 정보 만들기/20261114-124936-dance-joints-f011')
JOINTS_JOB = Path('/srv/training/jobs/20261114-124936-dance-joints-f011')
CHALLENGE_PROGRAM = Path('/srv/training/projects/dance-challenge')
ROBOT_VIDEO = Path('/home/steerix-train-server/바탕화면/danceC.MP4')
DANCES = {int(p.name[:2]): p for p in sorted(JOINTS.glob('[0-9][0-9]_*'))}


@functools.lru_cache(maxsize=None)
def dance(number):
    return Dance(DANCES[number])


@functools.lru_cache(maxsize=4)
def outlines(number):
    """The dancer's outline in every source frame of a dance (the tracker's masks, half size), turned the way the joints are."""
    work = JOINTS_JOB / 'outputs/videos' / f'{number:02d}' / 'work'
    packed = np.load(work / 'tracks/masks.npz')
    height, width = int(packed['shape'][1]), int(packed['shape'][2])
    masks = packed['A']
    flipped = bool(np.load(DANCES[number] / 'results.npz')['flipped_left_right'])

    def at(index):
        mask = np.unpackbits(masks[int(np.clip(index, 0, len(masks) - 1))], axis=1)[:, :width]
        return mask[:, ::-1] if flipped else mask

    return at


def bar_grid(beats, bars):
    """The steady bar length and the time of a bar start that fit the tracked bars best (seconds)."""
    bars = np.asarray(bars, float)
    index = np.arange(len(bars))
    period, first = np.polyfit(index, bars, 1)
    return float(period), float(first)


def envelope_of(sound, seconds):
    """Onset strength of a sound file over time, as the challenge program's beat tracker sees it."""
    sys.path.insert(0, str(CHALLENGE_PROGRAM))
    import beats as tracker
    signal = tracker.read_sound(sound, seconds)
    strength, _ = tracker.onset_strength(signal)
    t = np.arange(len(strength)) * tracker.HOP / tracker.RATE
    return t, gaussian_filter1d(strength, 2.0)


class Challenge:
    """A finished run of the challenge program: plan, beats, the 3D guide and the shots as made."""

    def __init__(self, job, name):
        self.job, self.name = Path(job), name
        work = self.job / 'outputs/work'
        self.plan = json.loads((work / 'plan.json').read_text())
        self.fps, self.frames = self.plan['fps'], self.plan['frames']
        self.seconds = self.frames / self.fps
        self.shots, self.cuts = self.plan['shots'], self.plan['cuts']
        record = json.loads((work / 'beats.json').read_text())
        self.beats, self.bars = np.array(record['beats']), np.array(record['bars'])
        self.bar, self.first_bar = bar_grid(self.beats, self.bars)
        self.checks = json.loads((self.job / 'outputs/results/checks.json').read_text())
        self.inputs = json.loads((self.job / 'outputs/input/inputs.json').read_text())
        self.work = work
        self.clean = Frames(f'{name}_clean', WORK / 'clips' / f'{name}_clean.mp4')
        self.owner = np.zeros(self.frames + 2, int)
        for index, shot in enumerate(self.shots):
            self.owner[shot['first']:shot['last'] + 1] = index
        cache = WORK / 'beats' / f'{name}_envelope.npz'
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            music = self.job / 'outputs/input' / self.inputs['music']['file']
            t, v = envelope_of(music, self.seconds)
            np.savez(cache, t=t, v=v)
        data = np.load(cache)
        self.envelope = (data['t'], data['v'])

    def number(self, seconds):
        return int(np.clip(int(seconds * self.fps) + 1, 1, self.frames))

    def shot(self, seconds):
        return self.shots[self.owner[self.number(seconds)]]

    def shot_of(self, direction, which=0, framing='full'):
        """The which-th shot filmed from a direction (front, left, right, back), longest ones first."""
        found = sorted((s for s in self.shots if s['direction'] == direction and s['framing'] == framing), key=lambda s: -s['frames'])
        return found[min(which, len(found) - 1)] if found else None

    def shot_frame(self, shot, seconds):
        """The frame number `seconds` into a shot (held on its last frame)."""
        return int(min(shot['last'], shot['first'] + int(seconds * self.fps)))

    @functools.lru_cache(maxsize=8)
    def guide(self, number, view=None):
        """The 3D guide picture of a frame (the mannequin in the studio, seen by the camera of that shot)."""
        view = view or self.shots[self.owner[number]]['view']
        return cv2.imread(str(self.work / 'scene' / view / 'pair' / f'frame_{number:04d}.jpg'))

    @functools.lru_cache(maxsize=8)
    def made(self, number):
        """The frame as the video model made it (before the edit's effects)."""
        shot = self.shots[self.owner[number]]
        start, count = shot['window']
        k = int(np.clip(number - start + 1, 1, count))
        return cv2.imread(str(self.work / 'finished' / shot['name'] / f'frame_{k:04d}.png'))


class Robot:
    """The owner's footage of the real robot dancing dance 09, with the eyes followed frame by frame."""
    MUSIC_START = 13.05   # where the dance music starts in the footage (sound match)
    LAG = 0.17            # the robot's head is this much behind the music (best fit of the head tilt)

    def __init__(self):
        self.times = np.array([float(v.strip(',')) for v in subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', str(ROBOT_VIDEO)],
            capture_output=True, text=True, check=True).stdout.split()])
        self.frames = Frames('robot', ROBOT_VIDEO, pattern='%05d.jpg')
        rows = [r for r in json.loads((WORK / 'robot/eyes.json').read_text())['rows'] if r['found']]
        self.eye_rows = {r['frame']: r for r in rows}
        self.eye_times = np.array([r['time'] for r in rows])
        rest = np.mean([r['tilt_deg'] for r in rows if 9.0 < r['time'] < 12.5])
        self.tilt = np.array([r['tilt_deg'] for r in rows]) - rest
        self.eye_centre = np.array([r['centre'] for r in rows])
        self.dance = dance(9)
        ears = self.dance.joints[:, 59] - self.dance.joints[:, 58]
        self.head_roll = gaussian_filter1d(np.degrees(np.arcsin(unit(ears)[:, 1])), self.dance.fps * 0.05)
        self.dance_times = np.arange(self.dance.frames) / self.dance.fps
        joints = self.dance.joints
        shoulders = joints[:, 16] - joints[:, 17]
        facing = np.cross(shoulders, [0.0, 1.0, 0.0])
        ahead = self.dance.forward
        side = np.cross([0.0, 1.0, 0.0], ahead)
        self.body_turn = gaussian_filter1d(np.degrees(np.arctan2(facing @ side, facing @ ahead)), self.dance.fps * 0.05)   # + = turned to the dancer's left
        arm = lambda shoulder, wrist: 90 - np.degrees(np.arccos(np.clip(unit(joints[:, wrist] - joints[:, shoulder])[:, 1], -1, 1)))
        self.arms = (gaussian_filter1d(arm(16, 20), self.dance.fps * 0.05), gaussian_filter1d(arm(17, 21), self.dance.fps * 0.05))   # + = hand above the shoulder
        record = json.loads((WORK / 'beats/dance09.json').read_text())
        self.beats, self.bars = np.array(record['beats']), np.array(record['bars'])
        self.bar, self.first_bar = bar_grid(self.beats, self.bars)
        # how well the robot's head tilt follows the dancer's (mirrored), over the frames where the eyes were found
        t, robot_tilt = self.tilt_series()
        inside = (t >= 0) & (t <= self.dance.seconds)
        dancer = np.interp(t[inside], self.dance_times, self.head_roll)
        self.correlation = float(np.corrcoef(dancer, robot_tilt[inside])[0, 1])
        self.fit = tuple(float(v) for v in np.polyfit(dancer, robot_tilt[inside], 1))   # robot degrees per dancer degree, offset
        self.measured = int(inside.sum())

    def eyes(self, footage_seconds):
        """The measured eye positions of the footage frame shown at this moment (None if they were not found there)."""
        index = int(np.clip(np.searchsorted(self.times, footage_seconds), 0, len(self.times) - 1))
        return self.eye_rows.get(index + 1)

    def footage_time(self, dance_seconds):
        return self.MUSIC_START + self.LAG + dance_seconds

    def picture(self, footage_seconds):
        return self.frames.get(int(np.clip(np.searchsorted(self.times, footage_seconds), 0, len(self.times) - 1)))

    def seen(self, footage_seconds, reach=0.2):
        """Was an eye measurement taken within `reach` seconds of this moment?"""
        k = np.clip(np.searchsorted(self.eye_times, footage_seconds), 1, len(self.eye_times) - 1)
        return min(abs(self.eye_times[k] - footage_seconds), abs(self.eye_times[k - 1] - footage_seconds)) < reach

    def tilt_series(self):
        """(dance time, robot head tilt mirrored to the dancer's side) with gaps where the eyes were not found."""
        t = self.eye_times - self.MUSIC_START - self.LAG
        return t, -self.tilt


@functools.lru_cache(maxsize=None)
def steerix_logo(height, symbol_colour=GREEN):
    """The STEERIX logo (symbol + letters), white with the symbol in brand green, as a layer of the given height."""
    image = Image.open(BRAND / '1-STEERIX/로고(심볼+글자)/steerix-logo-white.png').convert('RGBA')
    image = image.crop((40, 40, 3160, 434))
    array = np.asarray(image)
    colour = np.empty((*array.shape[:2], 3), np.uint8)
    colour[:] = bgr('#FFFFFF')
    colour[:, :615] = symbol_colour
    width = int(round(array.shape[1] * height / array.shape[0]))
    alpha = cv2.resize(array[..., 3], (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return cv2.resize(colour, (width, height), interpolation=cv2.INTER_NEAREST), alpha


def brand_clip(name, relative):
    """A brand animation as frames at the film's rate."""
    return Frames(name, BRAND / relative, fps=FPS)


def picture(path, height=None, width=None, background=(255, 255, 255)):
    """A still (webp/png/jpg) as BGR, transparent parts on a plain background, optionally resized."""
    image = Image.open(path).convert('RGBA')
    plain = Image.new('RGBA', image.size, (*background, 255))
    array = cv2.cvtColor(np.asarray(Image.alpha_composite(plain, image).convert('RGB')), cv2.COLOR_RGB2BGR)
    if height or width:
        scale = (height / array.shape[0]) if height else (width / array.shape[1])
        array = cv2.resize(array, (int(round(array.shape[1] * scale)), int(round(array.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return array


class Duo:
    """A finished run of the two-dancer program: the 3D guide of each camera, the finished videos, the numbers of its summary."""

    def __init__(self, job, desktop):
        self.job, self.desktop = Path(job), Path(desktop)
        work = self.job / 'outputs/work'
        self.work = work
        names = {'original': '1_원본과_같은_카메라.mp4', 'orbit': '2_360도_회전.mp4'}
        if (self.desktop / '결과_요약.json').exists():
            self.summary = json.loads((self.desktop / '결과_요약.json').read_text())
        else:   # the run is still going: only the videos already in the result folder, no numbers yet
            self.summary = dict(videos={view: dict(file=name) for view, name in names.items() if (self.desktop / name).exists()})
        self.fps = 24.0
        self.clean = json.loads((WORK / 'duo_clean.json').read_text())[self.job.name]
        self.videos = {}
        for view, record in self.summary['videos'].items():
            self.videos[view] = Frames(f'duo_{view}', self.desktop / record['file'])
        self.frames = min(v.count for v in self.videos.values())
        self.seconds = self.frames / self.fps
        record = json.loads((WORK / 'beats/duo_neon.json').read_text())
        self.beats, self.bars = np.array(record['beats']), np.array(record['bars'])
        self.bar, self.first_bar = bar_grid(self.beats, self.bars)
        world = np.load(work / 'world/people_world.npz', allow_pickle=True)
        from skeleton import Motion, Stage
        self.stage = Stage([Motion(world['A_joints'], float(world['fps'])), Motion(world['B_joints'], float(world['fps']))])
        self.pieces, self.piece_checks = {}, {}
        for view in self.videos:
            run = work / 'gen' / view / 'run.json'
            if run.exists():   # the stretches of frames each piece of the video was made for (5 frames shared between neighbours)
                record = json.loads(run.read_text())
                lengths = record['chunk_frames']
                starts = np.concatenate([[0], np.cumsum(np.array(lengths[:-1]) - 5)])
                self.pieces[view] = [(float(a) / self.fps, float(a + n) / self.fps) for a, n in zip(starts, lengths)]
                kept = [entry['tries'][entry['kept_try']] for entry in record['runs']]   # what the program's own checks found in each kept piece
                self.piece_checks[view] = dict(
                    pieces=len(record['runs']), planned=len(lengths), made_again=sum(len(entry['tries']) - 1 for entry in record['runs']),
                    out_of_step=sum(len(k['frames_out_of_step']) for k in kept), off_the_room=sum(len(k['frames_off_the_room']) for k in kept),
                    wrong_head=sum(len(k['frames_with_the_wrong_head']) for k in kept),
                    wrong_clothes=sum(len(k.get('frames_in_the_other_ones_colours', [])) for k in kept),
                    clothes_checked=all('frames_in_the_other_ones_colours' in k for k in kept),
                    mannequin_left=sum(len(k['frames_with_a_mannequin_left']) for k in kept))
        self.paths = {}
        for view in self.videos:
            path = work / 'scene' / view / 'camera-path.json'
            if path.exists():
                self.paths[view] = json.loads(path.read_text())
        # floor positions in the room's own axes (the 3D program's x and y; the reconstruction's x and -z), per frame of the videos
        self.set_centre = tuple(json.loads((work / 'scene' / next(iter(self.videos)) / 'manifest.json').read_text())['set']['centre'])
        times = np.arange(self.frames) / self.fps
        self.floor = []
        for dancer in self.stage.dancers:
            pelvis = dancer.joints[np.clip(np.round(times * dancer.fps).astype(int), 0, dancer.frames - 1), 0]
            self.floor.append(np.stack([pelvis[:, 0], -pelvis[:, 2]], axis=1))
        self.camera_floor = {view: np.array([row['location'][:2] for row in rows]) for view, rows in self.paths.items()}

    def number(self, seconds):
        return int(np.clip(int(seconds * self.fps) + 1, 1, self.frames))

    # ---- the stretches found right in a look at every frame (work/duo_clean.json: {view: [[from, to], ...]} in seconds)
    def is_clean(self, view, start, end):
        return any(a - 1e-6 <= start and end <= b + 1e-6 for a, b in self.clean.get(view, []))

    def clean_from(self, view, wanted, seconds):
        """`wanted` if `seconds` from there are clean, else the start of the clean stretch that fits and is nearest."""
        if self.is_clean(view, wanted, wanted + seconds):
            return wanted
        fitting = [(a, b) for a, b in self.clean.get(view, []) if b - a >= seconds]
        if not fitting:
            raise ValueError(f'no clean stretch of {seconds:.1f} s in the {view} video')
        a, b = min(fitting, key=lambda span: abs(np.clip(wanted, span[0], span[1] - seconds) - wanted))
        return float(np.clip(wanted, a, b - seconds))

    def reviewed_clean(self, view):
        """Is the whole video right, from its first frame to its last?"""
        return self.is_clean(view, 0.0, self.seconds - 0.05)

    @functools.lru_cache(maxsize=8)
    def guide(self, view, number):
        return cv2.imread(str(self.work / 'scene' / view / 'pair' / f'frame_{number:04d}.jpg'))

    @functools.lru_cache(maxsize=8)
    def plate(self, view, number):
        """The empty 3D room as that camera sees it."""
        return cv2.imread(str(self.work / 'scene' / view / 'plate' / f'frame_{number:04d}.jpg'))

    @functools.lru_cache(maxsize=8)
    def mask(self, view, who, number):
        return cv2.imread(str(self.work / 'scene' / view / f'mask_{who}' / f'frame_{number:04d}.png'), cv2.IMREAD_GRAYSCALE)

    def result(self, view, number):
        return self.videos[view].get(number - 1)

    def orbit_angle(self, number):
        """Where the orbit camera stands, in degrees round the stage centre from the first frame's place."""
        rows = self.paths['orbit']
        centre = np.array(json.loads((self.work / 'scene/orbit/manifest.json').read_text())['set']['centre'])
        first = np.array(rows[0]['location'][:2]) - centre
        now = np.array(rows[min(len(rows), number) - 1]['location'][:2]) - centre
        return float(np.degrees(np.arctan2(now[1], now[0]) - np.arctan2(first[1], first[0])))
