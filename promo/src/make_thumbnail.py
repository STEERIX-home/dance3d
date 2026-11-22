"""A cover picture for the long film (1920x1080 and 1280x720): the three results side by side and a short line.

    make_thumbnail.py OUT_FOLDER
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources  # noqa: E402
import widgets as wd  # noqa: E402
from common import covered, video_card  # noqa: E402
from film import Film  # noqa: E402
from kit import AMBER, GREEN, PANEL, SKY, WHITE, box, put, text_layer  # noqa: E402
from skeleton import figure_panel  # noqa: E402

out = Path(sys.argv[1])
film = Film('long')
W, H = film.W, film.H
c = film.backdrop.copy()
robot, main, duo, nine = film.robot, film.main, film.duo, film.nine
d = robot.first_bar + 2 * robot.bar + 1.2      # a lively moment of the robot's dance, in a front shot of the character video
wide, tall, y, gap = 400, 720, 250, 26
x0 = W - 64 - 3 * wide - 2 * gap
box(c, x0, y, wide, tall, PANEL, 1.0, 20)
figure_panel(c, robot.dance, d, (x0 + 4, y + 4, wide - 8, tall - 8), yaw_deg=-18, glow_strength=1.0)
source = nine or main
shot = source.shot_of('front', 0)
character = source.made(source.shot_frame(shot, 0.6)) if nine is None else source.clean.get(source.number(d) - 1)
video_card(c, character, (x0 + wide + gap, y, wide, tall), focus=(0.5, 0.2), zoom=1.15, border=WHITE)
video_card(c, robot.picture(robot.footage_time(d))[:, 600:1240], (x0 + 2 * (wide + gap), y, wide, tall), focus=(0.5, 0.35), border=WHITE)
for index, (name, colour) in enumerate((('3D JOINTS', GREEN), ('CHARACTER', SKY), ('REAL ROBOT', AMBER))):
    wd.chip(c, name, x0 + index * (wide + gap) + wide / 2, y + tall + 18, colour, 1.0, 26, anchor='ct', over_picture=False)
put(c, sources.steerix_logo(44), 72, 70)
for index, (line, accents) in enumerate((('ONE', None), ('DANCE', None), ('VIDEO', (('VIDEO', GREEN),)))):
    put(c, text_layer(line, 'semibold', 158, WHITE, accents=accents), 64, 270 + index * 178)
put(c, text_layer('→ characters, cameras,', 'medium', 40, WHITE), 72, 840)
put(c, text_layer('   and a real robot', 'medium', 40, WHITE), 72, 892)
out.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out / '썸네일_롱폼_1920.jpg'), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite(str(out / '썸네일_롱폼_1280.jpg'), cv2.resize(c, (1280, 720), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
short = Film('short')
hook = next(scene for scene in short.scenes if scene.name == 'hook_2')
cv2.imwrite(str(out / '썸네일_숏폼_1080x1920.jpg'), short.frame(int(round((hook.start + 1.3) * 24))), [cv2.IMWRITE_JPEG_QUALITY, 92])
print('written', out)
