"""Stream aligned overlapping clips to one native-size video, without retiming."""
import argparse
import json
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def merge(manifest, output):
    data = json.loads(Path(manifest).read_text())
    width,height,total = data['width'],data['height'],data['frames']
    overlap = data['overlap']
    command = ['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24',
        '-s',f'{width}x{height}','-r','24','-i','pipe:0','-an','-c:v','libx264',
        '-crf','16','-preset','slow','-pix_fmt','yuv420p','-threads','4',
        '-movflags','+faststart',str(output)]
    encoder = subprocess.Popen(command,stdin=subprocess.PIPE)
    tail = deque()
    written = 0

    def emit(frame):
        nonlocal written
        if written < total:
            encoder.stdin.write(frame.tobytes())
            written += 1

    try:
        for index,part in enumerate(data['segments']):
            if index and part['start'] != written:
                raise ValueError('Segment time does not match the retained overlap')
            cap = cv2.VideoCapture(part['path'])
            if not cap.isOpened():
                raise ValueError('Cannot read segment: '+part['path'])
            count = 0
            previous = list(tail)
            tail.clear()
            try:
                while True:
                    ok,frame = cap.read()
                    if not ok:
                        break
                    if frame.shape != (height,width,3):
                        raise ValueError('Segment dimensions differ')
                    if index and count < overlap:
                        alpha = (count+1)/(overlap+1)
                        frame = cv2.addWeighted(previous[count],1-alpha,frame,alpha,0)
                    tail.append(frame)
                    if len(tail) > overlap:
                        emit(tail.popleft())
                    count += 1
            finally:
                cap.release()
            if count != part['frames']:
                raise ValueError(f'Segment {index}: expected {part["frames"]}, got {count}')
        while tail:
            emit(tail.popleft())
        if written != total:
            raise ValueError(f'Missing output frames: {written}/{total}')
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('Video encoding failed')
    finally:
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
    print(json.dumps({'frames':written,'width':width,'height':height,'fps':24,'upscaled':False}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    merge(args.manifest,args.output)
