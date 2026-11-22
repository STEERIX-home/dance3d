"""English-captioned landscape demonstration of one motion and two appearances."""
import argparse
import json
import math
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

WIDTH, HEIGHT, FPS = 1920, 1080, 24
MOTION_FRAMES, INTRO_FRAMES, OUTRO_FRAMES = 699, 120, 144
TOTAL_FRAMES = INTRO_FRAMES+4*MOTION_FRAMES+OUTRO_FRAMES
BG = '#101724'
WHITE, MUTED = '#edf2fa', '#a8b5c8'
ACCENTS = ['#76ded5','#ffc99d','#c4a4ff']
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


@lru_cache(maxsize=32)
def font(size, bold=False):
    return ImageFont.truetype(BOLD_PATH if bold else FONT_PATH,size)


def text(draw,xy,value,size=28,fill=WHITE,bold=False):
    draw.text(xy,value,font=font(size,bold),fill=fill)


def wrapped(draw,xy,value,width,size=28,fill=MUTED,line_height=None,bold=False):
    face=font(size,bold)
    x,y=xy
    line_height=line_height or int(size*1.45)
    for paragraph in value.split('\n'):
        line=''
        for word in paragraph.split():
            candidate=(line+' '+word).strip()
            if line and draw.textlength(candidate,font=face)>width:
                draw.text((x,y),line,font=face,fill=fill);y+=line_height;line=word
            else:
                line=candidate
        draw.text((x,y),line,font=face,fill=fill);y+=line_height
    return y


def contain(canvas,image,box,background='#1b2638'):
    x,y,w,h=map(int,box)
    draw=ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x,y,x+w,y+h),radius=18,fill=background)
    if not isinstance(image,Image.Image):
        image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    scale=min(w/image.width,h/image.height,1)
    resized=image.convert('RGB').resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)
    canvas.paste(resized,(x+(w-resized.width)//2,y+(h-resized.height)//2))


def first_frame(path):
    cap=cv2.VideoCapture(str(path));ok,frame=cap.read();cap.release()
    if not ok:
        raise ValueError('Cannot read '+str(path))
    return frame


def camera_phase(seconds):
    ends=[3,6,9,12,15,18,21,24,26.5,29.125]
    labels=['Front-facing dolly in','Left arc and a closer view','Pull back to the full performance',
            'Right arc with parallax','A gentle height change','Dolly in with subtle roll',
            'Pull back and recover the full body','A right-side approach','A closer front view','Return to the front']
    return labels[min(len(labels)-1,int(np.searchsorted(ends,seconds,side='right')))]


CAPTIONS = [
    [
        (0,4,'Reuse the existing 3D motion. Choreography and timing stay unchanged.'),
        (4,8,'Dolly in: move the 50 mm camera closer in 3D.'),
        (8,12,'Dolly out and arc: stay within the front viewing range.'),
        (12,16,'Change camera height while following the same dancer.'),
        (16,20,'Subtle roll adds motion while the face and hands stay framed.'),
        (20,24,'Render this camera path once, then reuse the same guide.'),
        (24,29.125,'Return to the front. No full orbit or digital enlargement.'),
    ],
    [
        (0,5,'Condition H3 on the photograph and the shared guide video.'),
        (5,10,'Body and face references guide the appearance and clothing.'),
        (10,15,'The rendered camera guides perspective, framing and motion.'),
        (15,20,'Generate six overlapping windows on the same timeline.'),
        (20,25,'Carry five generated frames forward to anchor each continuation.'),
        (25,29.125,'Correct clearly visible faces using the supplied photograph.'),
    ],
    [
        (0,5,'Change the appearance reference to reze.webp.'),
        (5,10,'Use the reference for anime linework, colors, hair and costume.'),
        (10,15,'Reuse the guide: no new motion extraction or camera design.'),
        (15,20,'Keep the anime appearance free of photographic face correction.'),
        (20,25,'Join overlapping windows without repeating the dance.'),
        (25,29.125,'The appearance changes; the motion and camera guide stay shared.'),
    ],
    [
        (0,5,'Shared motion and camera input. All three videos share the same timeline.'),
        (5,10,'Distance changes in 3D. The generated videos are not enlarged.'),
        (10,15,'Each view uses the same frame number and source timestamp.'),
        (15,20,'3D geometry controls the guide; H3 synthesizes two appearances.'),
        (20,25,'Shared guidance does not guarantee identical poses, details or backgrounds.'),
        (25,29.125,'One performance. One camera path. Two appearance references.'),
    ],
]


def caption(chapter,seconds):
    for start,end,line in CAPTIONS[chapter]:
        if start<=seconds<end:
            return line
    return CAPTIONS[chapter][-1][2]


class Design:
    def __init__(self,job,source):
        self.job=job
        self.camera=json.loads((job/'outputs/blender/camera-path.json').read_text())
        self.validation=json.loads((job/'outputs/blender/camera-validation.json').read_text())
        self.photo=Image.open(job/'outputs/inputs/character-photo.png').convert('RGB')
        self.reze=Image.open(job/'outputs/inputs/reze.png').convert('RGB')
        self.source=first_frame(source)
        self.corrected=None
        stage=job/'outputs/photo-stage.json'
        if stage.exists():
            report=Path(json.loads(stage.read_text())['job'])/'outputs/face-report.json'
            self.corrected=json.loads(report.read_text())['corrected_frames']

    def intro(self,frame):
        canvas=Image.new('RGB',(WIDTH,HEIGHT),BG);draw=ImageDraw.Draw(canvas)
        text(draw,(96,50),'ONE MOTION. ONE CAMERA.',58,bold=True)
        text(draw,(96,122),'Two character appearances.',48,fill=ACCENTS[2],bold=True)
        text(draw,(98,195),'A 3D-guided character replacement workflow',28,fill=MUTED)
        images=[self.source,self.photo,self.reze]
        names=['SOURCE DANCE','PHOTO REFERENCE','REZE REFERENCE']
        for index,(image,name) in enumerate(zip(images,names)):
            x=250+index*570
            contain(canvas,image,(x,290,280,430))
            text(draw,(x-2,744),name,23,fill=ACCENTS[index],bold=True)
        stages=['Source video','ExMR / SMPL-X','Blender camera','H3 + reference','Generated video']
        current=min(4,int(frame/INTRO_FRAMES*5))
        for i,label in enumerate(stages):
            x=96+i*350
            draw.rounded_rectangle((x,866,x+318,958),radius=14,fill='#263347' if i!=current else '#34535d',
                                   outline=ACCENTS[0] if i==current else '#36445a',width=2)
            text(draw,(x+20,897),label,23,bold=True)
            if i<4:
                text(draw,(x+327,895),'>',26,fill=MUTED)
        text(draw,(98,1000),'Individual results first, then a synchronized three-way comparison.',25,fill=MUTED)
        return canvas

    def single_base(self,chapter):
        canvas=Image.new('RGB',(WIDTH,HEIGHT),BG);draw=ImageDraw.Draw(canvas)
        titles=['SHARED 3D GUIDE','PHOTO-CONDITIONED RESULT','REZE-CONDITIONED RESULT']
        text(draw,(64,28),f'{chapter+1:02d} / {titles[chapter]}',34,bold=True)
        text(draw,(1510,33),'768 x 1344 / 24 fps',23,fill=MUTED)
        draw.line((64,84,1856,84),fill='#344157',width=2)
        left_titles=['Keep the motion.\nChange the camera.',
                     'Use a photo.\nFollow the guide.',
                     'Use an anime reference.\nKeep the same guide.']
        y=wrapped(draw,(64,160),left_titles[chapter],560,39,WHITE,53,True)
        summaries=[
            'The existing 699-frame SMPL-X mesh animation is reused. Only the camera changes.',
            'The photograph supplies appearance. The shared render supplies motion, perspective and framing.',
            'The Reze illustration supplies appearance and style. Motion and camera input are unchanged.',
        ]
        y=wrapped(draw,(64,y+28),summaries[chapter],540,27,line_height=40)
        details=[
            ['50 mm focal length','Front arc: -24 to +26 degrees','Dolly, height change and subtle roll','No motion retiming'],
            ['Body image + cropped facial reference','Six overlapping generation windows','Five-frame continuation anchors','Visible-face correction after generation'],
            ['Same guide video and timestamps','Same six-window timeline','Anime reference retained','No photographic face correction'],
        ]
        for item in details[chapter]:
            y+=38
            draw.ellipse((67,y+10,75,y+18),fill=ACCENTS[chapter])
            y=wrapped(draw,(94,y),item,500,24,fill=WHITE,line_height=34)
        ref=[self.source,self.photo,self.reze][chapter]
        contain(canvas,ref,(1350,162,340,465))
        text(draw,(1280,660),['SOURCE VIDEO','photo reference','reze.webp'][chapter],25,fill=ACCENTS[chapter],bold=True)
        notes=[
            'The guide is an actual 3D scene.\nAll 699 mesh frames match the previous motion.',
            'The photograph also supplies the face reference.\nFace correction follows full-video assembly.',
            'The character appearance is generated as video.\nThis is not a textured 3D Reze asset.',
        ]
        wrapped(draw,(1280,710),notes[chapter],540,24,line_height=36)
        if chapter==1 and self.corrected is not None:
            wrapped(draw,(1280,902),f'{self.corrected} / 699 frames received visible-face correction.',530,23,ACCENTS[1],34)
        return canvas

    def single(self,base,chapter,frame,index):
        canvas=base.copy();draw=ImageDraw.Draw(canvas)
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        canvas.paste(image.resize((512,896),Image.Resampling.LANCZOS),(704,100))
        cam=self.camera[index]
        y=820
        text(draw,(64,y),'CAMERA / SAME TIMELINE',21,fill=ACCENTS[chapter],bold=True)
        text(draw,(64,y+42),f'Distance  {cam["distance_metres"]:.2f} m',29,bold=True)
        text(draw,(64,y+84),f'Arc  {cam["yaw"]:+.1f} deg     Roll  {cam["roll"]:+.1f} deg',25,fill=MUTED)
        text(draw,(64,y+125),f'Frame {index+1:03d} / 699     {index/24:05.2f} s',24,fill=MUTED)
        self.footer(canvas,caption(chapter,index/24),index,ACCENTS[chapter])
        return canvas

    def comparison(self,frames,index):
        canvas=Image.new('RGB',(WIDTH,HEIGHT),BG);draw=ImageDraw.Draw(canvas)
        text(draw,(64,12),'04 / SYNCHRONIZED COMPARISON',21,fill=MUTED,bold=True)
        names=['3D GUIDE','PHOTO RESULT','REZE RESULT']
        for i,(frame,name) in enumerate(zip(frames,names)):
            x=168+i*536
            text(draw,(x,48),name,29,fill=ACCENTS[i],bold=True)
            image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            canvas.paste(image.resize((512,896),Image.Resampling.LANCZOS),(x,92))
        cam=self.camera[index]
        text(draw,(168,999),f'Frame {index+1:03d} / 699   |   {index/24:05.2f} s   |   50 mm   |   Distance {cam["distance_metres"]:.2f} m   |   Arc {cam["yaw"]:+.1f} deg',23,fill=MUTED)
        text(draw,(168,1036),caption(3,index/24),23,fill=WHITE)
        return canvas

    def footer(self,canvas,line,index,accent):
        draw=ImageDraw.Draw(canvas)
        draw.rectangle((64,1006,1856,1009),fill='#2a364a')
        draw.rectangle((64,1006,64+int(1792*(index+1)/699),1009),fill=accent)
        text(draw,(64,1031),line,25,fill=WHITE)

    def outro(self):
        canvas=Image.new('RGB',(WIDTH,HEIGHT),BG);draw=ImageDraw.Draw(canvas)
        text(draw,(96,70),'WHAT STAYS FIXED. WHAT CHANGES.',48,bold=True)
        columns=[(96,'FIXED',ACCENTS[0],[
            'The same 699-frame 3D performance',
            'One 50 mm camera path',
            'The same guide frames and 24 fps timeline']),
            (1030,'CHANGED',ACCENTS[2],[
            'The appearance reference image',
            'Photo appearance versus anime appearance',
            'Photo face correction only'])]
        for x,title,accent,items in columns:
            text(draw,(x,250),title,31,fill=accent,bold=True)
            y=324
            for item in items:
                y=wrapped(draw,(x,y),item,780,31,WHITE,46)+36
        draw.line((96,710,1824,710),fill='#36445a',width=2)
        wrapped(draw,(96,764),'The gray guide is 3D geometry. The two character appearances are generated 2D videos, not textured 3D character assets.',1720,30,line_height=45)
        wrapped(draw,(96,880),'A shared guide makes comparison possible. Generated poses, movement timing, details and backgrounds can still differ.',1720,27,line_height=40)
        text(draw,(96,1000),'No generated-video upscaling  |  English captions  |  Source music  |  No narration',23,fill=ACCENTS[0])
        return canvas


def read_video(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot open '+str(path))
    if (int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=699
        or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))!=768
        or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))!=1344
        or abs(cap.get(cv2.CAP_PROP_FPS)-24)>.001):
        raise ValueError('Input timeline or dimensions differ: '+str(path))
    return cap


def render(args):
    job=Path(args.job);out=job/'outputs'
    design=Design(job,args.source_audio)
    if args.preview_only:
        directory=out/'layout-preview';directory.mkdir(exist_ok=True)
        design.intro(70).save(directory/'intro.png')
        sample=np.asarray(Image.open(out/'blender/previews/frame_0073.png').convert('RGB'))[:,:,::-1].copy()
        for i,name in enumerate(['guide','photo','reze']):
            preview=design.single(design.single_base(i),i,sample,72)
            ImageDraw.Draw(preview).text((710,130),'LAYOUT PREVIEW',font=font(26,True),fill='#db5967')
            preview.save(directory/(name+'.png'))
        preview=design.comparison([sample]*3,72)
        ImageDraw.Draw(preview).text((1500,12),'LAYOUT PREVIEW',font=font(21,True),fill='#db5967')
        preview.save(directory/'comparison.png')
        design.outro().save(directory/'outro.png')
        return
    paths=[Path(args.guide),Path(args.photo_video),Path(args.reze_video)]
    assert all(p.is_file() for p in paths)
    silent=out/'explainer-silent.mp4'
    command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','24','-i','pipe:0',
        '-an','-c:v','libx264','-crf','17','-preset','fast','-pix_fmt','yuv420p',
        '-threads','6','-movflags','+faststart',str(silent)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    written=0
    def emit(image):
        nonlocal written
        encoder.stdin.write(image.tobytes());written+=1
        if written%240==0:
            print(f'Explanation frames: {written}/{TOTAL_FRAMES}',flush=True)
    try:
        for index in range(INTRO_FRAMES):
            emit(design.intro(index))
        for chapter,path in enumerate(paths):
            base=design.single_base(chapter);cap=read_video(path)
            try:
                for index in range(MOTION_FRAMES):
                    ok,frame=cap.read();assert ok,(chapter,index)
                    emit(design.single(base,chapter,frame,index))
            finally:
                cap.release()
        captures=[read_video(path) for path in paths]
        try:
            for index in range(MOTION_FRAMES):
                images=[]
                for cap in captures:
                    ok,frame=cap.read();assert ok,index;images.append(frame)
                emit(design.comparison(images,index))
        finally:
            for cap in captures:cap.release()
        ending=design.outro()
        for _ in range(OUTRO_FRAMES):emit(ending)
        encoder.stdin.close()
        assert encoder.wait()==0
    finally:
        if encoder.poll() is None:
            encoder.terminate();encoder.wait(timeout=15)
    assert written==3060
    final=out/'3D_Photo_Reze_Dynamic_Camera_Explained.mp4'
    audio_filter=(
        '[1:a]aresample=48000,apad=whole_dur=29.125,atrim=duration=29.125,'
        'asetpts=PTS-STARTPTS,asplit=4[a0][a1][a2][a3];'
        '[a0][a1][a2][a3]concat=n=4:v=0:a=1,adelay=5000:all=1,'
        'apad=whole_dur=127.5,atrim=duration=127.5[a]')
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(silent),
        '-i',str(args.source_audio),'-filter_complex',audio_filter,'-map','0:v:0','-map','[a]',
        '-c:v','copy','-c:a','aac','-b:a','256k','-t','127.5','-movflags','+faststart',str(final)],check=True)
    chapters=[dict(name='Introduction',start_frame=0,frames=120)]
    for i,name in enumerate(['Shared 3D guide','Photo result','Reze result','Synchronized comparison']):
        chapters.append(dict(name=name,start_frame=120+i*699,frames=699,source_start_frame=0))
    chapters.append(dict(name='Technical summary',start_frame=2916,frames=144))
    (out/'chapters.json').write_text(json.dumps(chapters,indent=2))
    (out/'composition.json').write_text(json.dumps(dict(width=1920,height=1080,fps=24,
        frames=3060,duration=127.5,chapters=chapters,subtitles='English burned-in',narration=False,
        soundtrack='Original source music, repeated and aligned for each full-length chapter',
        panel_size=[512,896],input_size=[768,1344],input_upscaled=False,
        synchronized_comparison_source_frame_range=[0,698]),indent=2))
    print('EXPLANATION_COMPLETED',str(final),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    parser.add_argument('--guide')
    parser.add_argument('--photo-video')
    parser.add_argument('--reze-video')
    parser.add_argument('--source-audio',required=True)
    parser.add_argument('--preview-only',action='store_true')
    render(parser.parse_args())
