"""360-degree and dynamic-camera explainer without displaying source footage."""
import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import dynamic_explainer as base
from dynamic_explainer import BG, WHITE, MUTED, ACCENTS, font, text, wrapped, contain

FPS, MOTION, INTRO, VIEWPOINTS, OUTRO = 24, 699, 120, 240, 192
TOTAL = INTRO+VIEWPOINTS+5*MOTION+OUTRO
KOREAN_FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def source_card():
    canvas=Image.new('RGB',(432,756),'#1b2638');draw=ImageDraw.Draw(canvas)
    draw.rounded_rectangle((140,90,292,202),radius=18,outline=ACCENTS[0],width=5)
    draw.polygon([(198,119),(198,174),(241,146)],fill=ACCENTS[0])
    def centered(y,value,face,fill=WHITE):
        width=draw.textlength(value,font=face)
        draw.text(((432-width)/2,y),value,font=face,fill=fill)
    centered(264,'ILLIT',font(52,True))
    centered(348,'아일릿 체포하겠어',ImageFont.truetype(KOREAN_FONT,32))
    centered(400,'댄스 챌린지 영상',ImageFont.truetype(KOREAN_FONT,32))
    centered(510,'DANCE CHALLENGE',font(27,True),ACCENTS[0])
    centered(564,'Motion reference',font(25),MUTED)
    return canvas


ORBIT_CAPTIONS=[
    (0,5,'One reconstructed 3D performance, viewed through a complete camera orbit.'),
    (5,10,'Move the camera around the body to reveal side and rear views.'),
    (10,15,'Render a new guide from the chosen camera path. Reuse the same motion.'),
    (15,20,'The same photograph conditions the 360-degree character result.'),
    (20,25,'Viewpoint control happens in 3D; H3 generates the final appearance.'),
    (25,29.125,'Next: move closer, pull back and change height with a second camera path.'),
]


class Design(base.Design):
    def __init__(self,job):
        # Source footage is deliberately not opened or passed into this design.
        self.job=job
        self.plan=json.loads((job/'outputs/edit-plan.json').read_text())
        self.camera=json.loads((job/'outputs/blender/camera-path.json').read_text())
        self.orbit_camera=json.loads((job/'outputs/orbit-camera-path.json').read_text())
        self.validation=json.loads((job/'outputs/blender/camera-validation.json').read_text())
        self.photo=Image.open(job/'outputs/inputs/character-photo.png').convert('RGB')
        self.reze=Image.open(job/'outputs/inputs/reze.png').convert('RGB')
        self.source=source_card()
        self.views=json.loads((job/'outputs/viewpoints/viewpoints.json').read_text())
        self.view_images=[Image.open(v['image']).convert('RGB') for v in self.views]
        stage=json.loads((job/'outputs/photo-stage.json').read_text())
        self.corrected=json.loads((Path(stage['job'])/'outputs/face-report.json').read_text())['corrected_frames']

    def intro(self,index):
        canvas=Image.new('RGB',(1920,1080),BG);draw=ImageDraw.Draw(canvas)
        text(draw,(96,50),'ONE PERFORMANCE. FREE CAMERA.',54,bold=True)
        text(draw,(96,124),'360-degree orbit. Dynamic shots. Two appearances.',38,ACCENTS[0],True)
        text(draw,(98,192),'Reconstruct the motion once. Choose where the virtual camera goes.',29,MUTED)
        for i,(image,label) in enumerate(zip([self.source,self.photo,self.reze],
                ['MOTION REFERENCE','PHOTO REFERENCE','REZE REFERENCE'])):
            x=250+i*570
            contain(canvas,image,(x,290,280,430))
            text(draw,(x-2,744),label,23,ACCENTS[i],True)
        stages=['Motion input','3D performance','Choose camera','H3 + reference','Generated video']
        selected=min(4,int(index/INTRO*5))
        for i,label in enumerate(stages):
            x=96+i*350
            draw.rounded_rectangle((x,866,x+318,958),radius=14,
                fill='#34535d' if i==selected else '#263347',
                outline=ACCENTS[0] if i==selected else '#36445a',width=2)
            text(draw,(x+18,897),label,23,bold=True)
            if i<4:text(draw,(x+327,895),'>',26,MUTED)
        text(draw,(98,1000),'Free viewpoints, a full 360-degree result, individual dynamic shots, then a three-way comparison.',25,MUTED)
        return canvas

    def viewpoints(self,index):
        canvas=Image.new('RGB',(1920,1080),BG);draw=ImageDraw.Draw(canvas)
        text(draw,(48,28),'CHOOSE THE CAMERA POSITION.',43,bold=True)
        text(draw,(48,91),'One frozen 3D pose. Four actual camera positions. No new motion capture.',27,MUTED)
        selected=min(3,index//48)
        for i,(view,image) in enumerate(zip(self.views,self.view_images)):
            x=48+464*i
            canvas.paste(image,(x,156))
            draw.rectangle((x-3,153,x+434,914),outline=ACCENTS[0] if i==selected or index>=192 else '#344157',width=3)
            text(draw,(x,930),view['label'].upper(),30,ACCENTS[0],True)
            text(draw,(x,978),f'Yaw {view["yaw"]:+.0f}°   Elevation {view["elevation"]:.0f}°',22,MUTED)
        text(draw,(48,1034),'Same mesh pose at 3.00 s. Only the virtual camera position changes.',27,WHITE)
        return canvas

    def orbit(self,frames,index):
        canvas=Image.new('RGB',(1920,1080),BG);draw=ImageDraw.Draw(canvas)
        text(draw,(64,17),'01 / FULL 360-DEGREE CAMERA ORBIT',32,bold=True)
        for i,(frame,label) in enumerate(zip(frames,['3D ORBIT GUIDE','PHOTO / 360-DEGREE RESULT'])):
            x=64+544*i
            text(draw,(x,67),label,25,ACCENTS[i],True)
            image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            canvas.paste(image.resize((512,896),Image.Resampling.LANCZOS),(x,104))
        wrapped(draw,(1192,130),'Move around\nthe performance.',650,40,WHITE,55,True)
        cx,cy,radius=1515,400,129
        draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline='#425269',width=3)
        yaw=float(self.orbit_camera[index]['yaw'])
        points=[(cx+math.sin(math.radians(t))*radius,cy+math.cos(math.radians(t))*radius)
                for t in np.linspace(0,yaw,max(2,int(yaw)+1))]
        draw.line(points,fill=ACCENTS[0],width=5)
        px,py=points[-1]
        draw.line((cx,cy,px,py),fill='#63758c',width=2)
        draw.ellipse((cx-15,cy-15,cx+15,cy+15),fill=WHITE)
        draw.ellipse((px-11,py-11,px+11,py+11),fill=ACCENTS[0])
        text(draw,(cx-36,cy-radius-43),'REAR',19,MUTED)
        text(draw,(cx-42,cy+radius+19),'FRONT',19,MUTED)
        text(draw,(1192,606),f'Camera orbit  {yaw:06.1f} / 360 deg',30,ACCENTS[0],True)
        wrapped(draw,(1192,668),'The virtual camera travels around the same reconstructed 3D performance.',640,28,MUTED,42)
        wrapped(draw,(1192,805),'Choose the view in 3D, then generate the character appearance from that guide.',640,27,WHITE,40)
        text(draw,(1192,952),f'Frame {index+1:03d} / 699   |   {index/24:05.2f} s',25,MUTED)
        line=next(line for start,end,line in ORBIT_CAPTIONS if start<=index/24<end)
        text(draw,(64,1034),line,26,WHITE)
        return canvas

    def single_base(self,chapter):
        canvas=super().single_base(chapter);draw=ImageDraw.Draw(canvas)
        draw.rectangle((60,15,1470,78),fill=BG)
        titles=['DYNAMIC CAMERA / 3D GUIDE','DYNAMIC CAMERA / PHOTO','DYNAMIC CAMERA / REZE']
        text(draw,(64,28),f'{chapter+2:02d} / {titles[chapter]}',34,bold=True)
        if chapter==0:
            draw.rectangle((1276,654,1830,700),fill=BG)
            text(draw,(1280,660),'MOTION REFERENCE',25,ACCENTS[0],True)
        return canvas

    def comparison(self,frames,index):
        canvas=super().comparison(frames,index);draw=ImageDraw.Draw(canvas)
        draw.rectangle((60,7,1200,43),fill=BG)
        text(draw,(64,12),'05 / SAME DYNAMIC GUIDE / THREE-WAY COMPARISON',21,MUTED,True)
        return canvas

    def outro(self):
        canvas=Image.new('RGB',(1920,1080),BG);draw=ImageDraw.Draw(canvas)
        text(draw,(96,66),'THE ADVANTAGE: CHOOSE THE VIEW.',49,bold=True)
        columns=[(96,'REUSE THE PERFORMANCE',ACCENTS[0],[
            'One reconstructed 3D motion sequence',
            'The same 699-frame choreography',
            'No new motion extraction for each camera']),
            (1030,'DESIGN THE CAMERA',ACCENTS[2],[
            'A full 360-degree orbit',
            'Dolly in, pull back, change height and roll',
            'Front, side, rear and elevated viewpoints'])]
        for x,title,accent,items in columns:
            text(draw,(x,245),title,29,accent,True);y=320
            for item in items:y=wrapped(draw,(x,y),item,780,30,WHITE,44)+34
        draw.line((96,700,1824,700),fill='#36445a',width=2)
        wrapped(draw,(96,752),'Camera placement is editable in the 3D scene. H3 turns each rendered guide into a character video.',1720,30,WHITE,45)
        wrapped(draw,(96,859),'Character appearances are generated 2D video. Poses, movement timing, details and unseen views can differ from the guide.',1720,27,MUTED,40)
        text(draw,(96,1000),'360-degree result + dynamic shots  |  English explanations  |  Source music  |  No narration',23,ACCENTS[0])
        return canvas


def chapters(plan):
    result=[];start=0
    def add(name,frames,panels=None):
        nonlocal start
        result.append(dict(name=name,start_frame=start,frames=frames,panels=panels or []));start+=frames
    def panel(path,x,y):return dict(path=path,x=x,y=y,width=512,height=896)
    add('Introduction',INTRO)
    add('Four free camera positions',VIEWPOINTS)
    add('Full 360-degree orbit',MOTION,[panel(plan['orbit_guide'],64,104),panel(plan['orbit_photo'],608,104)])
    for label,key in [('Dynamic 3D guide','dynamic_guide'),('Dynamic photo result','dynamic_photo'),('Dynamic Reze result','dynamic_reze')]:
        add(label,MOTION,[panel(plan[key],704,100)])
    add('Dynamic three-way comparison',MOTION,[panel(plan[key],168+i*536,92)
        for i,key in enumerate(['dynamic_guide','dynamic_photo','dynamic_reze'])])
    add('Technical summary',OUTRO)
    assert start==TOTAL==4047
    return result


def render(args):
    job=Path(args.job);out=job/'outputs';design=Design(job);plan=design.plan
    base.CAPTIONS[0][-1]=(24,29.125,'Return to the front. A second camera path, using the same 3D motion.')
    timeline=chapters(plan)
    (out/'chapters.json').write_text(json.dumps(timeline,indent=2))
    if args.preview_only:
        directory=out/'layout-preview';directory.mkdir(exist_ok=True)
        design.intro(70).save(directory/'intro.png')
        design.viewpoints(210).save(directory/'viewpoints.png')
        captures=[base.read_video(plan[k]) for k in ['orbit_guide','orbit_photo']]
        try:
            for index in [0,175,349,523,698]:
                frames=[]
                for cap in captures:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();assert ok;frames.append(frame)
                design.orbit(frames,index).save(directory/f'orbit-{index}.png')
        finally:
            for cap in captures:cap.release()
        design.single(design.single_base(0),0,base.first_frame(plan['dynamic_guide']),0).save(directory/'dynamic-guide.png')
        design.outro().save(directory/'outro.png')
        return
    silent=out/'free-camera-silent.mp4'
    encoder=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','24','-i','pipe:0',
        '-an','-c:v','libx264','-crf','17','-preset','fast','-pix_fmt','yuv420p','-threads','6',
        '-movflags','+faststart',str(silent)],stdin=subprocess.PIPE)
    written=0
    def emit(image):
        nonlocal written
        encoder.stdin.write(image.tobytes());written+=1
        if written%240==0:print(f'Export frames: {written}/{TOTAL}',flush=True)
    try:
        for i in range(INTRO):emit(design.intro(i))
        for i in range(VIEWPOINTS):emit(design.viewpoints(i))
        for chapter in timeline[2:7]:
            captures=[base.read_video(p['path']) for p in chapter['panels']]
            single_index=timeline.index(chapter)-3
            single_base=design.single_base(single_index) if len(captures)==1 else None
            try:
                for index in range(MOTION):
                    frames=[]
                    for cap in captures:
                        ok,frame=cap.read();assert ok;frames.append(frame)
                    if len(frames)==2:canvas=design.orbit(frames,index)
                    elif len(frames)==3:canvas=design.comparison(frames,index)
                    else:canvas=design.single(single_base,single_index,frames[0],index)
                    emit(canvas)
            finally:
                for cap in captures:cap.release()
        ending=design.outro()
        for _ in range(OUTRO):emit(ending)
        encoder.stdin.close();assert encoder.wait()==0
    finally:
        if encoder.poll() is None:encoder.terminate();encoder.wait(timeout=15)
    assert written==TOTAL
    audio_filter=('[1:a]aresample=48000,apad=whole_dur=29.125,atrim=duration=29.125,'
        'asetpts=PTS-STARTPTS,asplit=5[a0][a1][a2][a3][a4];'
        '[a0][a1][a2][a3][a4]concat=n=5:v=0:a=1,adelay=15000:all=1,'
        f'apad=whole_dur={TOTAL/FPS},atrim=duration={TOTAL/FPS}[a]')
    final=out/'3D_Photo_Reze_360_Free_Camera_Explained.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(silent),'-i',plan['music'],
        '-filter_complex',audio_filter,'-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac',
        '-b:a','256k','-t',str(TOTAL/FPS),'-map_metadata','-1','-map_chapters','-1',
        '-metadata','title=3D Camera: 360 Orbit and Dynamic Shots','-movflags','+faststart',str(final)],check=True)
    (out/'composition.json').write_text(json.dumps(dict(width=1920,height=1080,fps=24,frames=TOTAL,
        duration=TOTAL/FPS,chapters=timeline,source_footage_displayed=False,source_url_displayed=False,
        source_reference_label=plan['source_label'],narration=False,technical_captions='English',
        source_label_language='Korean, as requested',input_upscaled=False,full_orbit_frames=699,
        frozen_pose_views=4,soundtrack='Existing source music repeated and aligned to five full dance chapters'),ensure_ascii=False,indent=2))
    print('FREE_CAMERA_EXPORT_COMPLETE',str(final),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    parser.add_argument('--preview-only',action='store_true')
    render(parser.parse_args())
