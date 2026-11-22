"""Full-length dance processing with a continuous motion/camera timeline."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import runner
from settings import EXMR, EXMR_PYTHON, FACE_MODELS, FACE_PYTHON, REQUIRED_MODELS, ROOT


def full_duration(path):
    data = json.loads(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)], text=True))
    video = next(s for s in data['streams'] if s.get('codec_type') == 'video')
    durations = [float(data['format'].get('duration') or 0)]
    durations += [float(s.get('duration') or 0) for s in data['streams']]
    duration = max(durations)
    frames = math.ceil(duration * 24 - 1e-6)
    if not 2 <= frames <= 1800:
        raise ValueError('전체 영상은 75초 이하여야 합니다.')
    return dict(duration=duration, video_duration=float(video['duration']),
                frames=frames, output_duration=frames/24)


def reconstruct_entire(video, report=print, *, width=768, height=1344, orbit=True,
                       static_cam=True,ground=False,yaw=0,elevation=8,zoom=1):
    info = runner.probe_video(video)
    timeline = full_duration(video)
    runner.require_models()
    with runner.managed_job('rgbmocap-entire-guide') as job:
        inputs = job/'outputs/inputs'
        inputs.mkdir()
        source = inputs/('source'+Path(video).suffix.lower())
        shutil.copy2(video, source)
        clip = inputs/'motion.mp4'
        env = runner.environment(job)
        # Cover the complete media, including the last audio packet. Only the
        # sub-frame tail is held; the choreography is never slowed or looped.
        runner.run(['ffmpeg','-nostdin','-v','error','-y','-i',source,
            '-map','0:v:0','-map','0:a:0?',
            '-vf','tpad=stop_mode=clone:stop_duration=1,fps=24,scale=1920:1920:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1',
            '-frames:v',timeline['frames'],'-t',timeline['output_duration'],
            '-c:v','libx264','-crf','18','-preset','fast','-c:a','aac',
            '-threads','4','-filter_threads','4',clip],job,env,'complete-source',report)
        checked = runner.probe_video(clip,count=True)
        if checked['frames'] != timeline['frames']:
            raise ValueError('원본 전체 구간 변환에서 프레임이 누락되었습니다.')
        settings = dict(input=info, start=0, frames=timeline['frames'], fps=24,
            static_cam=bool(static_cam), ground=bool(ground), width=width, height=height,
            yaw=yaw,elevation=elevation,zoom=zoom,orbit=orbit,motion_retimed=False,
            full_source=True,source_duration=timeline['duration'],
            tail_hold_seconds=timeline['output_duration']-timeline['video_duration'])
        runner.write_json(job/'outputs/settings.json',settings)
        with runner.gpu_stage(job,report) as gpu_env:
            mocap = job/'outputs/mocap'
            command=[EXMR_PYTHON,EXMR/'tools/demo.py','--video',clip,
                '--output_dir',mocap,'--checkpoint_dir',runner.CHECKPOINTS,
                '--export_fps','24','--no_render','--keep_frames']
            if static_cam:
                command.append('--static_cam')
            if ground:
                command.append('--snap_to_ground')
            runner.run(command,job,gpu_env,'exmr-entire',report)
            mesh = job/'outputs/animated_mesh.npz'
            runner.run([EXMR_PYTHON,ROOT/'mesh_export.py','--source',mocap/'results.npz',
                '--model',runner.CHECKPOINTS/REQUIRED_MODELS['SMPL-X 인체 모델'],
                '--output',mesh,'--frames',timeline['frames']],job,env,'mesh-export',report)
            runner.render(job,mesh,timeline['frames'],width,height,yaw,elevation,zoom,
                          gpu_env,report,clip,orbit)
        runner.write_json(job/'outputs/completed.json',dict(stage='full-3d-guide',settings=settings))
        return job


def segment_plan(frames, window=124, overlap=5):
    if not 2 <= frames <= 1800 or window != 124 or overlap != 5:
        raise ValueError('Unsupported full-video segment plan')
    result = []
    start = 0
    while start < frames:
        needed = min(window,frames-start)
        length = max(56,5+17*math.ceil((needed-5)/17))
        result.append(dict(index=len(result),start=start,frames=length,
                           valid_frames=min(length,frames-start),overlap=overlap if result else 0))
        if start+length >= frames:
            break
        start += length-overlap
    return result


def extract_window(source,output,start,frames,job,report,*,audio=True):
    command = ['ffmpeg','-nostdin','-v','error','-y','-ss',start/24,'-i',source,
               '-map','0:v:0']
    if audio:
        command += ['-map','0:a:0?']
    else:
        command += ['-an']
    command += ['-vf','tpad=stop_mode=clone:stop_duration=6,fps=24,setsar=1',
        '-frames:v',frames,'-t',frames/24,'-c:v','libx264','-crf','16','-preset','fast',
        '-c:a','aac','-threads','4','-filter_threads','4','-movflags','+faststart',output]
    runner.run(command,job,runner.environment(job),'segment-extract',report)
    checked = runner.probe_video(output,count=True)
    if checked['frames'] != frames or abs(checked['fps']-24) > .001:
        raise ValueError('Segment extraction lost frames')


def swap_entire(guide,character,prompt,seed,report=print,*,preserve_face=True,
                face_image=None,reference_detail=True,source_audio=None):
    info = runner.probe_video(guide,count=True)
    width,height,total = info['width'],info['height'],info['frames']
    if (abs(info['fps']-24) > .001 or width%32 or height%32
            or width*height > 1344*768):
        raise ValueError('전체 가이드는 24 fps·약 1 MP 이하·32픽셀 배수 해상도여야 합니다.')
    plan = segment_plan(total)
    audio_source = Path(source_audio or guide)
    original = runner.probe_video(audio_source)
    with runner.managed_job('rgbmocap-entire-swap') as job:
        report('전체 영상 작업 폴더: '+str(job))
        inputs = job/'outputs/inputs'
        inputs.mkdir()
        body = inputs/'character.png'
        runner.validate_image(character,body)
        face_source = body
        if face_image:
            face_source = inputs/'face-source.png'
            runner.validate_image(face_image,face_source)
        if preserve_face:
            runner.prepare_face(job,face_source,report,reference_detail=reference_detail)
        shutil.copy2(audio_source,inputs/'source-audio.mp4')
        manifest = dict(frames=total,width=width,height=height,fps=24,overlap=5,
                        segments=plan,source_audio=str(audio_source),motion_retimed=False)
        runner.write_json(job/'outputs/segments.json',manifest)
        # Reuse a leased backend across the sequence; native H3 keyframes carry
        # the previous five frames. The source and camera share one timeline.
        with runner.h3_session(report):
            previous = None
            for part in plan:
                index = part['index']
                report(f'전체 영상 캐릭터 생성 · 구간 {index+1}/{len(plan)}')
                clip = inputs/f'guide-{index:03}.mp4'
                extract_window(guide,clip,part['start'],part['frames'],job,report)
                anchor = None
                if previous:
                    anchor = inputs/f'anchor-{index:03}.mp4'
                    extract_window(previous['path'],anchor,previous['frames']-5,5,job,report,audio=False)
                runner.write_json(job/'outputs/progress.json',dict(stage='generation',
                    segment=index+1,segments=len(plan),source_start=part['start']/24))
                result = runner.swap(clip,body,prompt,seed,report,preserve_face=preserve_face,
                    face_image=face_source if face_image else None,reference_detail=reference_detail,
                    anchor_video=anchor,correct_face=False)
                part.update(path=str(result/'outputs/character-swap-raw.mp4'),job=str(result))
                previous = part
                runner.write_json(job/'outputs/segments.json',manifest)
        raw = job/'outputs/character-swap-raw.mp4'
        env = runner.environment(job)
        runner.run([FACE_PYTHON,ROOT/'merge_video_segments.py','--manifest',job/'outputs/segments.json',
                    '--output',raw],job,env,'merge-segments',report)
        result_source = raw
        face_report = None
        if preserve_face:
            runner.run([FACE_PYTHON,ROOT/'face_identity.py','correct','--models',FACE_MODELS,
                        '--image',face_source,'--video',raw,'--output',job/'outputs'],
                       job,env,'whole-face-correction',report)
            result_source = job/'outputs/face-corrected-silent.mp4'
            face_report = json.loads((job/'outputs/face-report.json').read_text())
        final = job/'outputs/character-swap.mp4'
        runner.run(['ffmpeg','-nostdin','-v','error','-y','-i',result_source,'-i',audio_source,
            '-map','0:v:0','-map','1:a:0?','-c:v','copy','-c:a','copy','-t',total/24,
            '-movflags','+faststart',final],job,env,'original-audio',report)
        checked = runner.probe_video(final,count=True)
        if checked['frames'] != total or checked['audio'] != original['audio']:
            raise ValueError('전체 영상 길이 또는 원본 오디오가 일치하지 않습니다.')
        runner.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',final,'-f','null','-'],
                   job,env,'full-video-decode',report)
        runner.write_json(job/'outputs/completed.json',dict(stage='full-h3-swap',output=checked,
            segment_count=len(plan),reference_detail=reference_detail,preserve_face=preserve_face,
            corrected_frames=face_report['corrected_frames'] if face_report else 0,
            motion_retimed=False,upscaled=False,source_audio=str(audio_source)))
        return job
