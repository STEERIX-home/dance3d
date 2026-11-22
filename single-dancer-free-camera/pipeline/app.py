"""Local Korean UI for ExMR -> Blender -> H3."""
import json
import os
import shutil
from pathlib import Path

import gradio as gr

import runner
import full_video
from settings import PORT, PROMPT, missing_models

UI_JOB = Path(os.environ['RGBMOCAP_UI_JOB'])
PUBLISHED = UI_JOB / 'outputs/published'
PUBLISHED.mkdir(parents=True, exist_ok=True)


def ready_status():
    missing = missing_models()
    if not missing:
        return '**모델 준비 완료** · 동작 영상과 캐릭터 이미지를 넣고 실행하세요.'
    return ('**3D 추출 모델 준비 대기** · ' + ', '.join(missing)
            + '\n\n아래 「모델 준비 안내」에서 필요한 파일과 승인 상태를 확인하세요. '
            '모델이 없으면 GPU 실행을 시작하지 않습니다.')


def preparation_guide():
    missing = set(missing_models())
    if not missing:
        return ('**필수 모델이 모두 설치되어 있습니다.** 추가 ZIP이나 접근 승인을 기다릴 필요가 없습니다.\n\n'
                '- SMPL-X: 설치 완료\n- MANO: 설치 완료\n- SAM 3D Body: 설치 완료\n\n'
                '동작 영상과 캐릭터 이미지를 넣고 **입력 파일 검사**부터 실행하세요.')
    smplx = '`models_smplx_v1_1.zip` 필요' if 'SMPL-X 인체 모델' in missing else '설치 완료'
    mano = '`mano_v1_2.zip` 필요' if 'MANO 손 모델' in missing else '설치 완료'
    sam = '접근 승인 후 모델 파일 설치 필요' if any(name.startswith('SAM 3D Body') for name in missing) else '설치 완료'
    return ('모델별 준비 상태입니다. 필요한 ZIP은 공식 사이트에서 내려받으세요.\n\n'
            f'- [SMPL-X](https://smpl-x.is.tue.mpg.de/): {smplx}\n'
            f'- [MANO](https://mano.is.tue.mpg.de/): {mano}\n'
            f'- [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3): {sam}\n\n'
            'ZIP은 바탕화면의 **3D 모캡 모델 넣는 곳**에 원본 그대로 넣으세요. '
            '모델 등록 시 파일 검사를 진행합니다.')


def publish(job):
    job = Path(job)
    destination = PUBLISHED / job.name
    destination.mkdir(exist_ok=True)
    files = []
    candidates = [*sorted(p for p in (job/'outputs').iterdir() if p.is_file()
                          and p.suffix in ('.mp4', '.png', '.jpg', '.blend', '.glb', '.bvh', '.txt')),
                  job/'outputs/blender/preview.glb', job/'outputs/blender/scene.blend',
                  *sorted((job/'outputs').glob('*_amass.npz')),
                  job/'outputs/mocap/results.npz', *sorted((job/'outputs/mocap').glob('*.bvh')),
                  *sorted((job/'outputs/mocap').glob('*_amass.npz'))]
    for source in candidates:
        if source.is_file():
            target = destination / source.name
            if not target.exists():
                shutil.copy2(source, target)
            files.append(str(target))
    return {Path(path).name: path for path in files}, files


def dimensions(aspect):
    return {'가로 16:9': (1344, 768), '세로 9:16': (768, 1344),
            '정사각형': (992, 992), '정사각형 768': (768, 768),
            '정사각형 768 · 기본': (768, 768)}[aspect]


def check_face(image, face_image, report, reference_detail=False):
    with runner.managed_job('rgbmocap-face-input-check') as job:
        normalized = job/'outputs/source.png'
        runner.validate_image(face_image or image, normalized)
        runner.prepare_face(job, normalized, report, reference_detail=reference_detail)
        data, _ = publish(job)
        return data['face-reference.png']


def create_guide(video, image, start, frames, fixed, ground, aspect, yaw, elevation, zoom,
                 text, seed, orbit, preserve_face, face_image, reference_detail, all_stages, progress=gr.Progress()):
    if not video:
        raise gr.Error('동작 영상을 먼저 선택해 주세요.')
    messages = []
    def report(message):
        progress(0, desc=message)
    try:
        with runner.exclusive():
            runner.validate_options(video, start, int(frames), *dimensions(aspect), yaw, elevation, zoom)
            face_preview = None
            if all_stages:
                runner.validate_image(image)
                if preserve_face:
                    face_preview = check_face(image, face_image, report, reference_detail)
            job = runner.reconstruct(video, start, int(frames), fixed, ground, *dimensions(aspect),
                                     yaw, elevation, zoom, report, orbit=orbit)
            by_name, files = publish(job)
            guide = by_name['guide.mp4']
            messages.append('3D 추출 · Blender 렌더 완료: ' + str(job))
            result = overview = comparison = None
            if all_stages:
                try:
                    method = full_video.swap_entire if int(frames)==0 else runner.swap
                    extras = {'source_audio':video} if int(frames)==0 else {}
                    swap_job = method(guide, image, text, int(seed), report,
                                           preserve_face=preserve_face, face_image=face_image,
                                           reference_detail=reference_detail,**extras)
                    final, final_files = publish(swap_job)
                    result = final['character-swap.mp4']
                    files += final_files
                    messages.append('캐릭터 교체 · 출력 파일 검사 완료: ' + str(swap_job))
                    face_preview = final.get('face-reference.png')
                    comparison = final.get('face-comparison.jpg')
                    bundle, destination = runner.make_bundle(swap_job, job, report)
                    outputs, bundle_files = publish(bundle)
                    files = bundle_files
                    overview = outputs['입력부터_결과까지.mp4']
                    messages.append('바탕화면 실제 파일 저장 완료: ' + str(destination))
                    if preserve_face:
                        face_data = json.loads((swap_job/'outputs/face-report.json').read_text())
                        messages.append(f'얼굴 보정: {face_data["corrected_frames"]}/{face_data["frames"]}프레임. '
                                        '얼굴이 보이지 않거나 감지가 불확실한 프레임은 보정하지 않았습니다.')
                except Exception as error:
                    messages.append('캐릭터 생성 또는 결과 저장 단계 실패: ' + str(error))
                    gr.Warning('3D 가이드는 보존했습니다. 아래 오류를 확인한 뒤 2단계를 다시 실행해 주세요.')
            return (guide, by_name.get('preview.glb'), files, str(job), '\n\n'.join(messages),
                    result, overview, comparison, face_preview)
    except Exception as error:
        raise gr.Error(str(error)) from error


def preview_camera(previous, yaw, elevation, zoom, orbit, progress=gr.Progress()):
    if not previous:
        raise gr.Error('먼저 1단계를 실행해 3D 모션을 만드세요.')
    try:
        with runner.exclusive():
            job = runner.rerender(previous, yaw, elevation, zoom, lambda text: progress(0, desc=text), orbit=orbit)
            data, files = publish(job)
            return data['guide.mp4'], data.get('preview.glb'), files, str(job), '카메라 변경 렌더 완료: '+str(job)
    except Exception as error:
        raise gr.Error(str(error)) from error


def swap_only(previous, external_guide, image, text, seed, preserve_face, face_image, reference_detail, progress=gr.Progress()):
    try:
        with runner.exclusive():
            if external_guide:
                guide = external_guide
            elif previous:
                guide = runner.trusted_job(previous) / 'outputs/guide.mp4'
            else:
                raise ValueError('먼저 3D 가이드를 만들거나 직접 렌더한 가이드 MP4를 선택해 주세요.')
            report = lambda text: progress(0, desc=text)
            info=runner.probe_video(guide,count=True)
            whole = info['frames']>226 or info['frames']%17!=5 or info['frames']<56
            source_audio = None
            if whole and previous and not external_guide:
                candidates=list((runner.trusted_job(previous)/'outputs/inputs').glob('source.*'))
                if candidates:
                    source_audio=candidates[0]
            method=full_video.swap_entire if whole else runner.swap
            extras={'source_audio':source_audio} if whole else {}
            job = method(guide, image, text, int(seed), report,
                         preserve_face=preserve_face,face_image=face_image,
                         reference_detail=reference_detail,**extras)
            data, files = publish(job)
            bundle, destination = runner.make_bundle(job, previous if not external_guide else None, report)
            outputs, files = publish(bundle)
            return (data['character-swap.mp4'], files, '완료 · 바탕화면 저장: '+str(destination),
                    outputs['입력부터_결과까지.mp4'], data.get('face-comparison.jpg'), data.get('face-reference.png'))
    except Exception as error:
        raise gr.Error(str(error)) from error


def inspect_inputs(video, image, start, frames, aspect, yaw, elevation, zoom, preserve_face, face_image, reference_detail=False):
    try:
        messages = []
        face_preview = None
        if video:
            info = runner.validate_options(video, start, int(frames), *dimensions(aspect), yaw, elevation, zoom)
            end=full_video.full_duration(video)['duration'] if int(frames)==0 else start+int(frames)/24
            messages.append(f'영상 통과: {info["width"]}×{info["height"]}, {info["fps"]:.2f} fps, '
                            f'{info["duration"]:.2f}초 · 선택 {start:.2f}~{end:.2f}초')
        if image:
            info = runner.validate_image(image)
            messages.append(f'이미지 통과: {info["width"]}×{info["height"]}, 정상 디코딩')
            if preserve_face:
                with runner.exclusive():
                    face_preview = check_face(image, face_image, lambda text: None, reference_detail)
                messages.append('얼굴 감지 통과: 참고할 얼굴을 아래에서 확인하세요.')
        if not messages:
            raise ValueError('검사할 동작 영상 또는 캐릭터 이미지를 선택해 주세요.')
        messages.append('검사는 파일 형식·크기·길이·디코딩을 확인합니다. 인체 추적 품질은 3D 결과에서 확인하세요.')
        return '\n'.join(messages), face_preview
    except Exception as error:
        return '입력 검사 실패: ' + str(error), None


def cancel():
    runner.CANCEL.set()
    return '이 UI의 현재 작업에 중지를 요청했습니다. 중간 결과는 작업 폴더에 보존됩니다.'


with gr.Blocks(title='3D 모캡 → H3 캐릭터 스왑', analytics_enabled=False,
               theme=gr.themes.Soft(primary_hue='indigo',font=['Arial','sans-serif'],font_mono=['monospace']),
               css='.gradio-container {max-width: 1320px !important;}') as demo:
    gr.Markdown('# 영상 + 이미지 → 캐릭터 춤\n원본 춤을 추출해 캐릭터를 입히고, '
                '360도 카메라 영상까지 만듭니다. **영상과 이미지를 넣고 전체 실행을 누르세요.**\n\n'
                '결과는 바탕화면 **영상과 이미지로 춤 만들기** 폴더에 자동 저장됩니다.')
    status = gr.Markdown(ready_status())
    refresh = gr.Button('모델 상태 새로고침', size='sm')
    with gr.Accordion('모델 준비 안내', open=bool(missing_models())):
        preparation = gr.Markdown(preparation_guide())
    with gr.Row():
        with gr.Column():
            video = gr.Video(label='1 · 동작 원본 영상', sources=['upload'])
            image = gr.Image(label='2 · 캐릭터 이미지', type='filepath', sources=['upload'])
            gr.Markdown('한 사람의 전신과 손·발이 잘 보이는 영상을 권장합니다. '
                        '여러 사람이 있으면 ExMR이 주요 인물을 추적합니다.')
            with gr.Row():
                start = gr.Number(label='시작 위치 (초)', value=0, minimum=0)
                frames = gr.Dropdown(label='출력 길이', choices=[('영상 끝까지 · 최대 75초',0),('2.33초 · 56프레임',56),('3.04초 · 73프레임',73),
                                      ('3.75초 · 90프레임',90),('5.17초 · 124프레임',124),
                                      ('9.42초 · 226프레임 · 정사각형 768 / 세로',226)], value=0)
            orbit = gr.Checkbox(label='춤추는 동안 카메라 360도 회전', value=True)
            preserve_face = gr.Checkbox(label='입력 얼굴 유지 · 얼굴 확대 참고 + 생성 후 보정', value=True)
            reference_detail = gr.Checkbox(label='참고사진 세부정보 우선 · 생성 시간이 늘어납니다', value=True)
            with gr.Accordion('얼굴 사진 추가 (선택)', open=False):
                face_image = gr.Image(label='같은 캐릭터의 선명한 얼굴 사진', type='filepath', sources=['upload'])
                gr.Markdown('추가하지 않으면 캐릭터 이미지에서 얼굴을 자동으로 찾습니다. '
                            '얼굴이 작거나 흐리면 선명한 사진을 추가하세요. 만화 캐릭터 등 얼굴 감지가 안 되는 이미지는 얼굴 보정을 끄세요.')
            with gr.Row():
                fixed = gr.Checkbox(label='원본 카메라가 고정되어 있음', value=True)
                ground = gr.Checkbox(label='평평한 바닥에 발 맞추기', value=False)
            aspect = gr.Radio(['정사각형 768','가로 16:9','세로 9:16','정사각형'],
                              label='가이드·결과 화면 비율', value='세로 9:16')
            with gr.Accordion('카메라 시작 각도와 확대', open=False):
                yaw = gr.Slider(-180,180,value=0,step=5,label='회전 각도')
                elevation = gr.Slider(-10,60,value=8,step=1,label='높이 각도')
                zoom = gr.Slider(.5,1.5,value=1,step=.05,label='확대 (1 = 전체 동작 자동 맞춤)')
                rerender = gr.Button('기존 3D 동작으로 카메라 다시 렌더')
            with gr.Accordion('H3 프롬프트와 기존 가이드', open=False):
                text = gr.Textbox(label='캐릭터 교체 지시', value=PROMPT, lines=5)
                seed = gr.Number(label='시드', value=42, precision=0, minimum=0)
                external = gr.Video(label='직접 렌더한 3D 가이드 (선택, 2단계 버튼에서 사용)', sources=['upload'])
                gr.Markdown('외부 가이드는 24 fps · 약 1 MP 이하 · 가로·세로 32의 배수여야 합니다. '
                            '긴 가이드는 구간을 나눠 끝까지 연결하며 최대 75초까지 처리합니다. 결과 영상을 확대하지 않습니다.')
            check = gr.Button('입력 파일 검사')
            full = gr.Button('전체 실행 · 춤 추출 → 캐릭터 생성 → 바탕화면 저장', variant='primary')
            with gr.Row():
                first = gr.Button('1단계 · 3D 가이드 만들기')
                second = gr.Button('2단계 · 캐릭터 교체')
            stop = gr.Button('현재 작업 중지', variant='stop')
        with gr.Column():
            guide_video = gr.Video(label='Blender 3D 가이드')
            with gr.Accordion('3D 첫 프레임 · 드래그해서 회전', open=False):
                preview = gr.Model3D(label='3D 인체 메시', height=360)
            face_preview = gr.Image(label='입력에서 찾은 얼굴', interactive=False, height=220)
            result_video = gr.Video(label='최종 캐릭터 춤')
            comparison = gr.Image(label='입력 얼굴 / 생성 얼굴 / 보정 후 비교', interactive=False)
            overview = gr.Video(label='입력부터 결과까지 · 통합 영상')
            gr.Markdown('뒷모습과 가려진 부분은 생성 모델이 추정합니다. 얼굴 보정은 얼굴이 감지된 프레임에 적용됩니다.')
            log = gr.Textbox(label='검사·실행 상태', lines=7, interactive=False)
            files = gr.Files(label='결과 다운로드 · Blender 장면 / BVH / AMASS / 영상')
    current = gr.State('')
    all_flag, first_flag = gr.State(True), gr.State(False)
    common_inputs = [video,image,start,frames,fixed,ground,aspect,yaw,elevation,zoom,text,seed,orbit,preserve_face,face_image,reference_detail]
    stage_outputs = [guide_video,preview,files,current,log,result_video,overview,comparison,face_preview]
    full.click(create_guide,common_inputs+[all_flag],stage_outputs,concurrency_id='pipeline',concurrency_limit=1,api_name='run_dance')
    first.click(create_guide,common_inputs+[first_flag],stage_outputs,concurrency_id='pipeline',concurrency_limit=1,api_name='make_guide')
    rerender.click(preview_camera,[current,yaw,elevation,zoom,orbit],[guide_video,preview,files,current,log],concurrency_id='pipeline',concurrency_limit=1)
    second.click(swap_only,[current,external,image,text,seed,preserve_face,face_image,reference_detail],
                 [result_video,files,log,overview,comparison,face_preview],concurrency_id='pipeline',concurrency_limit=1)
    check.click(inspect_inputs,[video,image,start,frames,aspect,yaw,elevation,zoom,preserve_face,face_image,reference_detail],
                [log,face_preview],api_name='check_inputs')
    refresh.click(lambda: (ready_status(), preparation_guide()), outputs=[status, preparation], queue=False)
    demo.load(lambda: (ready_status(), preparation_guide()), outputs=[status, preparation], queue=False)
    stop.click(cancel,outputs=log,queue=False)

if __name__ == '__main__':
    demo.queue(max_size=2).launch(server_name='127.0.0.1',server_port=PORT,share=False,
        allowed_paths=[str(PUBLISHED)],max_file_size='1gb',show_error=True,inbrowser=False)
