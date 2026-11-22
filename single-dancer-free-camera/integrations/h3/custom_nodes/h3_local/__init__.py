import hashlib
import json
import math
import os
import subprocess
import urllib.request
from pathlib import Path

from aiohttp import web
from comfy_api.latest import ComfyExtension, InputImpl, io, ui
import folder_paths
from server import PromptServer

ROOT=Path(os.environ.get('H3_ROOT', str(Path(__file__).resolve().parents[3])))
MODEL_INSTALL=Path(os.environ.get('H3_MODEL_INSTALL', '/srv/training/jobs/SET_H3_MODEL_INSTALL'))
WEB_DIRECTORY='./web'


def source_path(file):
    path=Path(folder_paths.get_annotated_filepath(file)).resolve()
    path.relative_to(Path(folder_paths.get_input_directory()).resolve())
    if not path.is_file() or not 0<path.stat().st_size<=1024**3:
        raise ValueError('가이드 영상은 비어 있지 않은 1 GiB 이하 파일이어야 합니다.')
    return path


def probe(path):
    result=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],capture_output=True,text=True,timeout=30)
    if result.returncode:
        raise ValueError('영상을 읽을 수 없습니다. MP4/MOV/WebM 파일이 손상되지 않았는지 확인해 주세요.')
    metadata=json.loads(result.stdout)
    streams=[s for s in metadata['streams'] if s.get('codec_type')=='video']
    if not streams:
        raise ValueError('영상 트랙이 없습니다.')
    video=streams[0]
    duration=float(video.get('duration') or metadata['format'].get('duration') or 0)
    if not math.isfinite(duration) or duration<=0:
        raise ValueError('영상 길이를 읽을 수 없습니다.')
    return {'width':video['width'],'height':video['height'],'duration':duration,'fps':video.get('avg_frame_rate','0/1'),
            'audio':any(s.get('codec_type')=='audio' for s in metadata['streams'])}


def disk_preflight():
    with urllib.request.urlopen(os.environ.get('GPU_DASHBOARD_URL','http://127.0.0.1:8080').rstrip('/')+'/api/v1/status',timeout=10) as response:
        disk=json.load(response)['host']['disk']
    if disk['free_gib']-20<disk['reserve_target_gib']:
        raise ValueError('생성 예상 공간 20 GiB와 여유 공간 5%를 확보할 수 없습니다.')


class H3PrepareGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        folder=Path(folder_paths.get_input_directory())
        files=[p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4','.mov','.mkv','.webm','.avi','.m4v')]
        return io.Schema(node_id='H3PrepareGuide',display_name='H3 · 가이드 영상 검사 / 24 fps 변환',
            category='H3 Local',has_intermediate_output=True,
            inputs=[io.Combo.Input('file',options=sorted(files),upload=io.UploadType.video),
                    io.Float.Input('start_seconds',default=0,min=0,max=86400,step=.01),
                    io.Int.Input('length',default=90,min=5,max=3600,step=17),
                    io.Int.Input('width',default=768,min=32,max=4096,step=32),
                    io.Int.Input('height',default=1344,min=32,max=4096,step=32)],
            outputs=[io.Video.Output(),io.String.Output(display_name='검사 결과')])

    @classmethod
    def validate_inputs(cls,file):
        try:
            probe(source_path(file))
        except (ValueError,OSError,subprocess.TimeoutExpired) as error:
            return str(error)
        return True

    @classmethod
    def execute(cls,file,start_seconds,length,width,height):
        disk_preflight()
        original=source_path(file)
        info=probe(original)
        if length%17!=5:
            raise ValueError('H3 프레임 수는 17n+5여야 합니다. 기본값 90프레임을 사용해 주세요.')
        if start_seconds+length/24>info['duration']+1/24:
            raise ValueError(f"영상이 너무 짧습니다. 시작 {start_seconds:.2f}초부터 {length/24:.2f}초가 필요합니다.")
        if width%32 or height%32:
            raise ValueError('H3 출력 가로·세로는 32의 배수이어야 합니다.')
        identity=f'{original}:{original.stat().st_mtime_ns}:{start_seconds}:{length}:{width}:{height}'
        filename='h3-guide-'+hashlib.sha256(identity.encode()).hexdigest()[:16]+'.mp4'
        target=Path(folder_paths.get_temp_directory())/filename
        vf=f'fps=24,scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1'
        command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-ss',str(start_seconds),'-i',str(original),
                 '-map','0:v:0','-map','0:a:0?','-vf',vf,'-frames:v',str(length),'-t',str(length/24),
                 '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-ar','48000',
                 '-threads','4','-filter_threads','4','-movflags','+faststart',str(target)]
        run=subprocess.run(command,capture_output=True,text=True,timeout=600)
        if run.returncode:
            raise ValueError('가이드 영상 변환에 실패했습니다: '+run.stderr[-500:])
        video=InputImpl.VideoFromFile(str(target))
        if video.get_frame_count()!=length:
            raise ValueError('변환 영상 프레임 수가 요청과 다릅니다. 손상 구간이나 시작 위치를 확인해 주세요.')
        report=f"검사 통과 · {info['width']}×{info['height']} / {info['fps']} fps → {width}×{height} / 24 fps / {length}프레임 ({length/24:.2f}초)"
        record={'source':str(original),'input':info,'frames':length,'width':width,'height':height,'fps':24,'start_seconds':start_seconds}
        Path(os.environ['H3_JOB_DIR'],'outputs',filename+'.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
        return io.NodeOutput(video,report,ui=ui.PreviewVideo([ui.SavedResult(filename,'',io.FolderType.temp)]))

    @classmethod
    def fingerprint_inputs(cls,file,**kwargs):
        return source_path(file).stat().st_mtime_ns


class LocalExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3PrepareGuide]


async def comfy_entrypoint():
    return LocalExtension()


@PromptServer.instance.routes.get('/h3-local/workflow')
async def workflow(request):
    name='H3_Character_Swap_5090.json'
    return web.json_response(json.loads((ROOT/'workflows'/name).read_text()))


@PromptServer.instance.routes.get('/h3-local/status')
async def status(request):
    manifest=MODEL_INSTALL/'outputs/models.json'
    data=json.loads(manifest.read_text()) if manifest.exists() else {}
    ready=len(data)==6 and all(v.get('complete') and Path(v['path']).is_file() and Path(v['path']).stat().st_size==v['size_bytes'] for v in data.values())
    return web.json_response({'models_ready':ready,'mode':'gpu' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'setup'})
