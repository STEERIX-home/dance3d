"""Verify orbit inclusion, source-footage omission and delivered timeline."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.signal import correlate,correlation_lags

from free_camera_explainer import Design,TOTAL,FPS
from verify_dynamic_explainer import frame,sound


def error(actual,expected):
    return float(np.abs(actual.astype(np.float32)-expected.astype(np.float32)).mean())


def main(args):
    job=Path(args.job);movie=Path(args.video)
    plan=json.loads((job/'outputs/edit-plan.json').read_text())
    timeline=json.loads((job/'outputs/chapters.json').read_text())
    design=Design(job)
    output=cv2.VideoCapture(str(movie))
    assert int(output.get(cv2.CAP_PROP_FRAME_COUNT))==TOTAL==4047
    assert (int(output.get(cv2.CAP_PROP_FRAME_WIDTH)),int(output.get(cv2.CAP_PROP_FRAME_HEIGHT)))==(1920,1080)
    assert abs(output.get(cv2.CAP_PROP_FPS)-24)<.001
    records=[];private_cards=[];still_checks=[];captures={}
    expected_start=0
    for chapter in timeline:
        assert chapter['start_frame']==expected_start
        expected_start+=chapter['frames']
    assert expected_start==TOTAL
    try:
        for chapter in timeline:
            for panel in chapter['panels']:
                path=panel['path']
                if path not in captures:captures[path]=cv2.VideoCapture(path)
                for index in [0,175,349,523,698]:
                    actual=frame(output,chapter['start_frame']+index)
                    source=frame(captures[path],index)
                    expected=Image.fromarray(cv2.cvtColor(source,cv2.COLOR_BGR2RGB)).resize((512,896),Image.Resampling.LANCZOS)
                    expected=cv2.cvtColor(np.asarray(expected),cv2.COLOR_RGB2BGR)
                    x,y=panel['x'],panel['y']
                    measured=error(actual[y:y+896,x:x+512],expected)
                    assert measured<8,(chapter['name'],path,index,measured)
                    records.append(dict(chapter=chapter['name'],source=path,source_frame=index,mean_pixel_error=measured))
        # The only two places that previously showed source footage now contain
        # the text-only reference card. Check the start, middle and end of each.
        for start,indices,box,template in [
            (0,[0,60,119],(250,290,280,430),design.intro(60)),
            (timeline[3]['start_frame'],[0,349,698],(1350,162,340,465),design.single_base(0)),
        ]:
            expected=cv2.cvtColor(np.asarray(template),cv2.COLOR_RGB2BGR)
            x,y,w,h=box
            for index in indices:
                actual=frame(output,start+index)
                measured=error(actual[y:y+h,x:x+w],expected[y:y+h,x:x+w])
                assert measured<5,('Private reference card differs',start,index,measured)
                private_cards.append(dict(frame=start+index,mean_pixel_error=measured))
        actual=frame(output,120+210)
        for i,view in enumerate(design.views):
            expected=cv2.imread(view['image'])
            x=48+464*i
            measured=error(actual[156:912,x:x+432],expected)
            assert measured<5,(i,measured)
            assert view['frame']==73 and view['mesh_error']<1e-5
            still_checks.append(dict(view=view['label'],mean_pixel_error=measured,mesh_error=view['mesh_error']))
        examples=[('intro',60),('free-viewpoints',330),('orbit-front',360),
            ('orbit-side',360+175),('orbit-rear',360+349),('orbit-last',360+698),
            ('dynamic-guide',1059+72),('photo',1758+72),('reze',2457+72),
            ('comparison',3156+72),('ending',TOTAL-96)]
        for label,index in examples:
            cv2.imwrite(str(job/'outputs'/f'delivered-{label}.jpg'),frame(output,index),[cv2.IMWRITE_JPEG_QUALITY,96])
    finally:
        output.release()
        for cap in captures.values():cap.release()
    music=[]
    reference=sound(plan['music'],.5);reference-=reference.mean()
    for chapter in timeline:
        if not chapter['panels']:continue
        start=chapter['start_frame']/FPS
        actual=sound(movie,start+.5);actual-=actual.mean()
        n=min(len(reference),len(actual));a=actual[:n];b=reference[:n]
        correlations=correlate(a,b,mode='full',method='fft')
        lags=correlation_lags(n,n,mode='full');valid=np.abs(lags)<=1600
        peak=np.flatnonzero(valid)[np.argmax(correlations[valid])]
        lag=int(lags[peak]);score=float(correlations[peak]/np.sqrt(np.dot(a,a)*np.dot(b,b)))
        assert abs(lag)<=160 and score>.95,(chapter['name'],lag,score)
        music.append(dict(chapter=chapter['name'],offset_ms=lag/16,correlation=score))
    for start in (1,8,163):
        audio=sound(movie,start,2)
        assert float(np.sqrt(np.mean(audio**2)))<1e-5
    orbit=design.orbit_camera
    assert abs(orbit[0]['yaw'])<1e-5 and abs(orbit[-1]['yaw']-360)<1e-5
    assert all(b['yaw']>a['yaw'] for a,b in zip(orbit,orbit[1:]))
    old={'sha256':args.orbit_sha256}
    assert hashlib.sha256(Path(plan['orbit_photo']).read_bytes()).hexdigest()==old['sha256']
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(movie)],text=True))
    assert sorted(s['codec_type'] for s in meta['streams'])==['audio','video']
    tags=json.dumps(meta.get('format',{}).get('tags',{})).lower()
    assert not any(value in tags for value in ['youtube','youtu.be','http'])
    report=dict(status='passed',video=str(movie),frames=TOTAL,duration=TOTAL/FPS,resolution=[1920,1080],fps=24,
        full_360_included=True,full_360_frames=699,orbit_yaw=[orbit[0]['yaw'],orbit[-1]['yaw']],
        original_360_result_sha256=old['sha256'],video_panel_checks=records,source_reference_card_checks=private_cards,
        fixed_pose_view_checks=still_checks,soundtrack_alignment=music,
        source_footage_displayed=False,source_link_displayed=False,source_reference_label=plan['source_label'],
        english_technical_captions=True,narration=False,input_video_upscaled=False,
        known_limitations=['Generated poses and timing may differ from the 3D guide.','Generated appearances are 2D videos.'])
    (job/'outputs/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('video_panel_checks','source_reference_card_checks','fixed_pose_view_checks')},ensure_ascii=False,indent=2))
    print('Panel samples:',len(records),'Private reference cards:',len(private_cards),'Fixed-pose views:',len(still_checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--video',required=True)
    parser.add_argument('--orbit-sha256', required=True, help='Expected SHA-256 of the original 360-degree character result')
    main(parser.parse_args())
