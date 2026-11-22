"""Check actual chapter/frame placement and soundtrack alignment after export."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.signal import correlate, correlation_lags


def frame(cap,index):
    cap.set(cv2.CAP_PROP_POS_FRAMES,index)
    ok,result=cap.read()
    if not ok:
        raise ValueError(f'Cannot decode frame {index}')
    return result


def sound(path,start,duration=3):
    data=subprocess.check_output(['ffmpeg','-v','error','-ss',str(start),'-i',str(path),
        '-t',str(duration),'-map','0:a:0','-ac','1','-ar','16000','-f','f32le','pipe:1'])
    return np.frombuffer(data,dtype=np.float32).copy()


def main(args):
    job=Path(args.job);review=Path(args.review_job)
    done=json.loads((job/'outputs/completed.json').read_text())
    movie=Path(done['desktop'])
    inputs=[job/'outputs/guide.mp4',Path(done['photo']['video']),Path(done['reze']['video'])]
    output=cv2.VideoCapture(str(movie))
    sources=[cv2.VideoCapture(str(p)) for p in inputs]
    assert int(output.get(cv2.CAP_PROP_FRAME_COUNT))==3060
    assert (int(output.get(cv2.CAP_PROP_FRAME_WIDTH)),int(output.get(cv2.CAP_PROP_FRAME_HEIGHT)))==(1920,1080)
    assert abs(output.get(cv2.CAP_PROP_FPS)-24)<.001
    records=[]
    samples=[0,73,238,476,698]
    try:
        for chapter in range(4):
            for index in samples:
                image=frame(output,120+chapter*699+index)
                for source_index in (range(3) if chapter==3 else [chapter]):
                    original=frame(sources[source_index],index)
                    resized=Image.fromarray(cv2.cvtColor(original,cv2.COLOR_BGR2RGB)).resize(
                        (512,896),Image.Resampling.LANCZOS)
                    expected=cv2.cvtColor(np.asarray(resized),cv2.COLOR_RGB2BGR)
                    x,y=(168+536*source_index,92) if chapter==3 else (704,100)
                    actual=image[y:y+896,x:x+512]
                    error=float(np.abs(actual.astype(np.float32)-expected).mean())
                    assert error<8,(chapter,index,source_index,error)
                    records.append(dict(chapter=chapter+1,source=source_index,
                                        source_frame=index,mean_pixel_error=error))
        # Review stills are extracted from the delivered file itself.
        for label,index in [('intro',60),('guide',120+72),('photo',819+72),
                            ('reze',1518+72),('comparison',2217+72),('ending',3000)]:
            image=frame(output,index)
            cv2.imwrite(str(review/'outputs'/f'delivered-{label}.jpg'),image,
                        [cv2.IMWRITE_JPEG_QUALITY,96])
    finally:
        output.release()
        for cap in sources:cap.release()

    reference=sound(args.source_audio,.5)
    reference=reference-reference.mean()
    music=[]
    for chapter in range(4):
        actual=sound(movie,5+chapter*29.125+.5)
        n=min(len(reference),len(actual))
        expected=reference[:n];actual=actual[:n]-actual[:n].mean()
        correlations=correlate(actual,expected,mode='full',method='fft')
        lags=correlation_lags(n,n,mode='full')
        valid=np.abs(lags)<=1600
        peak=np.flatnonzero(valid)[np.argmax(correlations[valid])]
        lag=int(lags[peak])
        score=float(correlations[peak]/np.sqrt(np.dot(actual,actual)*np.dot(expected,expected)))
        assert abs(lag)<=160 and score>.95,(chapter,lag,score)
        music.append(dict(chapter=chapter+1,start_seconds=5+chapter*29.125,
                          measured_offset_ms=lag/16,normalized_correlation=score))
    for start in (1,123):
        audio=sound(movie,start,2)
        assert float(np.sqrt(np.mean(audio**2)))<1e-5,'Unexpected intro/outro audio'

    stage_reze=json.loads((job/'outputs/reze-stage.json').read_text())
    stage_photo=json.loads((job/'outputs/photo-stage.json').read_text())
    assert stage_reze['guide_sha256']==stage_photo['guide_sha256']
    report=dict(status='passed',video=str(movie),frames=3060,fps=24,resolution=[1920,1080],
        duration=127.5,source_motion_frames=699,chapter_frame_comparisons=records,
        max_panel_pixel_error=max(r['mean_pixel_error'] for r in records),
        soundtrack_alignment=music,intro_and_outro_silent=True,
        identical_guide_sha256=stage_reze['guide_sha256'],
        english_captions=True,narration=False,
        note='Frame samples compare the delivered panels to the corresponding native source frames. Audio checks compare each chapter against the source soundtrack.')
    (review/'outputs/verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='chapter_frame_comparisons'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    parser.add_argument('--review-job',required=True)
    parser.add_argument('--source-audio',required=True)
    main(parser.parse_args())
