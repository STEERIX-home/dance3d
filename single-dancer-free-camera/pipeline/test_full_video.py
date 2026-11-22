"""CPU checks for complete timeline coverage, overlap assembly and phone photos."""
import json
import os
import subprocess
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import full_video
import runner
from merge_video_segments import merge


class FullVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['RGBMOCAP_TEST_ROOT']).resolve()
        cls.root.relative_to(runner.JOBS.resolve())
        cls.root.mkdir(parents=True,exist_ok=True)

    def test_every_frame_including_final_frame_is_covered(self):
        for count in (2,55,56,57,123,124,125,240,699,1800):
            plan=full_video.segment_plan(count)
            indexes=[]
            for part in plan:
                self.assertEqual(part['frames']%17,5)
                self.assertGreaterEqual(part['frames'],56)
                self.assertLessEqual(part['frames'],124)
                begin=part['start']+part['overlap']
                indexes.extend(range(begin,part['start']+part['frames']))
            self.assertEqual(indexes[:count],list(range(count)))
        self.assertEqual([p['frames'] for p in full_video.segment_plan(699)],
                         [124,124,124,124,124,107])

    @unittest.skipUnless(os.environ.get('RGBMOCAP_MPO_TEST_IMAGE'), 'Set RGBMOCAP_MPO_TEST_IMAGE to an MPO phone photo')
    def test_phone_mpo_primary_image_is_preserved(self):
        source=Path(os.environ['RGBMOCAP_MPO_TEST_IMAGE'])
        target=self.root/'normalized.png'
        with Image.open(source) as im:
            self.assertEqual(im.format,'MPO')
            self.assertGreater(im.n_frames,1)
            im.seek(0)
            expected=np.array(ImageOps.exif_transpose(im).convert('RGB'))
        runner.validate_image(source,target)
        np.testing.assert_array_equal(expected,np.array(Image.open(target)))
        animation=self.root/'animated.webp'
        Image.new('RGB',(128,128),'red').save(animation,format='WEBP',save_all=True,
            append_images=[Image.new('RGB',(128,128),'blue')],duration=100,loop=0)
        with self.assertRaises(ValueError):
            runner.validate_image(animation)

    def test_real_video_merge_has_no_missing_repeated_or_shifted_frames(self):
        count=200
        rng=np.random.default_rng(73)
        frames=[np.repeat(np.repeat(rng.integers(0,256,(8,8,3),dtype=np.uint8),16,0),16,1)
                for _ in range(count)]
        plan=full_video.segment_plan(count)
        for part in plan:
            path=self.root/f'segment-{part["index"]}.mp4'
            command=['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24',
                '-s','128x128','-r','24','-i','pipe:0','-c:v','libx264rgb','-crf','0',
                '-preset','ultrafast','-threads','2',str(path)]
            payload=b''.join(frames[min(index,count-1)].tobytes()
                            for index in range(part['start'],part['start']+part['frames']))
            subprocess.run(command,input=payload,check=True)
            part['path']=str(path)
        manifest=self.root/'segments.json'
        manifest.write_text(json.dumps(dict(width=128,height=128,frames=count,overlap=5,segments=plan)))
        result=self.root/'merged.mp4'
        merge(manifest,result)
        self.assertEqual(runner.probe_video(result,count=True)['frames'],count)
        cap=cv2.VideoCapture(str(result))
        errors=[]
        try:
            for expected in frames:
                ok,actual=cap.read()
                self.assertTrue(ok)
                # Color conversion/compression is allowed; a wrong source
                # frame would have a large error on this unique random pattern.
                errors.append(float(np.abs(actual.astype(float)-expected).mean()))
            self.assertFalse(cap.read()[0])
        finally:
            cap.release()
        self.assertLess(max(errors),15)


if __name__=='__main__':
    unittest.main(verbosity=2)
