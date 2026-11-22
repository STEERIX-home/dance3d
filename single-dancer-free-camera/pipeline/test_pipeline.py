"""CPU regression checks. Set RGBMOCAP_TEST_ROOT to a managed job subdirectory."""
import contextlib
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import runner
import result_bundle


class PipelineRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['RGBMOCAP_TEST_ROOT']).resolve()
        cls.root.relative_to(runner.JOBS.resolve())
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.guide = cls.root/'guide-audio.mp4'
        cls.raw = cls.root/'h3-generated-audio.mp4'
        for path, frequency in [(cls.guide, 440), (cls.raw, 880)]:
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                'color=c=gray:s=128x128:r=24:d=2.333333333', '-f', 'lavfi', '-i',
                f'sine=frequency={frequency}:sample_rate=48000:duration=2.333333333',
                '-frames:v', '56', '-c:v', 'libx264', '-c:a', 'aac', '-threads', '2', str(path)], check=True)
        from PIL import Image
        cls.character = cls.root/'character.png'
        Image.new('RGB', (256, 256), 'white').save(cls.character)

    def test_bad_input_rejected_before_compute(self):
        with self.assertRaisesRegex(ValueError, '구간'):
            runner.validate_options(self.guide, 20, 56, 768, 768, 0, 8, 1)
        with self.assertRaisesRegex(ValueError, '카메라'):
            runner.validate_options(self.guide, 0, 56, 768, 768, float('nan'), 8, 1)
        with self.assertRaises(Exception):
            runner.validate_image(self.guide)

    def test_two_reference_graph_preserves_video_and_timing(self):
        graph = runner.make_h3_prompt('body.png', 'dance.mp4', 768, 768, 124,
                    'Use <Picture 1> with <Video 1>.', 42, 'test', 'face.png')
        inputs = graph['136']['inputs']
        self.assertEqual(graph[inputs['ref_images.ref_image_0'][0]]['inputs']['image'], 'body.png')
        self.assertEqual(graph[inputs['ref_images.ref_image_1'][0]]['inputs']['image'], 'face.png')
        self.assertIn('<Picture 2>', graph['138']['inputs']['value'])
        self.assertEqual(graph['152']['inputs']['file'], 'dance.mp4')
        self.assertEqual(inputs['length'], 124)
        self.assertEqual(graph['152']['inputs']['length'], 124)

    def test_cancel_waits_for_own_prompt_and_leaves_other_queue_entries(self):
        queue = {'queue_running': [[0, 'ours']], 'queue_pending': [[1, 'other']]}
        calls = []
        def request(url, data=None):
            calls.append((url, data))
            if url.endswith('/interrupt'):
                queue['queue_running'] = []
                return {}
            return queue
        with patch.object(runner, 'get_json', request):
            runner.cancel_h3_prompt('ours')
        self.assertEqual(queue['queue_pending'], [[1, 'other']])
        self.assertEqual(sum(url.endswith('/interrupt') for url, _ in calls), 1)
        self.assertFalse(any(data and 'delete' in data for _, data in calls))

    def test_final_video_keeps_dance_audio_instead_of_generated_audio(self):
        import numpy as np
        job = self.root/'audio-job'
        for part in ('logs', 'outputs'):
            (job/part).mkdir(parents=True, exist_ok=True)
        session = self.root/'fake-h3-session'
        (session/'outputs/input').mkdir(parents=True, exist_ok=True)
        (session/'outputs/results').mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.raw, session/'outputs/results/result.mp4')

        @contextlib.contextmanager
        def managed(_):
            yield job

        @contextlib.contextmanager
        def server(_):
            yield session

        def request(url, data=None):
            if url.endswith('/prompt'):
                return {'prompt_id': 'test-prompt'}
            return {'test-prompt': {'status': {'status_str': 'success'},
                    'outputs': {'92': {'videos': [{'type': 'output', 'filename': 'result.mp4'}]}}}}

        with patch.object(runner, 'managed_job', managed), patch.object(runner, 'h3_session', server), \
                patch.object(runner, 'get_json', request):
            output = runner.swap(self.guide, self.character, 'Use <Picture 1> with <Video 1>.',
                                 42, preserve_face=False)
        result = output/'outputs/character-swap.mp4'
        info = runner.probe_video(result, count=True)
        self.assertEqual(info['frames'], 56)
        self.assertTrue(info['audio'])
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(result),
                    '-vn', '-f', 'f32le', '-ac', '1', '-ar', '48000', '-'])
        signal = np.frombuffer(raw, dtype=np.float32)
        spectrum = np.abs(np.fft.rfft(signal))
        peak = np.fft.rfftfreq(len(signal), 1/48000)[np.argmax(spectrum)]
        self.assertLess(abs(peak-440), 2, 'Output must retain the original 440 Hz audio, not H3 880 Hz audio')

    def test_desktop_export_is_real_verified_and_never_overwrites(self):
        job = self.root/'export-job'
        (job/'outputs').mkdir(parents=True, exist_ok=True)
        (job/'outputs/test.txt').write_text('사용자가 실제로 여는 결과 파일')
        with patch.object(result_bundle, 'DESKTOP_RESULTS', self.root/'desktop'):
            destination = result_bundle.export_desktop(job)
            self.assertFalse(destination.is_symlink())
            hashes = json.loads((destination/'sha256.json').read_text())
            self.assertEqual(hashes['test.txt'], result_bundle.digest(job/'outputs/test.txt'))
            with self.assertRaises(FileExistsError):
                result_bundle.export_desktop(job)


if __name__ == '__main__':
    unittest.main(verbosity=2)
