"""Press the two-dancer page's button the way a person does (Gradio API of the page), and write every status update.

    duo_page_run.py run VIDEO LEFT_PICTURE RIGHT_PICTURE CAMERA[,CAMERA...]     -> 전체 실행
    duo_page_run.py resume                                                     -> 멈춘 작업 이어서 하기
"""
import sys
import time

from gradio_client import Client, handle_file

client = Client('http://127.0.0.1:7863/', verbose=False)
if sys.argv[1] == 'run':
    job = client.submit({'video': handle_file(sys.argv[2])}, handle_file(sys.argv[3]), handle_file(sys.argv[4]), None, None, '', '',
                        sys.argv[5].split(','), '공식 H3 · 3D로 만든 방을 그대로 둡니다', 0, None, api_name='/run')
else:
    job = client.submit(api_name='/resume')
last = None
while not job.done():
    outputs = job.outputs()
    if outputs and outputs[-1][0] != last:
        last = outputs[-1][0]
        print(time.strftime('%H:%M:%S'), str(last).replace('\n', ' | ')[:400], flush=True)
    time.sleep(5)
try:
    final = job.result()
    print(time.strftime('%H:%M:%S'), 'FINAL', str(final[0]).replace('\n', ' | ')[:1500], '| videos:', final[1:5], '| summary:', final[5], flush=True)
except Exception as error:  # the page refused (gr.Error) or the run failed
    print(time.strftime('%H:%M:%S'), 'ERROR', type(error).__name__, str(error)[:1500], flush=True)
