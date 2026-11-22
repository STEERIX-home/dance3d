"""Press the page's buttons the way a person does (Gradio API of the page), and write every status update.

    page_run.py run   JOINTS_CHOICE PICTURE [TITLE]     -> 전체 실행
    page_run.py resume                                 -> 멈춘 작업 이어서 하기
"""
import sys
import time

from gradio_client import Client, handle_file

client = Client('http://127.0.0.1:7865/', verbose=False)
if sys.argv[1] == 'run':
    picture = handle_file(sys.argv[3]) if sys.argv[3] != '-' else None
    job = client.submit(sys.argv[2], picture, None, True, sys.argv[4] if len(sys.argv) > 4 else '', '보통', True, api_name='/run')
else:
    job = client.submit('', api_name='/resume')
last = None
while not job.done():
    outputs = job.outputs()
    if outputs and outputs[-1] != last:
        last = outputs[-1]
        print(time.strftime('%H:%M:%S'), str(last[0]).replace('\n', ' | ')[:400], flush=True)
    time.sleep(5)
try:
    final = job.result()
    print(time.strftime('%H:%M:%S'), 'FINAL', str(final[0]).replace('\n', ' | ')[:1500], '| video:', final[1], '| note:', final[2], flush=True)
except Exception as error:  # the page refused (gr.Error) or the run failed
    print(time.strftime('%H:%M:%S'), 'ERROR', type(error).__name__, str(error)[:1500], flush=True)
