"""Install the six selected model files after the operator confirms applicable usage rights."""
import json
import os
from pathlib import Path
from config import MODEL_INSTALL, MODELS, MODEL_FILES
from runtime import preflight, write_json

os.environ['HF_HOME']=str(MODEL_INSTALL/'cache/hf-download')
os.environ.setdefault('HF_TOKEN_PATH',str(Path.home()/'.cache/huggingface/token'))
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['TMPDIR']=str(MODEL_INSTALL/'cache/tmp')
Path(os.environ['TMPDIR']).mkdir(parents=True,exist_ok=True)

from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open

preflight(100)
api=HfApi()
manifest_path=MODEL_INSTALL/'outputs/models.json'
manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
revisions={}
for repo, filename,folder in MODEL_FILES:
    prior=manifest.get(filename,{})
    final=MODELS/folder/Path(filename).name
    if prior.get('complete') and final.is_file() and final.stat().st_size==prior.get('size_bytes'):
        print('Already ready:',final.name,flush=True)
        continue
    if repo not in revisions:
        revisions[repo]=api.model_info(repo).sha
    destination=MODELS/folder
    destination.mkdir(parents=True,exist_ok=True)
    print('Downloading:',filename,flush=True)
    # Preserve the repository hierarchy in a managed staging root, then atomically place the selected file.
    staging=MODEL_INSTALL/'cache/model-downloads'/repo.replace('/','--')
    downloaded=Path(hf_hub_download(repo,filename,revision=revisions[repo],local_dir=staging))
    with safe_open(str(downloaded),framework='numpy') as handle:
        tensors=len(handle.keys())
        if not tensors:
            raise RuntimeError('Empty safetensors file: '+filename)
    final=destination/Path(filename).name
    downloaded.replace(final)
    manifest[filename]={'repo':repo,'revision':revisions[repo],'path':str(final),
                        'size_bytes':final.stat().st_size,'tensor_count':tensors,'complete':True}
    write_json(manifest_path,manifest)
    print('Ready:',final.name,flush=True)
print('All H3 model files are ready.',flush=True)
