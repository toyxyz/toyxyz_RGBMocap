"""Versioned GVHMR observations: stable FP32 settings and safe cache migration."""
import hashlib
import json
from pathlib import Path
import shutil
import time

from .storage import atomic_json,sha256
from .paths import ROOT


def configure_precision():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def identity(project,batch):
    import torch
    from .models import BY_KEY
    def stamp(path):
        stat=path.stat()
        return [stat.st_size,stat.st_mtime_ns]
    code=[Path(__file__),ROOT/'mocap/inference_data.py',ROOT/'mocap/engines.py',
          ROOT/'third_party/GVHMR/hmr4d/utils/preproc/vitpose.py',
          ROOT/'third_party/GVHMR/hmr4d/utils/preproc/vitfeat_extractor.py',
          ROOT/'third_party/GVHMR/hmr4d/network/hmr2/__init__.py',
          ROOT/'third_party/GVHMR/hmr4d/utils/preproc/vitpose_pytorch/src/vitpose_infer/model_builder.py']
    from .resource_cache import code_hash
    # Exact-byte aliases retain preprocessing after platform-only edits to these two files.
    aliased={Path(__file__),ROOT/'mocap/engines.py'}
    value=dict(version=1,precision='float32',flip=True,deterministic_conv=True,
        batch=batch,torch=str(torch.__version__),cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),
        threads=torch.get_num_threads(),code=[(code_hash if path in aliased else sha256)(path) for path in code],
        tracking=sha256(project/'tracking.npz'),video=stamp(project/'input.mp4'),
        models={key:stamp(BY_KEY[key].path) for key in ('vitpose','hmr2')})
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),value


def preserve_previous(project):
    """Keep existing observations/results intact before replacing stale inputs."""
    names=['gvhmr_keypoints.pt','features.pt','gvhmr_preprocess.json']
    existing=[project/name for name in names if (project/name).exists()]
    if not existing:return
    archive=project/'preprocessing_history'/str(time.time_ns())
    archive.mkdir(parents=True)
    for path in existing:shutil.copy2(path,archive/path.name)
    # Results are not deleted. Normal stage invalidation will recompute them.
    for name in ('hmr4d_results.pt','opt_results.pt'):
        path=project/name
        if path.exists():shutil.copy2(path,archive/name)


def cache_state(project,signature,value):
    path=project/'gvhmr_preprocess.json'
    try:state=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):state={}
    if state.get('signature')!=signature:
        preserve_previous(project)
        state=dict(signature=signature,settings=value,complete={})
        atomic_json(path,state)
    return state


def cached(project,state,name):
    path=project/name
    # Hash only small extracted tensors, never the multi-GB source on each call.
    return path.is_file() and state['complete'].get(name)==sha256(path)


def mark_complete(project,state,name):
    state['complete'][name]=sha256(project/name)
    atomic_json(project/'gvhmr_preprocess.json',state)
