"""Optional pinned PromptHMR runtime; no implicit package or model downloads."""
from pathlib import Path
import json,sys,subprocess
from .paths import ROOT
from .storage import atomic_json,sha256

PACKAGES={'torch':'2.7.1+cu128','torchvision':'0.22.1+cu128','numpy':'1.26.4','timm':'0.9.0','smplx':'0.1.28'}


def snapshot(engine):
    engine=Path(engine).resolve();files={}
    for base in ('pipeline','prompt_hmr'):
        for p in (engine/base).rglob('*'):
            if p.suffix.lower() not in ('.py','.yaml','.yml','.json','.txt'):continue
            if '__pycache__' in p.parts:continue
            if not p.resolve().is_relative_to(engine):raise ValueError('PromptHMR engine path leaves its installation.')
            files[p.relative_to(engine).as_posix()]=sha256(p)
            if len(files)>8192:raise ValueError('PromptHMR engine inventory is too large.')
    files['LICENSE']=sha256(engine/'LICENSE')
    return files


def configuration(root=ROOT):
    path=Path(root)/'.setup/prompthmr-runtime.json'
    if not path.is_file():raise RuntimeError('PromptHMR runtime is not installed. Close the app and run Setup_PromptHMR.cmd. See docs/models/PROMPTHMR.md.')
    if path.stat().st_size>2*1024**2:raise ValueError('Invalid PromptHMR runtime configuration.')
    cfg=json.loads(path.read_text(encoding='utf8'))
    from tools.setup_prompthmr import identity
    if cfg.get('version')!=1 or not cfg.get('license_reviewed') or cfg.get('managed_id')!=identity(root):
        raise RuntimeError('PromptHMR runtime configuration changed. Close the app and run Setup_PromptHMR.cmd.')
    for key in ('python','engine'):
        if not isinstance(cfg.get(key),str) or not Path(cfg[key]).is_absolute() or not Path(cfg[key]).exists():
            raise RuntimeError('The configured PromptHMR runtime is missing. Run Setup_PromptHMR.cmd; saved work is preserved.')
    return cfg


def configure(folder,managed_id,root=ROOT):
    folder=Path(folder).resolve()
    info=json.loads((folder/'verification.json').read_text(encoding='utf8'))
    cfg=dict(version=1,python=str(folder/'python.exe'),engine=str(folder/'engine'),license_reviewed=True,
             managed_id=managed_id,compute_capability=info['compute_capability'],packages=PACKAGES,
             engine_files=snapshot(folder/'engine'))
    atomic_json(Path(root)/'.setup/prompthmr-runtime.json',cfg)
    return cfg


def activate():
    import importlib.metadata as metadata
    cfg=configuration()
    if Path(sys.executable).resolve()!=Path(cfg['python']).resolve() or sys.version_info[:2]!=(3,11):
        raise RuntimeError('PromptHMR must run in its separate Python 3.11 environment.')
    for name,version in PACKAGES.items():
        if metadata.version(name)!=version:raise RuntimeError('Unsupported PromptHMR '+name+' version. Run Setup_PromptHMR.cmd.')
    import torch
    if not torch.cuda.is_available() or list(torch.cuda.get_device_capability())!=cfg['compute_capability']:
        raise RuntimeError('PromptHMR native SAM2 kernels need verification for this GPU. Run Setup_PromptHMR.cmd.')
    if snapshot(cfg['engine'])!=cfg['engine_files']:raise RuntimeError('PromptHMR engine changed. Run Setup_PromptHMR.cmd before inference.')
    return Path(cfg['engine'])


def command(stage,args):
    return [configuration()['python'],'-I','-X','utf8','-u',str(ROOT/'prompthmr_worker_boot.py'),stage,*map(str,args)]
