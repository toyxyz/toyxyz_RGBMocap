"""Optional native camera overlay over the unchanged PromptHMR runtime."""
from pathlib import Path
import hashlib,json,sys
from .paths import ROOT
from .storage import sha256


def identity(root=ROOT):
    names=('installer/prompthmr-camera-lock.json','installer/prompthmr_camera_adapters.py','installer/verify_prompthmr_camera.py')
    return hashlib.sha256(json.dumps({n:sha256(Path(root)/n) for n in names},sort_keys=True).encode()).hexdigest()


def inventory(folder):
    result={};folder=Path(folder).resolve()
    for p in folder.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        if p.suffix not in ('.py','.pyd','.dll','.json') and p.name not in ('LICENSE','3rd-party-licenses.txt'):continue
        if not p.resolve().is_relative_to(folder):raise ValueError('Invalid camera support file path.')
        result[p.relative_to(folder).as_posix()]=sha256(p)
        if len(result)>2048:raise ValueError('Camera support inventory exceeds its limit.')
    return result


def configuration(root=ROOT,verify=False):
    from .prompthmr_runtime import configuration as base_configuration
    base=base_configuration(root);path=Path(root)/'.setup/prompthmr-camera-runtime.json'
    if not path.is_file():raise RuntimeError('Moving-camera support is not installed. Close the app, run Setup_PromptHMR.cmd and choose camera support. Fixed Camera remains available.')
    if path.stat().st_size>1024**2:raise ValueError('Invalid PromptHMR camera runtime configuration.')
    cfg=json.loads(path.read_text(encoding='utf8'))
    if (cfg.get('version')!=1 or cfg.get('support_id')!=identity(root) or cfg.get('base_id')!=base['managed_id']
            or cfg.get('compute_capability')!=base['compute_capability']):
        raise RuntimeError('PromptHMR camera support needs verification. Run Setup_PromptHMR.cmd and choose camera support.')
    folder=Path(cfg['folder']).resolve()
    if not folder.is_relative_to((Path(root)/'runtime_prompthmr').resolve()) or not folder.is_dir():raise ValueError('Invalid PromptHMR camera support folder.')
    if verify and inventory(folder)!=cfg['files']:raise RuntimeError('PromptHMR camera modules changed. Run Setup_PromptHMR.cmd to repair; previous motion was preserved.')
    return cfg


def activate():
    cfg=configuration(verify=True)
    import torch
    if list(torch.cuda.get_device_capability())!=cfg['compute_capability']:raise RuntimeError('Camera kernels must be rebuilt for this GPU.')
    folder=Path(cfg['folder']);sys.path.insert(0,str(folder));sys.path.insert(0,str(folder/'droid_slam'))
    return folder
