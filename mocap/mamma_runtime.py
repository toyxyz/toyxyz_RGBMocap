"""Optional, user-installed MAMMA engine; never distribute registered code.

The private environment is explicitly configured and checked. It cannot fall
back to an unrelated application runtime or download models during inference.
"""
from pathlib import Path
import json
import subprocess
import sys
from .paths import ROOT
from .storage import atomic_json, sha256

PACKAGES = {'torch': '2.7.1+cu128', 'numpy': '1.26.4', 'scipy': '1.17.1',
            'smplx': '0.1.28', 'timm': '1.0.30', 'torchvision': '0.22.1+cu128'}
ENGINE_FILES = ('LICENSE', 'segmentation/run_ma_masks.py', 'landmarks/run_ma_2d.py',
                'optimization/run_ma_3d.py', 'capture/run_ma_cap.py',
                'landmarks/lib/datasets/vitdet_dataset.py',
                'optimization/losses/losses.py', 'speed_landmarks.py', 'speed_crop.py')


def configuration(root=ROOT):
    path = Path(root)/'.setup/mamma-runtime.json'
    if not path.is_file():
        raise RuntimeError('MAMMA runtime is not installed. Close the app and run Setup_MAMMA.cmd. See docs/MAMMA.md.')
    if path.stat().st_size > 2*1024**2:
        raise ValueError('Invalid MAMMA runtime configuration.')
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('version') not in (2,3) or not value.get('license_reviewed'):
        raise ValueError('Review the official MAMMA license before configuring its optional environment.')
    for key in ('python', 'engine'):
        if not isinstance(value.get(key), str) or not Path(value[key]).is_absolute():
            raise ValueError('Invalid MAMMA '+key+' path.')
    if not Path(value['python']).is_file() or not Path(value['engine']).is_dir():
        raise RuntimeError('The configured MAMMA environment is missing. Its project data was preserved.')
    if value.get('version')==3:
        from tools.setup_mamma import identity
        if value.get('managed_id')!=identity(root):
            raise RuntimeError('MAMMA runtime configuration changed. Close the app and run Setup_MAMMA.cmd; other models and projects are preserved.')
    return value


def command(stage, args):
    cfg = configuration()
    return [cfg['python'], '-I','-X','utf8', '-u', str(ROOT/'mamma_worker_boot.py'), stage, *map(str, args)]


def activate():
    import importlib.metadata as metadata
    cfg = configuration()
    if Path(sys.executable).resolve() != Path(cfg['python']).resolve() or sys.version_info[:2] != (3, 11):
        raise RuntimeError('MAMMA must run in its separately configured Python 3.11 environment.')
    for name, version in PACKAGES.items():
        if metadata.version(name) != version:
            raise RuntimeError('Unsupported MAMMA '+name+' version. Keep its validated environment separate.')
    if cfg.get('version')==3:
        import torch
        if not torch.cuda.is_available() or list(torch.cuda.get_device_capability())!=cfg.get('compute_capability'):
            raise RuntimeError('MAMMA native kernels were built for a different GPU architecture. Run Setup_MAMMA.cmd on this PC.')
    engine = Path(cfg['engine'])
    verify_engine(cfg)
    return engine


def snapshot(engine):
    engine=Path(engine).resolve()
    names=set(ENGINE_FILES)
    for folder in ('capture','segmentation','landmarks','optimization'):
        for path in (engine/folder).rglob('*'):
            if path.suffix.lower() not in ('.py','.yaml','.yml','.json'):continue
            if any(part in ('__pycache__','.git','outputs','results','data') for part in path.relative_to(engine).parts):continue
            if not path.resolve().is_relative_to(engine):raise ValueError('Linked MAMMA engine code leaves its installation.')
            names.add(path.relative_to(engine).as_posix())
            if len(names)>8192:raise ValueError('MAMMA engine code inventory is too large.')
    return {name:sha256(engine/name) for name in sorted(names)}


def verify_engine(cfg):
    if snapshot(cfg['engine'])!=cfg.get('engine_files'):
        raise RuntimeError('MAMMA engine or configuration changed. Run Setup_MAMMA.cmd or revalidate the separately configured environment.')


def configure(python, engine, root=ROOT, managed_id=None):
    python, engine = Path(python).resolve(), Path(engine).resolve()
    for name in ENGINE_FILES:
        if not (engine/name).is_file():
            raise ValueError('Missing validated Windows engine component: '+name)
    # Version and imports are checked by the target interpreter, not this GUI.
    code = ('import sys,json,importlib.metadata as m;'
            'import torch,numpy,scipy,smplx,timm,torchvision,sam2,pytorch_sdf,open_clip;'
            'print(json.dumps(dict(python=list(sys.version_info[:3]),packages={n:m.version(n) for n in '+repr(list(PACKAGES))+'},cuda=torch.cuda.is_available(),compute_capability=list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None)))')
    result = subprocess.run([str(python), '-I','-X','utf8', '-c', code], capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=120, creationflags=0x08000000)
    if result.returncode:
        raise RuntimeError('MAMMA runtime import check failed:\n'+result.stderr[-4000:])
    checked = json.loads(result.stdout.strip().splitlines()[-1])
    if checked['python'][:2] != [3, 11] or checked['packages'] != PACKAGES or not checked['cuda']:
        raise ValueError('The environment does not match the validated CUDA/Python MAMMA profile: '+str(checked))
    entry = dict(version=2, python=str(python), engine=str(engine), license_reviewed=True,
                 packages=PACKAGES, engine_files=snapshot(engine))
    if managed_id:entry.update(version=3,managed_id=managed_id,compute_capability=checked['compute_capability'])
    atomic_json(Path(root)/'.setup/mamma-runtime.json', entry)
    return checked
