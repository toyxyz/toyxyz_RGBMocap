"""Dependency-free registry of validated, separated worker environments.

Overlays belong to a specific base, not an independently upgradable environment.
App startup never installs packages or changes environments.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

PROFILES = {
    'shared': dict(folder='runtime', dev='.venv', base=None, python='3.10.22', packages={'torch':'2.7.1+cu128','numpy':'1.23.5','scipy':'1.14.1','timm':'0.9.12'}),
    'gui': dict(folder='runtime_gui', dev='.venv', base='shared', python='3.10.22', packages={'torch':'2.7.1+cu128','numpy':'1.23.5','scipy':'1.14.1','timm':'0.9.12'}),
    'flow': dict(folder='runtime_flow', dev='.venv-flow', base='shared', python='3.10.22', packages={'torch':'2.7.1+cu128','numpy':'1.26.4','scipy':'1.15.3','timm':'1.0.30'}),
    'hand': dict(folder='runtime_hand', dev='.venv-hand', base='shared', python='3.10.22', packages={'torch':'2.7.1+cu128','numpy':'1.23.5','timm':'0.9.12','ultralytics':'8.3.80'}),
    'htd': dict(folder='runtime_htd', dev='.venv-htd', base='shared', python='3.10.22', packages={'torch':'2.7.1+cu128','numpy':'1.23.5','timm':'1.0.12'}),
    'exmr': dict(folder='runtime_exmr', dev='.venv-exmr', base=None, python='3.12.10', packages={'torch':'2.8.0+cu128','numpy':'2.3.5','scipy':'1.15.2','timm':'0.9.12'}),
}

WINDOWS = os.name == 'nt'
# Every platform pins the same package versions; only the wheel files differ.
LOCK = 'installer/runtime-lock.json' if WINDOWS else 'installer/runtime-lock-linux.json'
SUPPORT_LOCK = 'installer/git-support-lock.json' if WINDOWS else 'installer/git-support-lock-linux.json'
INSTALL_COMMAND = 'install.cmd' if WINDOWS else './install.sh'

def _minor(version):
    return 'python' + '.'.join(version.split('.')[:2])

def interpreter(folder, version):
    """Private CPython of a managed runtime: the Windows home folder or POSIX prefix bin/."""
    return Path(folder)/'python.exe' if WINDOWS else Path(folder)/'bin'/_minor(version)

def site_packages(folder, version):
    return Path(folder)/'Lib/site-packages' if WINDOWS else Path(folder)/'lib'/_minor(version)/'site-packages'

def venv_interpreter(folder):
    return Path(folder)/('Scripts/python.exe' if WINDOWS else 'bin/python')

def base_profile(key):
    spec = PROFILES[key]
    return PROFILES[spec['base']] if spec['base'] else spec

def managed_python(root, key):
    base = base_profile(key)
    return interpreter(Path(root)/base['folder'], base['python'])

def profile_for(stage, args=()):
    if stage.startswith('exmr'): return 'exmr'
    if stage.startswith('flow'): return 'flow'
    if stage == 'handflow': return 'hand'
    if stage in {'pva','refine'}: return 'htd'
    if stage in {'rig','mesh','export','correct','character'} and '--project' in args:
        try:
            path=Path(args[args.index('--project')+1])/'project.json'
            model=json.loads(path.read_text(encoding='utf-8')).get('pipeline',{}).get('model')
        except (OSError,ValueError,IndexError) as exc:
            raise RuntimeError('Cannot select a processing runtime: project metadata is unavailable.') from exc
        if model=='flowhmr': return 'flow'
        if model not in {None,'gvhmr','exmr','mamma','prompthmr'}: raise RuntimeError('Unsupported saved motion model: '+str(model))
    return 'shared'

def command(root,stage,args=(),*,frozen):
    root=Path(root);key=profile_for(stage,args);spec=PROFILES[key]
    if frozen or os.environ.get('RGB_MOCAP_MANAGED') == '1':
        python=managed_python(root,key)
        if not (root/spec['folder']).is_dir() or not python.is_file():
            raise RuntimeError(key+' runtime is missing. Run Setup to repair this installation.')
        # Isolated mode ignores PYTHONIOENCODING/PYTHONUTF8 from the GUI.
        return [str(python),'-I','-X','utf8',str(root/'worker_boot.py'),stage,*map(str,args)]
    python=venv_interpreter(root/spec['dev'])
    if not python.is_file(): raise RuntimeError(key+' development runtime is missing. Prepare its pinned environment.')
    return [str(python),'-X','utf8','-m','mocap.worker_progress',stage,*map(str,args)]

def activate(root,key,base_root=None):
    """Validate effective imports before loading model or CUDA modules."""
    import importlib.metadata as metadata
    import importlib.util
    import sys
    root=Path(root).resolve();spec=PROFILES[key];base=base_profile(key)
    base_root=Path(base_root).resolve() if base_root else root
    python=interpreter(base_root/base['folder'],base['python'])
    if Path(sys.executable).resolve()!=python.resolve():
        raise RuntimeError('Wrong interpreter for '+key+'. Run Setup to repair this installation.')
    if '.'.join(map(str,sys.version_info[:3]))!=spec['python']: raise RuntimeError('Unexpected Python version for '+key)
    site=site_packages(base_root/base['folder'],base['python'])
    overlay=root/spec['folder'] if spec['base'] else site
    if not overlay.is_dir(): raise RuntimeError(key+' runtime is missing. Run Setup to repair it.')
    if spec['base']: sys.path.insert(0,str(overlay))
    own={d.metadata['Name'].lower().replace('_','-') for d in metadata.distributions(path=[str(overlay)])}
    paths={}
    for name,version in spec['packages'].items():
        dist=metadata.distribution(name);module=importlib.util.find_spec(name)
        if dist.version!=version or module is None or not module.origin:
            raise RuntimeError('Incompatible '+name+' in '+key+'. Run Setup to repair it.')
        actual=Path(module.origin).resolve();expected=overlay if name in own else site
        if not actual.is_relative_to(expected.resolve()): raise RuntimeError('Unexpected '+name+' import path in '+key+': '+str(actual))
        paths[name]=str(actual)
    return dict(profile=key,python=str(python),packages=spec['packages'],imports=paths)
