"""Rebuild local pure-Python wheel sources and preserve pinned runtime groups."""
from pathlib import Path
import hashlib,importlib.util,json,os,re,subprocess,sys,zipfile

ROOT=Path(__file__).resolve().parents[1]

def _registry():
    # Loaded by path: installer scripts run isolated, without the app root on sys.path.
    spec=importlib.util.spec_from_file_location('runtime_registry',ROOT/'mocap/runtime_registry.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
REGISTRY=_registry()

def cache_dependency_files(root):
    """Keep historical worker receipt inputs, without using them to install packages."""
    root=Path(root)
    worker=(root/'mocap/worker.py').read_text(encoding='utf8')
    names={'requirements-windows.txt','requirements-flow-windows.txt','requirements-hand-windows.txt'}
    names.update(re.findall(r"ROOT\s*/\s*['\"](requirements-[^'\"]+\.txt)['\"]",worker))
    paths=[root/name for name in sorted(names)]
    missing=[p.name for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError('Missing worker cache dependency files: '+', '.join(missing))
    return paths

def application_version():
    match=re.search(r'^__version__ = "(\d+\.\d+\.\d+)"$',(ROOT/'mocap/__init__.py').read_text(encoding='utf8'),re.M)
    if match is None:raise ValueError('Expected an explicit three-part application version')
    return match.group(1)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def prepared_lock(work):
    work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    lock=json.loads((ROOT/REGISTRY.LOCK).read_text(encoding='utf8'))
    if sha(ROOT/'mocap/runtime_registry.py')!=lock['registry_sha256']:
        raise RuntimeError('Runtime registry changed. Review the profile pins and update the source lock explicitly.')
    for name in sorted({p['bundled'] for g in lock['groups'] for p in g['packages'] if p.get('bundled')}):
        filename=Path(name).name;source=ROOT/'installer/runtime_sources'/Path(filename).stem
        if not source.is_dir():raise FileNotFoundError(source)
        wheel=work/name;wheel.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(wheel,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p in sorted(source.rglob('*')):
                if not p.is_file() or p.name=='SOURCE_WHEEL.json':continue
                info=zipfile.ZipInfo(p.relative_to(source).as_posix(),date_time=(1980,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
                z.writestr(info,p.read_bytes())
        for group in lock['groups']:
            for p in group['packages']:
                if p.get('bundled')==name:p.update(sha256=sha(wheel),bytes=wheel.stat().st_size)
    # The source pins are unchanged; locally rebuilt wheel archive bytes have a new identity.
    lock.pop('id',None);lock['id']=hashlib.sha256(json.dumps(lock,sort_keys=True).encode()).hexdigest()
    (work/'runtime-lock.json').write_text(json.dumps(lock,indent=2),encoding='utf8')
    return lock

def find_uv(explicit=None):
    if explicit:return Path(explicit).resolve()
    p=ROOT/'.bootstrap/Scripts/uv.exe'
    if not p.is_file():
        subprocess.run([sys.executable,'-m','venv',str(ROOT/'.bootstrap')],check=True)
        subprocess.run([str(ROOT/'.bootstrap/Scripts/python.exe'),'-m','pip','install','uv==0.13.1'],check=True)
    return p

def run(args):
    env=os.environ.copy();env.update(PYTHONUTF8='1',UV_PYTHON_INSTALL_DIR=str(ROOT/'.runtime/python'),
        UV_CACHE_DIR=str(ROOT/'.cache/uv'),UV_PYTHON_NO_REGISTRY='1',UV_NO_CONFIG='1')
    subprocess.run([str(x) for x in args],cwd=ROOT,env=env,check=True)
