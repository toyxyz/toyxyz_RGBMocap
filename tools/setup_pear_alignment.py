"""Download pinned GeoCalib/Kornia support; no pip install or shared-package changes."""
from pathlib import Path
import json,zipfile,stat,uuid,subprocess
from mocap.paths import ROOT
from mocap.storage import atomic_json
from mocap.pear.alignment_support import identity,configuration

def prepare(cache=None):
    from tools.support_components import download
    try:
        current=configuration();print('Keeping verified PEAR camera alignment support.',flush=True);return current
    except (OSError,ValueError,KeyError,TypeError):pass
    lock=json.loads((ROOT/'installer/pear-alignment.json').read_text('utf8'))
    cache=Path(cache) if cache else ROOT/'.setup/pear-downloads';cache.mkdir(parents=True,exist_ok=True)
    folder=ROOT/'runtime_pear'/('alignment-'+identity()[:16]+'-'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
    for name,row in lock['sources'].items():
        print('Preparing PEAR camera alignment: '+name,flush=True)
        archive=download(cache,name,row)
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                p=Path(info.filename)
                if p.is_absolute() or '..' in p.parts or '\\' in info.filename or ':' in info.filename or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unsafe alignment archive')
                if info.file_size>128*1024**2:raise ValueError('Alignment support file too large')
                if info.is_dir() or '__pycache__' in p.parts or p.suffix in ('.pyc','.pyo'):continue
                if name=='geocalib':
                    p=Path(*p.parts[1:])
                    if not p.parts or (p.parts[0]!='geocalib' and p.name not in ('LICENSE','NOTICE','LICENSE-WEIGHTS')):continue
                    if p.name in ('LICENSE','NOTICE','LICENSE-WEIGHTS'):p=Path('licenses/geocalib')/p
                target=folder/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info))
    code='import sys;sys.path.insert(0,'+repr(str(ROOT))+');from mocap.runtime_registry import activate;activate('+repr(str(ROOT))+',"flow");sys.path.insert(0,'+repr(str(folder))+');import torch,kornia,kornia_rs;from geocalib import GeoCalib;assert kornia.__version__=="0.7.4";assert torch.cuda.is_available();print("PEAR camera alignment imports verified")'
    subprocess.run([str(ROOT/'runtime/python.exe'),'-I','-X','utf8','-B','-c',code],check=True,timeout=180)
    atomic_json(folder/'PROVENANCE.json',lock)
    files={p.relative_to(folder).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in folder.rglob('*') if p.is_file()}
    atomic_json(ROOT/'.setup/pear-alignment.json',dict(identity=identity(),folder=folder.relative_to(ROOT).as_posix(),files=files))
    print('PEAR camera alignment support ready; model weights remain separate.',flush=True);return folder
