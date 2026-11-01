"""Prepare PEAR source/support files without upgrading an installed runtime."""
from pathlib import Path
import sys,json,hashlib,zipfile,shutil,os,argparse,uuid,subprocess,stat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mocap.storage import atomic_json,sha256
from mocap.pear_runtime import identity,base_identity,configuration
from installer.pear_adapters import adapt
from tools.session_lock import app_lock

def prepare(cache=None):
    """Called under the installer lock, after committing the shared base."""
    from tools.support_components import download
    lock=json.loads((ROOT/'installer/pear-sources.json').read_text(encoding='utf-8'))
    if not (ROOT/'runtime/python.exe').is_file():raise RuntimeError('Run install.cmd first.')
    try:current=configuration()
    except (OSError,ValueError,KeyError,TypeError):current=None
    if current is not None:
        print('Keeping verified PEAR support over the shared runtime.',flush=True)
        from tools.setup_pear_alignment import prepare as prepare_alignment
        prepare_alignment(cache);return current
    base=base_identity()
    folder=ROOT/'runtime_pear'/('support-'+identity()[:20]+'-'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
    cache=Path(cache) if cache else ROOT/'.setup/pear-downloads';cache.mkdir(parents=True,exist_ok=True)
    try:
        for name,row in lock['sources'].items():
            print('Preparing PEAR support: '+name,flush=True)
            archive=download(cache,name,dict(row,filename=name+('.whl' if name=='roma' else '.zip')))
            with zipfile.ZipFile(archive) as z:
                for info in z.infolist():
                    p=Path(info.filename)
                    if p.is_absolute() or '..' in p.parts or '\\' in info.filename or ':' in info.filename or stat.S_ISLNK(info.external_attr >> 16):raise ValueError('Unsafe source archive')
                    if info.file_size>512*1024**2:raise ValueError('Source file too large')
                    if info.is_dir() or '__pycache__' in p.parts or p.suffix in ('.pyc','.pyo'):continue
                    parts=p.parts[1:]
                    target=None
                    if name=='pear' and parts and (parts[0] in ['models','utils','configs'] or Path(*parts).as_posix() in ['inference_images.py','LICENSE.txt']):
                        target=folder/'engine'/Path(*parts)
                    elif name=='pytorch3d' and parts and (tuple(parts[:2]) in (('pytorch3d','transforms'),('pytorch3d','common')) or Path(*parts).as_posix() in ['pytorch3d/__init__.py','LICENSE','LICENSE_BSD','NOTICE']):
                        target=folder/'packages'/Path(*parts)
                    elif name=='roma':target=folder/'packages'/p
                    if target:
                        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info))
            if name=='pear':adapt(folder/'engine')
        atomic_json(folder/'RGB_MOCAP_PROVENANCE.json',dict(sources=lock['sources'],
            adapter_sha256=sha256(ROOT/'installer/pear_adapters.py'),
            changes='Windows live-inference imports; shared runtime and pure-Python geometry constructor helpers. Original licenses retained.'))
        print('Verifying PEAR shared-runtime imports and CUDA kernels...',flush=True)
        code='import sys; sys.path[:0]='+repr([str(ROOT),str(folder/'packages'),str(folder/'engine')])+'; from mocap.runtime_registry import activate; activate('+repr(str(ROOT))+', "shared"); import torch, numpy, scipy, timm, cv2, roma; from pytorch3d.transforms import axis_angle_to_matrix; from torchvision.ops import nms; assert torch.cuda.is_available(), \"Compatible NVIDIA GPU required\"; x=torch.zeros((1,3),device=\"cuda\"); assert axis_angle_to_matrix(x).isfinite().all(); assert nms(torch.tensor([[0.,0.,1.,1.]],device=\"cuda\"),torch.ones(1,device=\"cuda\"),.5).numel()==1; print(\"PEAR CUDA verification passed\")'
        subprocess.run([str(ROOT/'runtime/python.exe'),'-I','-X','utf8','-B','-c',code],check=True,timeout=180)
        files={p.relative_to(folder).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
        atomic_json(ROOT/'.setup/pear-runtime.json',dict(version=2,identity=identity(),base_identity=base,folder=folder.relative_to(ROOT).as_posix(),files=files))
        print('PEAR support ready. Existing runtime packages, models and projects were not changed.',flush=True)
        from tools.setup_pear_alignment import prepare as prepare_alignment
        prepare_alignment(cache);return folder
    except Exception:
        # Keep failed staging for diagnosis; no previous configuration is overwritten.
        print('PEAR setup failed. Existing support generations were preserved.',flush=True);raise


def setup(cache=None):
    # Developer repair entry point; ordinary users run install.cmd/update.cmd.
    with app_lock(ROOT):return prepare(cache)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache');a=p.parse_args();setup(a.cache)
