"""Verify an installed group, its RECORD payloads and actual import precedence."""
import base64, hashlib, importlib.metadata as metadata, importlib.util, json, sys
from pathlib import Path

root=Path(sys.argv[1]).resolve();group=json.loads(Path(sys.argv[2]).read_text(encoding='utf8'));registry=Path(sys.argv[3])
spec=importlib.util.spec_from_file_location('runtime_registry',registry)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
base_root=root
if group.get('base') and not module.managed_python(root,'shared').is_file(): base_root=Path(sys.argv[4])
report=module.activate(root,group['key'],base_root)
folder=root/group['target'];site=module.site_packages(folder,group['python']) if not group.get('base') and group['key'] in {'shared','exmr'} else folder
distributions={d.metadata['Name'].lower().replace('_','-'):d for d in metadata.distributions(path=[str(site)])}
checked=0
for p in group['packages']:
    dist=distributions.get(p['name'])
    if dist is None or dist.version!=p['version']: raise RuntimeError('Missing/wrong package: '+p['name'])
    if not dist.files: raise RuntimeError('Missing RECORD: '+p['name'])
    for entry in dist.files:
        path=Path(dist.locate_file(entry)).resolve()
        # Pruned development products and scripts outside site-packages are not
        # shipped; everything used as package code/data retains wheel hashes.
        if not path.is_relative_to(site.resolve()) or path.suffix.lower() in {'.lib','.exp','.pdb','.obj','.o','.a','.ilk','.pyc','.pyo'}: continue
        if not path.is_file(): raise RuntimeError('Missing runtime file: '+str(path))
        if entry.hash:
            h=hashlib.new(entry.hash.mode)
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024**2),b''): h.update(block)
            value=base64.urlsafe_b64encode(h.digest()).rstrip(b'=').decode()
            if value!=entry.hash.value: raise RuntimeError('Damaged runtime file: '+str(path))
        checked+=1
import torch,numpy,scipy,cv2,timm
if group['key']=='hand': import ultralytics
if group['key']=='gui':
    import PySide6
    from PySide6 import QtWidgets,QtOpenGLWidgets
    if PySide6.__version__!='6.8.3':raise RuntimeError('GUI Qt version mismatch')
report.update(checked_files=checked,torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__)
print(json.dumps(report))
