"""Optional camera-align support over the pinned Flow runtime, without upgrades."""
from pathlib import Path
import json,hashlib,sys
from ..paths import ROOT

def identity():return hashlib.sha256((ROOT/'installer/pear-alignment.json').read_bytes()+(ROOT/'tools/setup_pear_alignment.py').read_bytes()).hexdigest()
def base_identity():
    """Bind the overlay to both installed runtime groups, including replacements."""
    receipt=json.loads((ROOT/'SETUP_RECEIPT.json').read_text('utf8'))
    rows={}
    for key,name in [('shared','runtime'),('flow','runtime_flow')]:
        group_id=receipt['group_ids'][key]
        if not isinstance(group_id,str) or not group_id:raise ValueError('Missing alignment base runtime receipt')
        folder=ROOT/name
        if folder.is_symlink() or not folder.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Invalid alignment base path')
        rows[key]=dict(group_id=group_id,created_ns=folder.stat().st_ctime_ns)
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()

def configuration():
    data=json.loads((ROOT/'.setup/pear-alignment.json').read_text('utf8'));folder=(ROOT/data['folder']).resolve()
    if not folder.is_relative_to((ROOT/'runtime_pear').resolve()) or data['identity']!=identity() or data.get('base_identity')!=base_identity():raise ValueError('Camera alignment support needs install.cmd')
    for name,stamp in data['files'].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder):raise ValueError('Invalid alignment support path')
        st=p.stat()
        if [st.st_size,st.st_mtime_ns]!=stamp:raise ValueError('Camera alignment support changed; run install.cmd')
    if not (folder/'geocalib/__init__.py').is_file():raise ValueError('Missing GeoCalib support')
    return folder
def activate():
    folder=configuration();sys.dont_write_bytecode=True;sys.path.insert(0,str(folder));return folder

