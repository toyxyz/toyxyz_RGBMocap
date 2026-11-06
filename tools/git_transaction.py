"""Journal directory promotions so interrupted installs can be rolled back."""
import base64,json,os,stat
from pathlib import Path

ALLOWED={'runtime','runtime_gui','runtime_flow','runtime_hand','runtime_htd','runtime_exmr',
         'runtime_character','tools/bin','third_party/KimodoMotionCorrection/bin'}

def store(path,value):
    tmp=path.with_name(path.name+'.partial');tmp.write_text(json.dumps(value),encoding='utf8');os.replace(tmp,path)

def inside(root,path):
    root=Path(root).resolve();candidate=Path(path).absolute()
    if candidate==root or not candidate.is_relative_to(root):raise ValueError('Transaction path leaves installation.')
    current=root
    for component in candidate.relative_to(root).parts:
        current=current/component
        if current.exists() and (current.is_symlink() or getattr(current.stat(),'st_file_attributes',0)&stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError('Transaction path contains a link.')
    resolved=candidate.resolve()
    if not resolved.is_relative_to(root):raise ValueError('Transaction path leaves installation.')
    return resolved

def begin(root):
    root=Path(root).resolve();folder=root/'.setup';backup=folder/'git-backup';backup.mkdir(parents=True,exist_ok=True)
    journal=folder/'git-promotion.json'
    if journal.exists() or any(backup.iterdir()):raise RuntimeError('Previous promotion requires recovery.')
    snapshots={}
    for name in ('SETUP_RECEIPT.json','.setup/git-runtime-state.json'):
        p=root/name;snapshots[name]=base64.b64encode(p.read_bytes()).decode() if p.is_file() else None
    value=dict(version=1,committed=False,moves=[],snapshots=snapshots)
    store(journal,value);return journal,value

def promote(root,journal,value,name,incoming):
    root=Path(root).resolve();folder=root/'.setup'
    if name not in ALLOWED:raise ValueError('Unexpected runtime path.')
    dest=inside(root,root/name);old=inside(folder,folder/'git-backup'/name);incoming=inside(folder,incoming)
    if not incoming.is_dir():raise ValueError('Staged runtime is missing.')
    entry=dict(name=name,incoming=incoming.relative_to(folder).as_posix(),had_old=dest.exists())
    value['moves'].append(entry);store(journal,value)
    dest.parent.mkdir(parents=True,exist_ok=True);old.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():os.replace(dest,old)
    os.replace(incoming,dest)

def recover(root,remove_owned):
    root=Path(root).resolve();folder=root/'.setup';journal=folder/'git-promotion.json';backup=folder/'git-backup'
    if not journal.exists():
        if backup.exists() and any(backup.iterdir()):raise RuntimeError('Unrecorded backup found; existing data was preserved.')
        return False
    value=json.loads(journal.read_text(encoding='utf8'))
    if value.get('version')!=1:raise ValueError('Unsupported transaction journal.')
    if not value.get('committed'):
        for entry in reversed(value['moves']):
            name=entry['name']
            if name not in ALLOWED:raise ValueError('Invalid recorded runtime path.')
            dest=inside(root,root/name);old=inside(folder,backup/name);incoming=inside(folder,folder/entry['incoming'])
            if old.exists():
                if dest.exists():
                    if incoming.exists():raise RuntimeError('Recovery found conflicting staged data; files were preserved.')
                    incoming.parent.mkdir(parents=True,exist_ok=True);os.replace(dest,incoming)
                dest.parent.mkdir(parents=True,exist_ok=True);os.replace(old,dest)
            elif entry['had_old']:
                if not dest.exists() or not incoming.exists():raise RuntimeError('Recovery cannot locate the previous runtime; files were preserved.')
            elif dest.exists():
                if incoming.exists():raise RuntimeError('Recovery found an unexpected destination; files were preserved.')
                incoming.parent.mkdir(parents=True,exist_ok=True);os.replace(dest,incoming)
        for name,encoded in value['snapshots'].items():
            if name not in ('SETUP_RECEIPT.json','.setup/git-runtime-state.json'):raise ValueError('Invalid receipt path.')
            path=inside(root,root/name)
            if encoded is None:path.unlink(missing_ok=True)
            else:
                temp=path.with_name(path.name+'.partial');temp.write_bytes(base64.b64decode(encoded,validate=True));os.replace(temp,path)
    if backup.exists():remove_owned(folder,backup)
    journal.unlink();return True
