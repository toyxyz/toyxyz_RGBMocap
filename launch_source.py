"""Launch editable application code using the private common runtime."""
import os
import hashlib
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
os.environ.pop('PYTHONHOME',None)
os.environ.pop('PYTHONPATH',None)
os.environ.update(RGB_MOCAP_MANAGED='1',RGB_MOCAP_DATA=str(root),
                  PYTHONDONTWRITEBYTECODE='1',HF_HOME=str(root/'cache/huggingface'),
                  TORCH_HOME=str(root/'cache/torch'),XDG_CACHE_HOME=str(root/'cache'))
sys.dont_write_bytecode=True
from tools.session_lock import app_lock
from mocap.runtime_registry import INSTALL_COMMAND, LOCK, SUPPORT_LOCK
with app_lock(root):
    if (root/'.setup/git-sync.pending').exists():
        raise SystemExit('Update was not completed. Run '+INSTALL_COMMAND+' to repair before launching.')
    try:
        state=json.loads((root/'.setup/git-runtime-state.json').read_text(encoding='utf8'))
        for key,path in [('lock_sha256',LOCK),('registry_sha256','mocap/runtime_registry.py'),('support_sha256',SUPPORT_LOCK),('assets_sha256','installer/external-assets.json')]:
            if state[key]!=hashlib.sha256((root/path).read_bytes()).hexdigest():
                raise ValueError('Installed dependencies do not match this source version.')
    except (OSError,KeyError,ValueError) as exc:
        raise SystemExit('Run '+INSTALL_COMMAND+' before launching: '+str(exc)) from None
    from mocap.runtime_registry import activate
    activate(root,'gui')
    from app import main
    raise SystemExit(main())

