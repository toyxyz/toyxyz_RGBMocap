"""Launch the headless command line (mocap/cli.py) with the GUI launcher's environment.

It does not take the app lock, so it can run while the app is open; do not
process the same project from both at once.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
os.environ.pop('PYTHONHOME', None)
os.environ.pop('PYTHONPATH', None)
os.environ.update(RGB_MOCAP_MANAGED='1', RGB_MOCAP_DATA=str(root),
                  PYTHONDONTWRITEBYTECODE='1', HF_HOME=str(root / 'cache/huggingface'),
                  TORCH_HOME=str(root / 'cache/torch'), XDG_CACHE_HOME=str(root / 'cache'))
sys.dont_write_bytecode = True
if (root / '.setup/git-sync.pending').exists():
    raise SystemExit('Update was not completed. Run install.cmd to repair before running.')
try:
    state = json.loads((root / '.setup/git-runtime-state.json').read_text(encoding='utf8'))
    for key, path in [('lock_sha256', 'installer/runtime-lock.json'), ('registry_sha256', 'mocap/runtime_registry.py'),
                      ('support_sha256', 'installer/git-support-lock.json'), ('assets_sha256', 'installer/external-assets.json')]:
        if state[key] != hashlib.sha256((root / path).read_bytes()).hexdigest():
            raise ValueError('Installed dependencies do not match this source version.')
except (OSError, KeyError, ValueError) as exc:
    raise SystemExit('Run install.cmd before running: ' + str(exc)) from None
from mocap.runtime_registry import activate
activate(root, 'gui')
from mocap.runtime_limits import configure_preview_cpu
configure_preview_cpu()
from mocap.cli import main
raise SystemExit(main())
