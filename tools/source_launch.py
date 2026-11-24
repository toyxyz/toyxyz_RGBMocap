"""Initialise the same managed environment used by both RGB MoCap frontends."""
import hashlib
import json
import os
import sys


def configure(root):
    os.environ.pop('PYTHONHOME', None)
    os.environ.pop('PYTHONPATH', None)
    os.environ.update(RGB_MOCAP_MANAGED='1', RGB_MOCAP_DATA=str(root),
                      PYTHONDONTWRITEBYTECODE='1', HF_HOME=str(root/'cache/huggingface'),
                      TORCH_HOME=str(root/'cache/torch'), XDG_CACHE_HOME=str(root/'cache'))
    sys.dont_write_bytecode = True


def verify(root):
    if (root/'.setup/git-sync.pending').exists():
        raise RuntimeError('Update was not completed. Run install.cmd to repair before launching.')
    try:
        state = json.loads((root/'.setup/git-runtime-state.json').read_text(encoding='utf8'))
        for key, path in [('lock_sha256', 'installer/runtime-lock.json'), ('registry_sha256', 'mocap/runtime_registry.py'),
                          ('support_sha256', 'installer/git-support-lock.json'), ('assets_sha256', 'installer/external-assets.json')]:
            if state[key] != hashlib.sha256((root/path).read_bytes()).hexdigest():
                raise ValueError('Installed dependencies do not match this source version.')
    except (OSError, KeyError, ValueError) as error:
        raise RuntimeError('Run install.cmd before launching: ' + str(error)) from None
