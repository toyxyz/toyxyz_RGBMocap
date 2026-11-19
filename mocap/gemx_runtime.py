"""An isolated GEM-X package environment over the ExMR CPython installation."""

from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import re
import sys

from .paths import ROOT

LOCK = 'installer/gemx-runtime-lock.json'
SETUP_FILES = (LOCK, 'tools/setup_gemx.py', 'installer/verify_gemx.py')
NOTICE = 'Close RGB MoCap and run Setup_GEMX.cmd.'


@lru_cache(maxsize=8)
def identity(root, stamps):
    from .storage import sha256
    return hashlib.sha256(''.join(sha256(root / name) for name in SETUP_FILES).encode()).hexdigest()


def runtime_id(root):
    from .result_artifacts import file_stat
    return identity(root, tuple(tuple(file_stat(root / name)) for name in SETUP_FILES))


def configuration(root=ROOT):
    if os.name != 'nt':
        raise RuntimeError('GEM-X is available on Windows.')
    root = Path(root).resolve()
    try:
        record = json.loads((root / '.setup/gemx-runtime.json').read_text(encoding='utf-8'))
        generation = record['generation']
        if not re.fullmatch(r'[0-9a-f]{12}-[0-9a-f]{8}', generation) or record['identity'] != runtime_id(root):
            raise ValueError('Runtime generation changed.')
        parent = (root / 'runtime_gemx').resolve()
        folder = (parent / generation).resolve()
        py = folder / 'Scripts/python.exe'
        if parent.parent != root or folder.parent != parent or not py.is_file() or not (folder / 'engine/LICENSE').is_file():
            raise ValueError('Runtime files are missing.')
        base = root / 'runtime_exmr'
        venv = folder / 'pyvenv.cfg'
        if not (base / 'python.exe').is_file() or venv.stat().st_size > 16 * 1024:
            raise ValueError('ExMR CPython base is unavailable.')
        settings = {key.strip(): value.strip() for key, _, value in
                    (line.partition('=') for line in venv.read_text(encoding='utf-8').splitlines())}
        if (Path(settings['home']).resolve() != base.resolve()
                or settings['include-system-site-packages'] != 'false'):
            raise ValueError('The GEM-X environment must use this installation\'s ExMR Python in isolation.')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError('GEM-X runtime needs setup. ' + NOTICE) from error
    return {**record, 'python': py, 'engine': folder / 'engine', 'folder': folder}


def command(stage, args):
    cfg = configuration()
    return [str(cfg['python']), '-I', '-B', '-X', 'utf8', '-u',
            str(ROOT / 'gemx_worker_boot.py'), stage, *map(str, args)]


def activate():
    cfg = configuration()
    if Path(sys.executable).resolve() != cfg['python'].resolve():
        raise RuntimeError('GEM-X worker started with the wrong interpreter. ' + NOTICE)
    if sys.version_info[:3] != (3, 12, 10):
        raise RuntimeError('GEM-X requires the verified Python 3.12.10 base. ' + NOTICE)
    if Path(sys.base_prefix).resolve() != (ROOT / 'runtime_exmr').resolve():
        raise RuntimeError('GEM-X Python base has moved. ' + NOTICE)
    sys.path.insert(0, str(cfg['engine']))
    return cfg
