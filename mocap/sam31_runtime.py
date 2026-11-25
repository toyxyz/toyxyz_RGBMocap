"""Optional SAM 3.1 overlay on the pinned ExMR runtime."""

import json
import sys

from .paths import ROOT
from .runtime_registry import INSTALL_COMMAND, WINDOWS, managed_python

SUPPORTED = frozenset(('exmr', 'flowhmr', 'gvhmr'))
REVISION = 'f1647e8d0c1d280991f4d1b7d974c2704537fe66'
PROFILE_VERSION = 1
SETUP_COMMAND = 'Setup_SAM31.cmd' if WINDOWS else './setup_sam31.sh'


def runtime_receipt():
    receipt = {'version': PROFILE_VERSION, 'revision': REVISION}
    if not WINDOWS:
        receipt['platform'] = 'linux-x86_64'
    return receipt


def location():
    folder = ROOT / 'runtime_sam31'
    try:
        receipt = json.loads((folder / 'runtime.json').read_text(encoding='utf-8'))
        if (receipt != runtime_receipt()
                or not (folder / 'sam3/model_builder.py').is_file()):
            raise ValueError('SAM 3.1 runtime is out of date')
    except (OSError, ValueError) as error:
        raise RuntimeError(f'Close RGB MoCap and run {SETUP_COMMAND} to prepare SAM 3.1.') from error
    return folder


def activate(folder):
    sys.path.insert(0, str(folder))


def command(args):
    location()
    python = managed_python(ROOT, 'exmr')
    if not python.is_file():
        raise RuntimeError(f'The ExMR runtime is missing. Run {INSTALL_COMMAND} before {SETUP_COMMAND}.')
    return [str(python), '-I', '-X', 'utf8', '-B', '-u',
            str(ROOT / 'sam31_worker_boot.py'), *map(str, args)]
