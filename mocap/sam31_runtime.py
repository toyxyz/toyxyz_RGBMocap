"""Optional SAM 3.1 overlay on the pinned Windows ExMR runtime."""

import json
import sys

from .paths import ROOT

SUPPORTED = frozenset(('exmr', 'flowhmr', 'gvhmr'))
REVISION = 'f1647e8d0c1d280991f4d1b7d974c2704537fe66'
PROFILE_VERSION = 1


def location():
    folder = ROOT / 'runtime_sam31'
    try:
        receipt = json.loads((folder / 'runtime.json').read_text(encoding='utf-8'))
        if (receipt != {'version': PROFILE_VERSION, 'revision': REVISION}
                or not (folder / 'sam3/model_builder.py').is_file()):
            raise ValueError('SAM 3.1 runtime is out of date')
    except (OSError, ValueError) as error:
        raise RuntimeError('Close RGB MoCap and run Setup_SAM31.cmd to prepare SAM 3.1.') from error
    return folder


def activate(folder):
    sys.path.insert(0, str(folder))


def command(args):
    location()
    python = ROOT / 'runtime_exmr/python.exe'
    if not python.is_file():
        raise RuntimeError('The ExMR runtime is missing. Run install.cmd before Setup_SAM31.cmd.')
    return [str(python), '-I', '-X', 'utf8', '-B', '-u',
            str(ROOT / 'sam31_worker_boot.py'), *map(str, args)]
