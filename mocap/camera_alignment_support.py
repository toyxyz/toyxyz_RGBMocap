"""GeoCalib support over the pinned Flow runtime for MAMMA."""
import hashlib
import json
import sys

from .paths import ROOT

RUNTIME = 'runtime_camera_alignment'
RECEIPT = '.setup/camera-alignment.json'
# Released camera support was stored alongside the former live runtime.
# Accept only its original lock/setup identity until a base runtime changes.
LEGACY_IDENTITY = 'db915e6bf500358716d334ee8c7f41f389f986c884494088406094327f207513'


def identity():
    lock = ROOT / 'installer/camera-alignment.json'
    setup = ROOT / 'tools/setup_camera_alignment.py'
    return hashlib.sha256(lock.read_bytes() + setup.read_bytes()).hexdigest()


def base_identity():
    receipt = json.loads((ROOT / 'SETUP_RECEIPT.json').read_text(encoding='utf-8'))
    rows = {}
    for key, name in (('shared', 'runtime'), ('flow', 'runtime_flow')):
        group_id = receipt['group_ids'][key]
        if not isinstance(group_id, str) or not group_id:
            raise ValueError('Missing camera alignment base runtime receipt.')
        folder = ROOT / name
        if folder.is_symlink() or not folder.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Invalid camera alignment base path.')
        rows[key] = dict(group_id=group_id, created_ns=folder.stat().st_ctime_ns)
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def configuration():
    base = base_identity()
    for receipt, runtime, expected in (
        (RECEIPT, RUNTIME, identity()),
        ('.setup/pear-alignment.json', 'runtime_pear', LEGACY_IDENTITY),
    ):
        try:
            data = json.loads((ROOT / receipt).read_text(encoding='utf-8'))
            parent = ROOT / runtime
            folder = (ROOT / data['folder']).resolve()
            if (parent.is_symlink() or parent.resolve().parent != ROOT.resolve()
                    or folder.parent != parent.resolve() or not folder.name.startswith('alignment-')
                    or data['identity'] != expected or data.get('base_identity') != base):
                continue
            files = data['files']
            if not {'geocalib/__init__.py', 'PROVENANCE.json'} <= files.keys():
                continue
            for name, stamp in files.items():
                path = (folder / name).resolve()
                if not path.is_relative_to(folder):
                    raise ValueError('Invalid camera alignment support path.')
                info = path.stat()
                if [info.st_size, info.st_mtime_ns] != stamp:
                    raise ValueError('Camera alignment support files changed.')
            return folder
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    raise ValueError('Camera alignment support needs install.cmd.')


def activate():
    folder = configuration()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(folder))
    return folder
