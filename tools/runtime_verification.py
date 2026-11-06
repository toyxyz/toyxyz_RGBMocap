"""Bounded metadata reuse of a successful full runtime verification.

This is change detection for ordinary update/retry, not an authenticity check.
--verify-all always runs the existing RECORD hashes and import checks.
"""
from pathlib import Path
import hashlib
import json
import os
import stat
import time

MAX_AGE_SECONDS = 7 * 24 * 60 * 60
MAX_FILES = 250000
MAX_CACHE_BYTES = 64 * 1024 * 1024


def inventory(root, target):
    root = Path(root).resolve()
    if target not in {'runtime', 'runtime_gui', 'runtime_flow', 'runtime_hand', 'runtime_htd', 'runtime_exmr'}:
        raise ValueError('Unexpected runtime inventory target')
    folder = root / target
    if not folder.is_dir():
        raise ValueError('Runtime directory is missing: ' + target)
    if folder.is_symlink() or getattr(folder.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError('Runtime directory is a link: ' + target)
    rows = {}
    def unreadable(exc):
        raise exc
    for current, dirs, files in os.walk(folder, onerror=unreadable):
        base = Path(current)
        for name in [*dirs, *files]:
            path = base / name
            info = path.lstat()
            if path.is_symlink() or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError('Runtime inventory contains a link: ' + target)
        dirs[:] = sorted(name for name in dirs if name != '__pycache__')
        for name in sorted(files):
            path = base / name
            if path.suffix.lower() in ('.pyc', '.pyo'):
                continue
            info = path.stat()
            rows[path.relative_to(folder).as_posix()] = [info.st_size, info.st_mtime_ns, info.st_ctime_ns]
            if len(rows) > MAX_FILES:
                raise ValueError('Runtime inventory exceeds the supported file count')
    if not rows:
        raise ValueError('Runtime inventory is empty: ' + target)
    return rows


def verifier_identity(root):
    return hashlib.sha256(b''.join((Path(root) / name).read_bytes() for name in
        ('installer/engine.py', 'installer/verify_runtime.py', 'mocap/runtime_registry.py',
         'tools/runtime_verification.py'))).hexdigest()


def reuse(root, lock, ids, force=False):
    """Return unchanged fully verified groups; missing/old caches require full checks."""
    if force:
        return {}, {}
    root = Path(root)
    try:
        receipt = json.loads((root / 'SETUP_RECEIPT.json').read_text('utf8'))
        verifier = verifier_identity(root)
    except (OSError, ValueError, TypeError):
        return {}, {}
    checked, records = {}, {}
    for group in lock['groups']:
        key = group['key']
        if group.get('base') and group['base'] not in checked:
            continue
        try:
            path = root / '.setup/runtime-verification' / (key + '.json')
            if path.stat().st_size > MAX_CACHE_BYTES:
                continue
            row = json.loads(path.read_text('utf8'))
            age = time.time() - row['verified_at']
            if (row['version'] != 1 or row['verifier'] != verifier or row['group_id'] != ids[key]
                    or receipt['group_ids'][key] != ids[key] or not 0 <= age < MAX_AGE_SECONDS
                    or not isinstance(row['report'], dict) or not row['report']
                    or row['report'] != receipt['runtimes'][key]
                    or not isinstance(row['files'], dict) or not row['files']
                    or inventory(root, group['target']) != row['files']):
                continue
            checked[key], records[key] = row['report'], row
            print('Using recent full verification and unchanged file metadata: ' + key, flush=True)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return checked, records


def save(root, lock, ids, reports, retained):
    root = Path(root)
    folder = root / '.setup/runtime-verification'
    folder.mkdir(parents=True, exist_ok=True)
    verifier = verifier_identity(root)
    for group in lock['groups']:
        key = group['key']
        # A fast reuse must never extend the full-verification expiry.
        if key in retained:
            continue
        row = dict(version=1, verifier=verifier, group_id=ids[key], verified_at=time.time(),
                   report=reports[key], files=inventory(root, group['target']))
        path = folder / (key + '.json')
        temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps(row, separators=(',', ':')), encoding='utf8')
        os.replace(temporary, path)
