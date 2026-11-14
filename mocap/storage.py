from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        # NTFS readers and antivirus can briefly hold a just-replaced name.
        # Bound retries to 31 ms; permanent access failures still propagate.
        for attempt in range(6):
            try:
                os.replace(temporary,path)
                break
            except PermissionError as error:
                if os.name!='nt' or getattr(error,'winerror',None) not in (5,32,33) or attempt==5:raise
                time.sleep(.001*(2**attempt))
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_key(source: Path, start: float, end: float) -> str:
    payload = f"{source.resolve()}|{source.stat().st_size}|{source.stat().st_mtime_ns}|{start:.6f}|{end:.6f}|v1"
    return hashlib.sha256(payload.encode()).hexdigest()[:14]


def path_message(error: BaseException) -> str:
    """Name the Windows path limit instead of a misleading missing-file error."""
    names = [os.fspath(p) for p in (getattr(error, 'filename', None), getattr(error, 'filename2', None)) if isinstance(p, (str, os.PathLike))]
    name = max(names, key=len, default='')
    # Files fail from 260 characters; WinError 206 also covers folders over 247.
    if os.name!='nt' or not isinstance(error, OSError) or not name or len(name)<260 and getattr(error,'winerror',None)!=206:return str(error)
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\FileSystem') as key:
            if winreg.QueryValueEx(key, 'LongPathsEnabled')[0]:return str(error)
    except FileNotFoundError:pass  # An absent value is the Windows default: disabled.
    except OSError:return str(error)
    return (f'The path is too long for Windows ({len(name)} characters; long paths are disabled). Use a shorter installation '
            f'folder such as C:\\RGBMoCap or save the project to a shorter folder, or enable Windows long paths. Path: {name}')
