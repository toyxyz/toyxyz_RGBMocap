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


def hard_link(source: Path, target: Path) -> None:
    """os.link that keeps the source's timestamps.

    Linux's NTFS driver stamps the file's mtime on link(); receipts, stage identities
    and project keys compare mtimes, so the original value is restored exactly.
    """
    if os.name == 'nt':
        os.link(source, target)
        return
    before = os.stat(source)
    os.link(source, target)
    try:
        if os.stat(target).st_mtime_ns != before.st_mtime_ns:
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    except OSError:
        target.unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def windows_long_paths_enabled():
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\FileSystem') as key:
            value, kind = winreg.QueryValueEx(key, 'LongPathsEnabled')
            return bool(value) if kind == winreg.REG_DWORD else None
    except FileNotFoundError:
        return False
    except OSError:
        return None


def path_error_message(error):
    """Return a displayable error, with context when Windows may hit MAX_PATH."""
    message = str(error).encode('utf-8', errors='backslashreplace').decode('utf-8')
    if os.name != 'nt' or not isinstance(error, OSError):
        return message
    winerror = getattr(error, 'winerror', None)
    hint = ' Use a shorter installation or project folder, or shorten the filename.'
    if winerror == 206:
        return 'Windows rejected a path as too long.' + hint + ' ' + message
    if winerror not in (2, 3, 123):
        return message

    def windows_units(value):
        return len(value.encode('utf-16-le', errors='surrogatepass')) // 2

    for name in (error.filename, error.filename2):
        if not isinstance(name, str) or name.startswith(('\\\\?\\', '\\\\.\\')):
            continue
        try:
            path = Path(os.path.abspath(name))
        except (OSError, ValueError):
            continue
        if winerror == 123:
            if any(windows_units(part) > 255 for part in path.parts[1:]):
                return 'A filename or folder name exceeds 255 UTF-16 characters.' + hint + ' ' + message
            continue
        length = windows_units(str(path))
        if length >= 260 and windows_long_paths_enabled() is False:
            return f'Windows long paths are disabled and this path has {length} UTF-16 characters.' + hint + ' ' + message
    return message


def project_key(source: Path, start: float, end: float) -> str:
    payload = f"{source.resolve()}|{source.stat().st_size}|{source.stat().st_mtime_ns}|{start:.6f}|{end:.6f}|v1"
    return hashlib.sha256(payload.encode()).hexdigest()[:14]
