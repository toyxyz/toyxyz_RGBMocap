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
    before = os.stat(source)
    os.link(source, target)
    if os.stat(source).st_mtime_ns != before.st_mtime_ns:
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_key(source: Path, start: float, end: float) -> str:
    payload = f"{source.resolve()}|{source.stat().st_size}|{source.stat().st_mtime_ns}|{start:.6f}|{end:.6f}|v1"
    return hashlib.sha256(payload.encode()).hexdigest()[:14]
