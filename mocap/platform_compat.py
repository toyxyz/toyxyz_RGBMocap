"""Cross-platform helpers shared by the GUI and its worker launchers.

Standard library only, so it is safe to import before any runtime is validated.
On Windows every value here equals what the code previously hard-coded, so
behaviour there is unchanged.

``mocap/runtime_registry.py`` is deliberately not touched by the Linux port:
it is loaded standalone by the Windows installer and its SHA-256 is pinned in
``installer/runtime-lock.json``. The POSIX layout lives here instead.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

IS_WINDOWS = os.name == 'nt'

# Hide child console windows on Windows. POSIX ``subprocess`` raises ValueError
# for any non-zero ``creationflags``, so this must be 0 everywhere else.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0

EXE_SUFFIX = '.exe' if IS_WINDOWS else ''


def exe_name(name: str) -> str:
    """``ffmpeg`` -> ``ffmpeg.exe`` on Windows, unchanged elsewhere."""
    return name + EXE_SUFFIX


def runtime_python(folder) -> Path:
    """Interpreter of a standalone (managed) runtime folder."""
    return Path(folder) / ('python.exe' if IS_WINDOWS else 'bin/python')


def venv_python(folder) -> Path:
    """Interpreter of a virtual environment."""
    return Path(folder) / ('Scripts/python.exe' if IS_WINDOWS else 'bin/python')


def site_packages(folder, python_version: str | None = None) -> Path:
    """The ``site-packages`` directory of a venv or runtime folder.

    On POSIX the path contains the Python minor version, so pass
    ``python_version`` (``'3.10'`` or ``'3.10.22'``) before the environment
    exists, or omit it to locate an existing one.
    """
    folder = Path(folder)
    if IS_WINDOWS:
        return folder / 'Lib/site-packages'
    if python_version:
        major, minor = str(python_version).split('.')[:2]
        return folder / 'lib' / f'python{major}.{minor}' / 'site-packages'
    found = sorted((folder / 'lib').glob('python3.*/site-packages'))
    if not found:
        raise FileNotFoundError(f'No site-packages found under {folder}')
    return found[-1]


def dev_worker_command(root, stage: str, args=()) -> list[str]:
    """Worker command for a source checkout: ``<profile venv>/python -m mocap.worker_progress``.

    Mirrors ``runtime_registry.command()`` in its development mode, using the
    per-profile environments (``.venv``, ``.venv-flow``, ...) that
    ``tools/setup_linux.py`` creates. Workers are started with the checkout as
    their working directory, so ``-m mocap...`` resolves.
    """
    from .runtime_registry import PROFILES, profile_for
    key = profile_for(stage, args)
    python = venv_python(Path(root) / PROFILES[key]['dev'])
    if not python.is_file():
        raise RuntimeError(f'The {key} environment is missing ({python}). '
                           f'Run: python tools/setup_linux.py --profiles {key}')
    return [str(python), '-X', 'utf8', '-m', 'mocap.worker_progress', stage, *map(str, args)]
