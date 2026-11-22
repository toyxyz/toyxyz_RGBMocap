"""Derive installer/runtime-lock-macos.json from the Windows lock: same packages and versions, Apple Silicon wheels.

Maintainer tool; run it like tools/linux_lock.py, whose derivation it reuses. CUDA local versions
(torch 2.7.1+cu128) become the same release from PyPI, and wheels are chosen for macOS 14 or older on arm64.
"""
from __future__ import annotations
import sys
from pathlib import Path
from packaging.tags import mac_platforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import linux_lock as base

MINIMUM_MACOS = (14, 0)
base.PLATFORMS = list(mac_platforms(MINIMUM_MACOS, 'arm64'))
base.LEGACY = {}
base.BASE_ENV = {**base.BASE_ENV, 'sys_platform': 'darwin', 'platform_system': 'Darwin', 'platform_machine': 'arm64'}
_convert_group = base.convert_group


def convert_group(group, base_names):
    packages = [dict(p, version=p['version'].split('+')[0]) for p in group['packages']]
    return _convert_group(dict(group, packages=packages), base_names)


base.convert_group = convert_group

if __name__ == '__main__':
    if not any(arg.startswith('--output') for arg in sys.argv[1:]):
        sys.argv += ['--output', str(base.ROOT / 'installer/runtime-lock-macos.json')]
    sys.exit(base.main())
