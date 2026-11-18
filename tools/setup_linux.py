#!/usr/bin/env python3
"""Create the Linux environments for a source checkout of RGB MoCap, using uv.

Standard library only; run it with any Python 3 (``python3 tools/setup_linux.py``).
It needs ``uv`` on PATH (Arch/CachyOS: ``sudo pacman -S uv``) and downloads the
exact Python versions the project pins, so the system Python does not matter.

It builds the development-style layout that ``mocap/platform_compat.py`` expects:

    .venv        shared profile: GUI + GVHMR and the base tools (Python 3.10)
    .venv-flow   FlowHMR overlay on .venv
    .venv-hand   HandFlow overlay on .venv
    .venv-htd    HTD-Refine overlay on .venv
    .venv-exmr   ExMR, its own full environment (Python 3.12)

Package pins come from ``requirements/runtime-<profile>.txt``, the same readable
pins the Windows lock was generated from. Differences from the Windows setup:

* torch/torchvision come from the PyTorch cu128 index *with* their dependencies
  (the Linux wheels need the nvidia-* CUDA libraries and triton); every other
  pin is installed with ``--no-deps`` so the pins stay authoritative.
* Windows-only pins are skipped, and the pure-Python source wheels that ship in
  ``installer/runtime_sources`` are rebuilt locally exactly as the Windows
  installer does.
* Nothing here is hash-verified. The Windows lock records wheel hashes for
  win_amd64 files only.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap.platform_compat import site_packages, venv_python  # noqa: E402  (stdlib only)
from mocap.runtime_registry import PROFILES  # noqa: E402  (pure data)

ORDER = ['shared', 'flow', 'hand', 'htd', 'exmr']
OVERLAYS = {key for key in ORDER if PROFILES[key]['base']}
TORCH_INDEX = 'https://download.pytorch.org/whl/cu128'
TORCH_FAMILY = {'torch', 'torchvision', 'torchaudio'}
WINDOWS_ONLY = {'win32-setctime'}
# GUI packages live in requirements-build.txt on Windows (the "gui" overlay).
GUI_PACKAGES = {'pyside6', 'pyside6-addons', 'pyside6-essentials', 'shiboken6', 'psutil'}
PIN = re.compile(r'^([A-Za-z0-9][A-Za-z0-9._-]*)==([^;\s]+)')


def norm(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def read_pins(path: Path) -> list[str]:
    pins = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            pins.append(line)
    return pins


def pin_name(pin: str) -> str | None:
    match = PIN.match(pin)
    return norm(match[1]) if match else None


def bundled_sources() -> dict[tuple[str, str], Path]:
    """Map (normalized name, version) -> unpacked wheel directory in installer/runtime_sources."""
    found = {}
    folder = ROOT / 'installer/runtime_sources'
    for source in sorted(folder.iterdir()) if folder.is_dir() else []:
        match = re.match(r'(.+?)-(\d[^-]*)-', source.name)
        if source.is_dir() and match:
            found[(norm(match[1]), match[2])] = source
    return found


def build_wheel(source: Path, destination: Path) -> Path:
    """Zip an unpacked wheel directory back into a wheel (same method as the Windows installer)."""
    wheel = destination / (source.name + '.whl')
    with zipfile.ZipFile(wheel, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob('*')):
            if not path.is_file() or path.name == 'SOURCE_WHEEL.json':
                continue
            info = zipfile.ZipInfo(path.relative_to(source).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return wheel


def plan_for(profile: str) -> dict:
    """Split a profile's pins into torch, locally built wheels and ordinary pins."""
    spec = PROFILES[profile]
    local = bundled_sources()
    torch, bundled, rest = [], [], []
    for pin in read_pins(ROOT / 'requirements' / f'runtime-{profile}.txt'):
        name = pin_name(pin)
        match = PIN.match(pin)
        if name in WINDOWS_ONLY:
            continue
        if name in TORCH_FAMILY:
            torch.append(pin)
        elif match and (name, match[2]) in local:
            bundled.append(local[(name, match[2])])
        else:
            rest.append(pin)
    if profile == 'shared':  # the Windows "gui" overlay: Qt and friends
        have = {pin_name(pin) for pin in rest}
        for pin in read_pins(ROOT / 'requirements-build.txt'):
            if pin_name(pin) in GUI_PACKAGES and pin_name(pin) not in have:
                rest.append(pin)
    return dict(profile=profile, python=spec['python'], overlay_of=spec['base'], folder=ROOT / spec['dev'],
                torch=torch, bundled=bundled, rest=rest)


def uv_environment() -> dict:
    env = os.environ.copy()
    env.update(UV_PYTHON_INSTALL_DIR=str(ROOT / '.runtime/python'), UV_CACHE_DIR=str(ROOT / '.cache/uv'),
               UV_NO_CONFIG='1')
    return env


def run(command, env) -> None:
    print('+', ' '.join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, env=env, check=True)


def install_profile(uv: str, plan: dict, env: dict) -> None:
    profile, python_version, folder = plan['profile'], plan['python'], plan['folder']
    python = venv_python(folder)
    run([uv, 'python', 'install', python_version, '--no-bin'], env)
    if not python.is_file():
        run([uv, 'venv', folder, '--python', python_version], env)
    install = [uv, 'pip', 'install', '--python', python]

    if plan['torch']:
        run([*install, '--extra-index-url', TORCH_INDEX, '--index-strategy', 'unsafe-best-match', *plan['torch']], env)
    with tempfile.TemporaryDirectory(prefix='rgbmocap-wheels-') as scratch:
        wheels = [build_wheel(source, Path(scratch)) for source in plan['bundled']]
        if wheels:
            run([*install, '--no-deps', *wheels], env)
    if plan['rest']:
        with tempfile.TemporaryDirectory(prefix='rgbmocap-pins-') as scratch:
            requirements = Path(scratch) / 'requirements.txt'
            requirements.write_text('\n'.join(plan['rest']) + '\n', encoding='utf-8')
            run([*install, '--no-deps', '-r', requirements], env)
    if plan['overlay_of']:
        base = ROOT / PROFILES[plan['overlay_of']]['dev']
        shared = site_packages(base, PROFILES[plan['overlay_of']]['python'])
        target = site_packages(folder, python_version)
        target.mkdir(parents=True, exist_ok=True)
        (target / 'shared_runtime.pth').write_text(f'import sys; sys.path.append({str(shared)!r})\n', encoding='ascii')
    print('READY', profile, flush=True)


def verify(plan: dict) -> bool:
    """Import torch in each environment; report, never raise."""
    python = venv_python(plan['folder'])
    code = ("import sys; import torch; "
            "print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'python', sys.version.split()[0])")
    result = subprocess.run([str(python), '-I', '-X', 'utf8', '-c', code], cwd=ROOT, capture_output=True,
                            text=True)
    ok = result.returncode == 0
    detail = result.stdout.strip() if ok else (result.stderr.strip().splitlines() or ['import failed'])[-1]
    print(f"  {'ok  ' if ok else 'FAIL'} {plan['profile']:<7} {detail}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--profiles', nargs='+', default=ORDER, choices=ORDER,
                        help='environments to create (default: all). Minimal start: shared exmr')
    parser.add_argument('--plan', action='store_true', help='show what would be installed and exit')
    parser.add_argument('--no-verify', action='store_true', help='skip the final torch import check')
    args = parser.parse_args()

    wanted = list(dict.fromkeys((['shared'] if OVERLAYS & set(args.profiles) else []) + args.profiles))
    wanted.sort(key=ORDER.index)
    plans = [plan_for(key) for key in wanted]

    if args.plan:
        for plan in plans:
            print(f"{plan['profile']:<7} python {plan['python']:<8} -> {plan['folder'].name:<11} "
                  f"torch={len(plan['torch'])} bundled={[p.name for p in plan['bundled']]} pins={len(plan['rest'])}"
                  + (f" overlay of {plan['overlay_of']}" if plan['overlay_of'] else ''))
        return 0

    if not sys.platform.startswith('linux'):
        print('This script is for Linux. On Windows use install.cmd.', file=sys.stderr)
        return 1
    uv = shutil.which('uv')
    if not uv:
        print('uv was not found on PATH. On Arch/CachyOS: sudo pacman -S uv', file=sys.stderr)
        return 1
    if not shutil.which('nvidia-smi'):
        print('Warning: nvidia-smi not found. The cu128 PyTorch builds need an NVIDIA driver.', file=sys.stderr)

    env = uv_environment()
    for plan in plans:
        install_profile(uv, plan, env)
    if not args.no_verify:
        print('Verification:')
        failed = [plan['profile'] for plan in plans if not verify(plan)]
        if failed:
            print('Some environments failed to import torch: ' + ', '.join(failed), file=sys.stderr)
            return 1
    print('Environments ready. Start the app with ./RGBMoCap.sh and import model files in Model Manager.')
    print('Not installed here: Blender 4.5 LTS (set RGBMOCAP_BLENDER) and the native motion-correction module.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
