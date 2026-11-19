"""Install a pinned, optional SAM overlay without changing base packages."""

import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap.sam31_runtime import PROFILE_VERSION, REVISION, location
from tools.session_lock import app_lock

PACKAGES = (
    'timm==1.0.30 --hash=sha256:c8e27adf6801a2fdfb570dbbdd171f2f084756eefc673df8cfaab06bd14864f3',
    'ftfy==6.1.1 --hash=sha256:0ffd33fce16b54cccaec78d6ec73d95ad370e5df5a25255c8966a6147bd667ca',
    'regex==2026.9.29 --hash=sha256:fdd88ed5e20b1bcdd234421e454962c971aa44b653bdb7f1ea9ef683e90fb649',
    'wcwidth==0.9.4 --hash=sha256:901eace6d2dd25ebb917d0ce5db18e5c537e1504c905cf32f0d006c12098b05d',
    'pycocotools==2.0.11 --hash=sha256:ffe806ce535f5996445188f9a35643791dc54beabc61bd81e2b03367356d604f',
)


def verify(folder):
    code = r'''
import sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from pathlib import Path
import torch, torchvision
from mocap.sam31_engine import windows_builder
model_builder = windows_builder()
from torch.nn import functional as F
if not torch.cuda.is_bf16_supported(including_emulation=False):
    raise RuntimeError("SAM 3.1 requires native CUDA bfloat16 support.")
x = torch.randn((1, 4, 32, 32), device="cuda", dtype=torch.bfloat16)
assert F.scaled_dot_product_attention(x, x, x).isfinite().all().item()
assert Path(model_builder.__file__).is_relative_to(Path(sys.argv[1]))
'''
    subprocess.run([sys.executable, '-I', '-X', 'utf8', '-B', '-c', code, str(folder), str(ROOT)],
                   check=True, timeout=180)


def install():
    import os
    from mocap.runtime_registry import activate
    activate(ROOT, 'exmr')
    try:
        existing = location()
    except RuntimeError:
        pass
    else:
        verify(existing)
        print('Verified SAM 3.1 runtime retained.')
        return
    with tempfile.TemporaryDirectory(prefix='.sam31-', dir=ROOT / '.setup') as name:
        stage = Path(name)
        overlay = stage / 'overlay'
        checkout = stage / 'upstream'
        checkout.mkdir()
        subprocess.run(['git', '-C', str(checkout), 'init', '-q'], check=True, timeout=30)
        git = ['git', '-C', str(checkout)]
        subprocess.run([*git, 'fetch', '--depth=1', 'https://github.com/facebookresearch/sam3.git', REVISION],
                       check=True, timeout=300)
        fetched = subprocess.check_output([*git, 'rev-parse', 'FETCH_HEAD'], text=True, timeout=30).strip()
        if fetched != REVISION:
            raise RuntimeError('Unexpected SAM 3.1 source revision.')
        subprocess.run([*git, 'checkout', '--detach', REVISION], check=True, timeout=60)
        requirements = stage / 'requirements.txt'
        requirements.write_text('\n'.join(PACKAGES) + '\n', encoding='utf-8')
        env = dict(os.environ, PIP_CONFIG_FILE=os.devnull)
        subprocess.run([sys.executable, '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'install',
                        '--no-deps', '--no-cache-dir', '--require-hashes', '--only-binary=:all:',
                        '--index-url', 'https://pypi.org/simple', '--target', str(overlay),
                        '-r', str(requirements)], check=True, env=env, timeout=300)
        shutil.copytree(checkout / 'sam3', overlay / 'sam3')
        shutil.copy2(checkout / 'LICENSE', overlay / 'SAM_LICENSE')
        verify(overlay)
        (overlay / 'runtime.json').write_text(json.dumps(dict(version=PROFILE_VERSION, revision=REVISION)),
                                              encoding='utf-8')
        destination = ROOT / 'runtime_sam31'
        backup = None
        if destination.exists():
            backup = ROOT / '.setup' / ('sam31-backup-' + uuid.uuid4().hex)
            destination.rename(backup)
        try:
            overlay.rename(destination)
        except OSError:
            if backup is not None:
                backup.rename(destination)
            raise
    print('SAM 3.1 runtime ready. Get the optional checkpoint in Model Manager.')


if __name__ == '__main__':
    try:
        with app_lock(ROOT):
            install()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(f'SAM 3.1 setup failed: {error}')
