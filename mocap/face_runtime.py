"""Verified, optional CPU Face Landmarker overlay for the base ExMR interpreter."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from .paths import HOME, ROOT

MODEL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
         '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff')
CANONICAL = ('https://raw.githubusercontent.com/google-ai-edge/mediapipe/v0.10.32/mediapipe/modules/face_geometry/data/canonical_face_model.obj',
             '8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618')
LOCK = ROOT / 'installer/face-windows.lock'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def command(args):
    python = ROOT / 'runtime_exmr/python.exe'
    if not python.is_file():
        raise RuntimeError('Run install.cmd to prepare the base runtime for Track Face.')
    return [str(python), '-B', '-I', '-X', 'utf8', '-u', str(ROOT/'face_worker_boot.py'), *map(str, args)]


def setup(emit):
    base = HOME / 'cache/face_runtime'
    cache = base / sha(LOCK.read_bytes())[:16]

    def ready():
        return ((cache/'packages/mediapipe').is_dir() and all(
            (cache/name).is_file() and sha((cache/name).read_bytes()) == spec[1]
            for name, spec in [('face_landmarker.task', MODEL), ('canonical.obj', CANONICAL)]))

    if not ready():
        emit('progress', current=0, total=0, message='Preparing the optional Face Landmarker CPU runtime')
        base.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='prepare-', dir=base) as term:
            staging = Path(term)/'runtime'
            staging.mkdir()
            subprocess.run([sys.executable, '-B', '-I', '-m', 'pip', 'install',
                '--disable-pip-version-check', '--no-input', '--no-cache-dir', '--only-binary=:all:',
                '--index-url', 'https://pypi.org/simple', '--require-hashes',
                '--target', str(staging/'packages'), '-r', str(LOCK)], check=True,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            for name, (url, expected) in [('face_landmarker.task', MODEL), ('canonical.obj', CANONICAL)]:
                with urlopen(url, timeout=40) as response:
                    data = response.read(8*1024**2)
                if sha(data) != expected:
                    raise ValueError('MediaPipe file verification failed: '+name)
                (staging/name).write_bytes(data)
            try:
                staging.rename(cache)
            except OSError:
                if not ready():
                    raise
    os.environ.setdefault('MPLCONFIGDIR', str(base/'matplotlib'))
    sys.path.insert(0, str(cache/'packages'))
    return cache
