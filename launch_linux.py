"""Launch RGB MoCap from this source checkout on Linux.

Uses the development-style environments created by ``tools/setup_linux.py``
(.venv for the GUI and shared models, .venv-flow/.venv-hand/.venv-htd overlays,
.venv-exmr for ExMR). Workers run as ``<env>/bin/python -m mocap.worker_progress``.

The Windows launcher (launch_source.py) is not used: it requires the managed
Windows installer state and the exact pinned ``runtime*`` folder layout.
"""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

if not sys.platform.startswith('linux'):
    raise SystemExit('launch_linux.py is for Linux. On Windows use RGBMoCap.cmd.')
if Path(sys.prefix).resolve() != (root / '.venv').resolve():
    raise SystemExit('Start the app with ./RGBMoCap.sh (it uses .venv). '
                     'Prepare the environments first: python3 tools/setup_linux.py')

os.environ.pop('PYTHONHOME', None)
os.environ.pop('PYTHONPATH', None)
os.environ.pop('RGB_MOCAP_MANAGED', None)  # use the per-profile .venv* layout, not runtime*/
# Models, projects and caches live under RGB_MOCAP_DATA (default: this folder), as on Windows.
data = Path(os.environ.setdefault('RGB_MOCAP_DATA', str(root))).resolve()
os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HOME=str(data / 'cache/huggingface'),
                  TORCH_HOME=str(data / 'cache/torch'), XDG_CACHE_HOME=str(data / 'cache'))
sys.dont_write_bytecode = True
# On Wayland Qt uses EGL, where the viewport's OpenGL 4.3 core context can fail with
# EGL_BAD_MATCH (error 3009). X11/XWayland uses GLX. Override: QT_QPA_PLATFORM=wayland
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

from app import main  # noqa: E402

raise SystemExit(main())
