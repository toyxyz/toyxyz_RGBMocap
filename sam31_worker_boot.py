"""Run SAM 3.1 using the base install's verified ExMR interpreter."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from mocap.runtime_registry import activate
from mocap.sam31_discovery import main
from mocap.worker_progress import emit

try:
    activate(root, 'exmr')
except (OSError, RuntimeError, ValueError) as error:
    emit('failed', stage='sam31_people', message=str(error))
    raise SystemExit(1) from None

raise SystemExit(main(sys.argv[1:], emit))
