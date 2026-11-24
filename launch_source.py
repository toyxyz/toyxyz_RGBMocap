"""Launch editable application code using the private common runtime."""
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from tools.source_launch import configure, verify
configure(root)
from tools.session_lock import app_lock
with app_lock(root):
    try:
        verify(root)
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    from mocap.runtime_registry import activate
    activate(root,'gui')
    from app import main
    raise SystemExit(main())
