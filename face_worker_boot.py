"""Track faces in the optional CPU overlay using the base ExMR Python."""
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from mocap.runtime_registry import activate
from mocap.worker_progress import emit

if __name__=='__main__':
    try:
        activate(root,'exmr')
    except (ValueError,OSError,RuntimeError) as error:
        emit('failed',stage='face_track',message=str(error))
        raise SystemExit(1) from None
    from mocap.face_capture import main
    raise SystemExit(main(sys.argv[1:],emit))
