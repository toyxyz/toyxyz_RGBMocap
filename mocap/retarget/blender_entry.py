"""Isolated Blender bridge; invoked with --disable-autoexec and no user startup."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')

if __name__ == '__main__':
    from mocap.retarget.blender_io import main
    main(sys.argv[sys.argv.index('--')+1:])
