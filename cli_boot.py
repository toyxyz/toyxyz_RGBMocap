"""Start the RGB MoCap command line using the installed Python."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from tools.source_launch import configure
configure(root)

from mocap.cli import main
raise SystemExit(main())
