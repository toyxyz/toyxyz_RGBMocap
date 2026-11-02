"""Private optional FlowHMR HaMeR entry point in the existing ExMR runtime."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from mocap.runtime_registry import activate
activate(root,'exmr')
from mocap.flow_hamer import run
def emit(kind,**value):print('@@FLOW_HAND@@'+json.dumps(dict(type=kind,**value)),flush=True)
if len(sys.argv)!=2:raise ValueError('An explicit FlowHMR project path is required')
run(Path(sys.argv[1]),emit)
