"""Launch ViDiHand2 with the installed FlowHMR profile or an external runtime."""
import builtins
import importlib
from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / ".setup" / "first-person-arms.json"
SOURCE_URL = "https://github.com/NTUYWANG103/ViDiHand.git"
SOURCE_REVISION = "23cf80c292f2fb2db20caa1dee96516a63ee30a7"
NATIVE = """import os,runpy,sys
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("OMP_NUM_THREADS","4")
root=Path(sys.argv.pop(1))
sys.path.insert(0,str(root))
if sys.argv.pop(1) == "managed":
    from mocap.runtime_registry import activate
    from mocap.first_person.runtime import load_chumpy
    activate(root,"flow")
    load_chumpy()
runpy.run_module('mocap.first_person.worker',run_name='__main__')
"""
WSL = """import json,os,runpy,subprocess,sys
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("OMP_NUM_THREADS","4")
def translate(value):
    return subprocess.check_output(['/usr/bin/wslpath','-u',value],text=True,encoding='utf-8').strip()
sys.path.insert(0,translate(sys.argv.pop(1)))
for name in ('--video','--camera','--output','--cancel','--refit-from'):
    if name in sys.argv:
        at=sys.argv.index(name)+1
        sys.argv[at]=translate(sys.argv[at])
if '--managed-models' in sys.argv:
    at=sys.argv.index('--managed-models')+1
    sys.argv[at]=json.dumps({key:translate(path) for key,path in json.loads(sys.argv[at]).items()})
runpy.run_module('mocap.first_person.worker',run_name='__main__')
"""


def source_path(root):
    return root / ".setup" / "vidihand2-source" / SOURCE_REVISION


def source_ready(path):
    return all((path / name).is_file() for name in (
        "vidihand_v2/inference.py", "vidihand_v2/config.yaml", "utils/io.py", "utils/mano.py"))


def managed_settings(root):
    from ..runtime_registry import PROFILES

    profile = PROFILES["flow"]
    python = root / PROFILES[profile["base"]]["folder"] / "python.exe"
    if not python.is_file() or not (root / profile["folder"]).is_dir():
        raise FileNotFoundError("Run install.cmd to prepare the shared FlowHMR runtime for ViDiHand2.")
    upstream = source_path(root)
    if not source_ready(upstream):
        raise FileNotFoundError("Run Setup_ViDiHand2.cmd to prepare the ViDiHand2 source.")
    return {"mode": "managed", "python": str(python), "upstream": str(upstream)}


def load_chumpy():
    """Provide the legacy aliases only while MANO's pickle dependency initializes."""
    import numpy as np

    aliases = {name: getattr(builtins, name) for name in ("bool", "int", "float", "complex", "object", "str")}
    aliases["unicode"] = str
    added = aliases.keys() - vars(np).keys()
    try:
        for name in added:
            setattr(np, name, aliases[name])
        importlib.import_module("chumpy")
    finally:
        for name in added:
            delattr(np, name)


def load_settings(path=SETTINGS, *, root=ROOT):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return managed_settings(root)
    if not isinstance(data,dict):
        raise ValueError("Expected a ViDiHand2 runtime settings object.")
    mode = data.get("mode","native")
    if mode == "managed":
        return managed_settings(root)
    for name in ("python","upstream"):
        value = data.get(name)
        if not isinstance(value,str) or not value.strip():
            raise ValueError(f"Missing {name} in ViDiHand2 runtime settings.")
        if mode=="wsl" and not value.startswith("/"):
            raise ValueError(f"Use an absolute Linux path for {name} in WSL mode.")
        if mode=="native" and not Path(value).is_absolute():
            raise ValueError(f"Use an absolute path for {name}.")
    if mode=="wsl":
        name = data.get("distribution","")
        if not isinstance(name, str) or not re.fullmatch(r"[0-9A-Za-z_.-]{1,80}",name):
            raise ValueError("Invalid WSL distribution name.")
    elif mode!="native":
        raise ValueError("ViDiHand2 runtime mode must be native or wsl.")
    return data


def command(settings,video,camera,output,cancel,camera_start=0.,camera_tilt=0.,refit_from=None):
    mode = settings.get("mode", "native")
    wsl = mode == "wsl"
    prefix=(["wsl.exe","--distribution",settings["distribution"],"--exec",settings["python"]]
            if wsl else [settings["python"]])
    values=["-I","-X","utf8","-B","-c",WSL if wsl else NATIVE,str(ROOT)]
    if not wsl:
        values.append(mode)
    values+=["--upstream",settings["upstream"],"--video",str(video),
             "--output",str(output),"--cancel",str(cancel)]
    if camera is not None:
        values+=["--camera",str(camera),"--camera-start",str(camera_start)]
    if camera_tilt:
        values+=["--camera-tilt",str(camera_tilt)]
    if refit_from is not None:
        values+=["--refit-from",str(refit_from)]
    else:
        from ..models import BY_KEY, ready
        from .assets import KEYS

        managed = {key: str(BY_KEY[key].path) for key in KEYS if ready(BY_KEY[key])}
        if managed:
            values += ["--managed-models", json.dumps(managed)]
    return prefix+values
