"""Launch a separately installed ViDiHand2 worker without changing base runtimes."""
from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / ".setup" / "first-person-arms.json"
NATIVE = """import os,runpy,sys
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("OMP_NUM_THREADS","4")
sys.path.insert(0,sys.argv.pop(1))
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


def load_settings(path=SETTINGS):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data,dict):
        raise ValueError("Expected a ViDiHand2 runtime settings object.")
    mode = data.get("mode","native")
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
    wsl=settings.get("mode")=="wsl"
    prefix=(["wsl.exe","--distribution",settings["distribution"],"--exec",settings["python"]]
            if wsl else [settings["python"]])
    values=["-I","-X","utf8","-c",WSL if wsl else NATIVE,str(ROOT),"--upstream",settings["upstream"],
            "--video",str(video),"--output",str(output),"--cancel",str(cancel)]
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
