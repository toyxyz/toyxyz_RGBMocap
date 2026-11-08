from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("RGB_MOCAP_DATA", ROOT)).resolve()
MODELS = HOME / "models"
PROJECTS = HOME / "projects"
UPSTREAM = ROOT / "third_party"
COMPAT = ROOT / "native_compat"
EXE = ".exe" if os.name == "nt" else ""
NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW; POSIX Popen rejects any flags
NATIVE_SUFFIX = ".pyd" if os.name == "nt" else ".so"
CORRECTION_BIN = UPSTREAM / "KimodoMotionCorrection/bin"


def worker_command(stage: str, *args: str) -> list[str]:
    if stage.startswith('prompthmr_') and stage not in ('prompthmr_prepare','prompthmr_publish'):
        from .prompthmr_runtime import command as phmr_command
        return phmr_command(stage,args)
    if stage.startswith('mamma_') and stage not in ('mamma_prepare','mamma_publish'):
        from .mamma_runtime import command as mamma_command
        return mamma_command(stage,args)
    from .runtime_registry import command
    return command(ROOT, stage, args, frozen=bool(getattr(sys, "frozen", False)))
