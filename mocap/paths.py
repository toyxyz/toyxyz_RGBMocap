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


def worker_command(stage: str, *args: str) -> list[str]:
    if stage == 'sam31_people':
        from .sam31_runtime import command as sam31_command
        return sam31_command(args)
    if stage in ('gemx_reconstruct', 'gemx_correct'):
        from .gemx_runtime import command as gemx_command
        return gemx_command(stage, args)
    if stage.startswith('prompthmr_') and stage not in ('prompthmr_prepare','prompthmr_publish'):
        from .prompthmr_runtime import command as phmr_command
        return phmr_command(stage,args)
    if stage.startswith('mamma_') and stage not in ('mamma_prepare','mamma_publish'):
        from .mamma_runtime import command as mamma_command
        return mamma_command(stage,args)
    from .runtime_registry import command
    return command(ROOT, stage, args, frozen=bool(getattr(sys, "frozen", False)))
