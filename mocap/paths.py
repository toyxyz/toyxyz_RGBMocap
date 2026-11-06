from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("RGB_MOCAP_DATA", ROOT)).resolve()
PROJECTS = HOME / "projects"
UPSTREAM = ROOT / "third_party"
COMPAT = ROOT / "native_compat"
EXE = ".exe" if os.name == "nt" else ""
NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW; POSIX Popen rejects any flags
NATIVE_SUFFIX = ".pyd" if os.name == "nt" else ".so"
CORRECTION_BIN = UPSTREAM / "KimodoMotionCorrection/bin"


def model_locations(home: Path) -> tuple[Path, dict[str, Path]]:
    """Models folder and per-key file locations from optional config/local.json.

    `models_dir` replaces <data>/models; `model_files` maps a model key to an
    existing file used in place. Malformed values are skipped with a warning.
    """
    from .resource_safety import read_settings
    try:
        settings = read_settings(home / "config/local.json")
    except (OSError, ValueError) as exc:
        print("RGB MoCap: ignoring model locations in config/local.json:", exc, file=sys.stderr)
        settings = {}

    def absolute(name, value):
        if isinstance(value, str) and value and value.strip() == value and Path(value).is_absolute():
            return Path(value)
        print(f"RGB MoCap: ignoring {name} in config/local.json; expected an absolute path.", file=sys.stderr)
        return None

    models = absolute("models_dir", settings["models_dir"]) if "models_dir" in settings else None
    files = settings.get("model_files", {})
    if not isinstance(files, dict):
        print("RGB MoCap: ignoring model_files in config/local.json; expected an object.", file=sys.stderr)
        files = {}
    located = {key: path for key, value in files.items() if (path := absolute("model_files." + str(key), value))}
    return models or home / "models", located


MODELS, MODEL_FILES = model_locations(HOME)


def reload_model_files() -> None:
    """Re-read per-key locations; MODEL_FILES is updated in place because every importer shares it."""
    MODEL_FILES.clear()
    MODEL_FILES.update(model_locations(HOME)[1])


def set_model_file(key: str, path: Path | None) -> None:
    """Remember `path` as the in-place file for `key` (None forgets it) in config/local.json."""
    import json
    from .resource_safety import MAX_BYTES, read_settings
    from .storage import atomic_json
    target = HOME / "config/local.json"
    settings = read_settings(target)  # A malformed file raises and is left untouched.
    files = settings.get("model_files", {})
    if not isinstance(files, dict):
        raise ValueError("config/local.json: model_files must be an object; fix it before changing model locations.")
    value = None if path is None else str(Path(path).resolve())
    if files.get(key) != value:
        if value is None:files.pop(key, None)
        else:files[key] = value
        settings["model_files"] = files
        if len(json.dumps(settings, ensure_ascii=False, indent=2).encode("utf-8")) > MAX_BYTES:
            raise ValueError("Local settings exceed the size limit.")
        atomic_json(target, settings)
    reload_model_files()


def worker_command(stage: str, *args: str) -> list[str]:
    if stage.startswith('prompthmr_') and stage not in ('prompthmr_prepare','prompthmr_publish'):
        from .prompthmr_runtime import command as phmr_command
        return phmr_command(stage,args)
    if stage.startswith('mamma_') and stage not in ('mamma_prepare','mamma_publish'):
        from .mamma_runtime import command as mamma_command
        return mamma_command(stage,args)
    from .runtime_registry import command
    return command(ROOT, stage, args, frozen=bool(getattr(sys, "frozen", False)))
