"""Commit a ViDiHand2 bone-only take and its camera settings."""
import argparse
import json
import os
from pathlib import Path
import time

from .motion import TAKE_NAME, TILT_TAKE_NAME, load
from ..storage import atomic_json


def publish(project, refit=False):
    manifest = project/"project.json"
    state = json.loads(manifest.read_text(encoding="utf-8"))
    if state.get("pipeline", {}).get("model") != "vidihand2":
        raise ValueError("Select a ViDiHand2 project before preparing its preview.")
    take = project/(TILT_TAKE_NAME if refit else TAKE_NAME)
    data = load(take)
    if len(data["tracked"]) != state["frames"] or abs(float(data["fps"]) - state["fps"]) > .1:
        raise ValueError("Hand tracks do not match the prepared clip.")
    mode = {"fixed_estimated": "fixed", "automatic": "automatic"}.get(str(data["camera_mode"]), "manual")
    tilt = int(data.get("camera_tilt", state.get("vidihand_camera_tilt", 0)))
    if refit:
        os.replace(take, project/TAKE_NAME)
    state.update(vidihand_camera_mode=mode, vidihand_camera_tilt=tilt)
    atomic_json(manifest, state)


def main(arguments, emit):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--apply-tilt", action="store_true")
    args = parser.parse_args(arguments)
    start = time.monotonic()
    try:
        publish(args.project, args.apply_tilt)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        emit("failed", stage="vidihand_publish", message=str(error))
        return 1
    emit("complete", stage="vidihand_publish", elapsed=time.monotonic()-start, output=str(args.project))
    return 0
