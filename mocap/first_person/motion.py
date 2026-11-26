"""Read ViDiHand2 hand observations and the saved camera track."""
from zipfile import BadZipFile, ZipFile
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

HAND_PARENTS = (-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14,15,3,6,12,9)
TAKE_NAME = "vidihand_take.npz"
TILT_TAKE_NAME = ".vidihand_tilt_take.npz"
Y_UP = np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
ARM_LENGTHS = (.32,.28)
ARM_LENGTH_BOUNDS = (.05,1.)


def validate_arm_lengths(lengths):
    lengths = np.asarray(lengths)
    minimum, maximum = ARM_LENGTH_BOUNDS
    if (lengths.shape != (2,) or lengths.dtype.kind not in "fiu"
            or not np.isfinite(lengths).all() or (lengths < minimum).any()
            or (lengths > maximum).any()):
        raise ValueError("Enter an upper arm and forearm length from 5 to 100 cm each.")
    return tuple(map(float,lengths))


def arm_lengths(data):
    return validate_arm_lengths(data.get("arm_ik_lengths",ARM_LENGTHS))


def load(path):
    try:
        with ZipFile(path) as archive:
            if sum(row.file_size for row in archive.infolist()) > 512*1024**2:
                raise ValueError("First-person take exceeds the supported size.")
    except BadZipFile as exc:
        raise ValueError("The file is not a valid first-person take.") from exc
    with np.load(path, allow_pickle=False) as payload:
        version = payload["version"]
        if version.shape != () or version.dtype.kind not in "iu" or int(version) not in (1, 2):
            raise ValueError("Unsupported first-person take version.")
        # Version 1 already stores raw hands separately from the fitted arms.
        data = {k: payload[k] for k in ("hand_world", "hand_camera", "tracked", "K", "fps",
                                        "source_size", "video_size", "camera_mode")}
        for name in ("camera_to_world", "camera_position", "camera_tilt"):
            if name in payload or int(version) == 2:
                data[name] = payload[name]
        if "camera_alignment" in payload:
            data["camera_alignment"] = payload["camera_alignment"]
        if "arm_ik_world" in payload:
            data["arm_ik_world"] = payload["arm_ik_world"]
            if "arm_ik_lengths" in payload:
                data["arm_ik_lengths"] = payload["arm_ik_lengths"]
            if "arm_ik_auto_lengths" in payload:
                auto = payload["arm_ik_auto_lengths"]
                if auto.shape != () or auto.dtype.kind != "b":
                    raise ValueError("Invalid ViDiHand2 arm length mode.")
                data["arm_ik_auto_lengths"] = auto
    tracked = data["tracked"]
    if tracked.ndim != 2 or tracked.shape[1] != 2 or tracked.dtype != np.bool_ or not tracked.any():
        raise ValueError("Invalid take hand visibility.")
    frames = len(tracked)
    fps = data["fps"]
    if frames < 2 or fps.shape != () or not np.isfinite(fps) or float(fps) <= 0:
        raise ValueError("Invalid first-person take timing.")
    for key, shape in {"hand_world": (frames,2,21,3), "hand_camera": (frames,2,21,3),
                       "K": (frames,3,3)}.items():
        if data[key].shape != shape:
            raise ValueError(f"Invalid first-person take array: {key}.")
        values = data[key] if key == "K" else data[key][tracked]
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite take values: {key}.")
    for size in ("source_size", "video_size"):
        if data[size].shape != (2,) or not (data[size] > 0).all():
            raise ValueError("Invalid take image dimensions.")
    if "arm_ik_world" in data:
        arms = data["arm_ik_world"]
        if (arms.dtype.kind != "f" or arms.shape != (frames,2,3,3)
                or not {"camera_to_world","camera_position"} <= data.keys()):
            raise ValueError("Invalid ViDiHand2 arm track.")
        solved = np.isfinite(arms).all(axis=(2,3))
        if ((solved & ~tracked).any() or not np.isnan(arms[~solved]).all()
                or not np.allclose(arms[solved,2],data["hand_world"][solved,0],rtol=0.,atol=2.e-5)
                or not np.allclose(np.linalg.norm(np.diff(arms[solved],axis=1),axis=-1),
                                   arm_lengths(data),rtol=0.,atol=2.e-5)):
            raise ValueError("Arm poses do not match the tracked ViDiHand2 wrists.")
    return data


def save_take(path, data, expected):
    """Replace only the take generation the operation read."""
    from ..result_artifacts import file_stat

    path = Path(path)
    temporary = NamedTemporaryFile(dir=path.parent,prefix=path.stem+"-",suffix=".part",delete=False)
    part = Path(temporary.name)
    try:
        with temporary:
            np.savez_compressed(temporary,version=np.int32(2),**{"camera_tilt":0.,**data})
        if file_stat(path) != expected:
            raise ValueError("The ViDiHand2 take changed. Reopen it and retry.")
        os.replace(part,path)
    finally:
        part.unlink(missing_ok=True)
