"""Apply a rigid world transform to a saved ViDiHand2 take."""
from pathlib import Path

import numpy as np

from ..camera_tracks import valid_rotations
from ..result_artifacts import file_stat
from ..scene_camera import validate_transform
from .motion import Y_UP, load, save_take


def read_alignment(data):
    return validate_transform(data.get("camera_alignment", np.eye(4)))


def save_alignment(path, transform, expected):
    """Transform the track from its currently saved world frame."""
    path = Path(path)
    if file_stat(path) != expected:
        raise ValueError("The camera take changed. Reopen Adjust Camera.")
    transform = validate_transform(transform)
    data = load(path)
    rotation, position = data["camera_to_world"], data["camera_position"]
    count = len(data["tracked"])
    valid_rotations(rotation)
    if len(rotation) != count or position.shape != (count, 3) or not np.isfinite(position).all():
        raise ValueError("Invalid saved camera track.")
    rotation_delta = Y_UP.T @ transform[:3, :3] @ Y_UP
    translation_delta = Y_UP.T @ transform[:3, 3]
    rotation = rotation_delta @ rotation
    position = position @ rotation_delta.T + translation_delta
    data.update(camera_alignment=transform @ read_alignment(data),
                camera_to_world=rotation, camera_position=position)
    # Reproject the unchanged camera-space observations using the new poses.
    world = np.einsum("tij,tskj->tski", rotation, data["hand_camera"]) + position[:, None, None]
    data["hand_world"] = world.astype(np.float32)
    if "arm_ik_world" in data:
        arms = data["arm_ik_world"] @ rotation_delta.T + translation_delta
        data["arm_ik_world"] = arms.astype(np.float32)
    data.setdefault("camera_tilt", 0.)
    save_take(path,data,expected)
