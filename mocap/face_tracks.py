"""Use saved body and camera results to locate faces in published scenes."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .people_store import safe_child
from .result_artifacts import file_stat


@dataclass(frozen=True)
class FaceTake:
    first: int
    end: int
    project: Path
    trajectory: Path | None

    @property
    def joints(self):
        return self.project / 'initial_joints.npz'

    @property
    def camera(self):
        return self.project / 'scene_camera.json'

    def identity(self, root):
        paths = [self.joints, self.camera]
        if self.trajectory is not None:
            paths.append(self.trajectory)
        return [self.first, self.end, [(p.relative_to(root).as_posix(), file_stat(p)) for p in paths]]


def projected_takes(root, person):
    takes = []
    for row in person['takes']:
        project = safe_child(root, row['project'])
        camera = project / 'scene_camera.json'
        if not camera.is_file() or not (project / 'initial_joints.npz').is_file():
            continue
        record = json.loads(camera.read_text(encoding='utf-8'))
        trajectory = None
        if record.get('camera_track'):
            trajectory = (project / record['camera_track']['path']).resolve()
            if not trajectory.is_relative_to(root) or not trajectory.is_file():
                continue
        takes.append(FaceTake(row['first'], row['end'], project, trajectory))
    return tuple(takes)


def project_boxes(joints, matrices, intrinsics, width, height):
    # SMPL-X joints 0, 12 and 15 are the pelvis, neck and head. Use torso
    # length to size the crop even during crouching or when legs are occluded.
    pelvis, neck, head = joints[:, [0, 12, 15]].transpose(1, 0, 2)
    torso = np.linalg.norm(neck - pelvis, axis=1)
    position = np.einsum('nij,nj->ni', matrices[:, :3, :3], head) + matrices[:, :3, 3]
    depth = position[:, 2]
    pixels = position @ intrinsics.T
    uv = pixels[:, :2] / np.maximum(depth[:, None], .01)
    scale = 3.1 * torso * intrinsics[1, 1] / np.maximum(depth, .01)
    observed = ((depth > .05) & np.isfinite(uv).all(axis=1) & np.isfinite(scale)
                & (scale > 25) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    scale = np.clip(scale, 25., max(width, height) * 2.)
    x, y = uv.T
    boxes = np.stack((x - .19*scale, y - .132*scale, x + .19*scale, y + .868*scale), axis=1)
    return boxes, observed


def read_projected(takes, info):
    from .scene_camera import read_camera

    count = info['frames']
    boxes = np.zeros((count, 4), np.float32)
    observed = np.zeros(count, bool)
    for take in takes:
        first, end = take.first, take.end
        if not 0 <= first < end <= count or observed[first:end].any():
            raise ValueError('Invalid saved actor frame range for face capture.')
        camera = read_camera(take.project)
        if camera is None:
            continue
        if take.trajectory is None:
            matrices = np.broadcast_to(np.asarray(camera['world_to_camera']), (end-first, 4, 4))
        else:
            track = camera['camera_track']
            if (track['stat'] != file_stat(take.trajectory) or track['fps'] != info['fps']
                    or abs(track['start'] - info['start']) > 1e-6):
                continue
            with np.load(take.trajectory, allow_pickle=False) as archive:
                matrices = archive['world_to_camera'][first:end]
        with np.load(take.joints, allow_pickle=False) as archive:
            joints = archive['joints']
        if (matrices.shape != (end-first, 4, 4) or joints.shape != (end-first, 22, 3)
                or not np.isfinite(matrices).all()):
            raise ValueError('Saved joints or camera do not cover the face capture range.')
        boxes[first:end], observed[first:end] = project_boxes(
            joints, matrices, np.asarray(camera['K']), info['width'], info['height'])
    return boxes, observed
