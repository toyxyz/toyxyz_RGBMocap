"""Prepare stock mannequin transforms for the Blender FBX writer."""
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from .blend_support import build
from .scene_camera import validate_transform
from .ue5_retarget import _world, retarget

# Blender's Z-up right-handed coordinates differ from Unreal by a Y reflection.
_UNREAL_TO_BLENDER = np.diag([1., -1., 1.])


def write(path, rig, positions, rotations, fps, emit, *, skin=None):
    if not np.isfinite(fps) or not 0.1 <= fps <= 1000.:
        raise ValueError('UE5 Mannequin export requires a valid source frame rate.')
    work = Path(path).parent
    basis = _UNREAL_TO_BLENDER
    world_p, world_r = _world(rig)
    root_basis = world_r[0].T
    rest_positions = (world_p - world_p[0]) @ root_basis.T
    rest_rotations = root_basis @ world_r
    # Matrix basis relative to each edit bone. The Blender armature object
    # carries the stock "root" track, so it does not add another skeleton node.
    inverse = rig.rotations.swapaxes(-1, -2)
    local_p = np.einsum('bij,fbj->fbi', inverse, positions - rig.positions)
    local_r = inverse @ rotations
    local_p[:, 0], local_r[:, 0] = positions[:, 0], rotations[:, 0]
    local_p = local_p @ basis
    local_p[:, 0] *= .01
    local_r = basis @ local_r @ basis
    quaternions = Rotation.from_matrix(local_r.reshape(-1, 3, 3)).as_quat().reshape(*local_p.shape[:2], 4)
    changes = np.sum(quaternions[:-1] * quaternions[1:], axis=-1) < 0.
    quaternions[1:] *= np.cumprod(np.where(changes, -1., 1.), axis=0)[..., None]
    np.savez(work / 'retarget.npz', names=np.asarray(rig.names), parents=rig.parents,
             rest_positions=rest_positions @ basis, rest_rotations=basis @ rest_rotations @ basis,
             locations=local_p.astype(np.float32), quaternions=quaternions[..., [3, 0, 1, 2]].astype(np.float32),
             fps=float(fps), skin_name='' if skin is None else f'SKM_{skin.stem.title()}_Simple')
    if skin is not None:
        shutil.copyfile(skin, work / 'skin.npz')
    build(work, 'ue5_fbx_scene.py', emit, 'save_fbx')
    if not Path(path).is_file():
        raise RuntimeError('Blender did not write the UE5 Mannequin FBX.')


def export_retargeted(path, source_names, rest_positions, rest_rotations, positions, rotations,
                      fps, target, *, in_place, emit, source_tip_local, transform=None, skin=None):
    if transform is not None:
        transform = validate_transform(transform)
        positions = positions @ transform[:3, :3].T + transform[:3, 3]
        rotations = transform[:3, :3] @ rotations
    emit('progress', stage='save_fbx', current=0, total=0, message='Retargeting motion to the UE5 mannequin')
    local_p, local_r = retarget(source_names, rest_positions, rest_rotations, positions, rotations,
                               target, in_place=in_place, source_tip_local=source_tip_local)
    write(path, target, local_p, local_r, fps, emit, skin=skin)
