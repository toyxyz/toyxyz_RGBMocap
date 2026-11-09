"""Export the selected SOMA Native77 motion without loading inference models."""

from pathlib import Path
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from .blend_support import Inputs
from .export_options import centered_transform
from .gemx_correct import KIND, selected
from .gemx_result import ASSET, FPS, JOINT_COUNT, POSE, RECEIPT, load_rig, manifest
from .motion_tools import selected_export
from .source_bvh import continuous_angles, write_channels


def write(path, row, source, destinations, fmt, progress):
    project = Path(row['project']).resolve()
    source = source.resolve()
    kind = row['kind']
    if kind not in ('initial', KIND):
        raise ValueError('Select the original or corrected SOMA motion.')
    if row.get('scene') is not None or row.get('symmetric_rest') or row.get('bvh_profile', 'generic') != 'generic':
        raise ValueError('SOMA export requires the native rest pose and Generic / Blender profile.')

    inputs = Inputs()
    for name in ('project.json', RECEIPT, ASSET):
        inputs.track(project / name)
    inputs.track(source)
    if selected_export(project, kind) != source:
        raise ValueError('The selected SOMA result changed. Export again.')
    folder = project
    if kind != 'initial':
        correction = selected(project)
        if correction is None:
            raise ValueError('Apply Correction before exporting the corrected SOMA motion.')
        folder = correction[0]
    pose_path = inputs.track(folder / POSE)
    if destinations.intersection(inputs.files):
        raise ValueError('An export destination overlaps a committed source file.')
    rig = load_rig(project)
    names = rig['names'].tolist()
    if any(not name.isascii() or any(c.isspace() or c in '{}:\0\1' for c in name) for name in names):
        raise ValueError('Invalid SOMA joint name.')
    parents = rig['parents']
    rest = rig['rest'][:, :3, 3].astype(float)
    basis = rig['rest'][:, :3, :3].astype(float)
    pose = np.load(pose_path, mmap_mode='r', allow_pickle=False)
    try:
        if pose.shape != (manifest(project)['frames'], JOINT_COUNT, 4, 4) or pose.dtype != np.float32:
            raise ValueError('Invalid SOMA animation dimensions.')
        frames = pose.astype(float)
    finally:
        pose._mmap.close()
    if not np.isfinite(frames).all():
        raise ValueError('Non-finite SOMA animation.')
    rotations = frames[:, :, :3, :3].copy()
    for axes in (basis, rotations):
        if (not np.allclose(axes.swapaxes(-1, -2) @ axes, np.eye(3), rtol=0, atol=1e-4)
                or not np.allclose(np.linalg.det(axes), 1., rtol=0, atol=1e-4)):
            raise ValueError('Invalid SOMA rotation basis.')
    # Express every world rotation relative to the corresponding native rest axes.
    frames[:, :, :3, :3] = rotations @ basis.transpose(0, 2, 1)
    transform = np.eye(4)
    if row.get('place_at_origin'):
        transform = centered_transform(transform, frames[0, 0, :3, 3], {}, project)
    prepend = row.get('prepend_tpose', False)
    if fmt == 'fbx':
        from .bone_overlay import reference_frame
        from .fbx_writer import write as write_fbx

        # FBX bone tails follow local +Y. Use segment directions for display
        # and apply the same basis in the bind pose and every animation frame.
        directions = rig['ends'].astype(float)
        primary = {'Hips': 'Spine1', 'Head': 'HeadEnd'}
        indices = {name: index for index, name in enumerate(names)}
        for joint, name in enumerate(names):
            children = np.flatnonzero(parents == joint)
            if len(children):
                child = indices.get(primary.get(name))
                tail = rest[child] if child in children else rest[children].mean(axis=0)
                directions[joint] = tail - rest[joint]
        directions[np.linalg.norm(directions, axis=1) < .001] = (0., .02, 0.)
        display = np.stack([reference_frame(direction) for direction in directions])
        tips = np.einsum('nji,nj->ni', display, directions)
        write_fbx(path, names, parents, rest, display, display, tips, frames, FPS,
                  transform=transform, progress=progress, rig_name=row.get('rig_name', 'SourceAnimation'),
                  prepend_tpose=prepend, expected_bones=JOINT_COUNT)
    else:
        parent = parents[1:]
        positions = frames[:, :, :3, 3]
        world = frames[:, :, :3, :3]
        offsets = positions[:, 1:] - positions[:, parent]
        fixed = np.einsum('tnji,tnj->tni', world[:, parent], offsets)
        if np.max(abs(fixed - (rest[1:] - rest[parent]))) > 2e-5:
            raise ValueError('Non-rigid bone offset in committed SOMA animation.')
        local = world.copy()
        local[:, 1:] = world[:, parent].swapaxes(-1, -2) @ world[:, 1:]
        roots = positions[:, 0] + transform[:3, 3]
        if prepend:
            roots = np.concatenate((roots[:1], roots))
            local = np.concatenate((np.broadcast_to(np.eye(3), (1, JOINT_COUNT, 3, 3)), local))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Gimbal lock detected')
            angles = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_euler('ZXY').reshape(len(roots), JOINT_COUNT, 3)
        for joint in range(JOINT_COUNT):
            angles[:, joint] = continuous_angles(angles[:, joint])
            progress(joint + 1, JOINT_COUNT)
        write_channels(path, names, parents, rest, rig['ends'], roots, np.degrees(angles), ['ZXY'] * JOINT_COUNT, FPS,
                       exclusive=True)

    def verify():
        if selected_export(project, kind) != source:
            raise ValueError('The selected SOMA result changed during export. Retry.')
        inputs.verify()

    return verify
