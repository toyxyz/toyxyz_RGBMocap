"""Bake Native77 rotations directly onto the existing character rest skeleton."""

import json
from pathlib import Path

import bpy
from mathutils import Matrix
import numpy as np

from mocap.gemx_result import FPS, MESH, POSE, load_rig
from mocap.gemx_retarget import BODY, finger_mapping
from .blender_bake import A, curves, key_values, rotation
from .file_paths import io_path, save_blend
from .mapping import body_trajectory_root


def bake(out, kind, project, motion=None):
    motion = project if motion is None else motion
    bpy.ops.wm.open_mainfile(filepath=io_path(out / f'{kind}-rest.blend'))
    rig = json.loads(Path(io_path(out / f'{kind}-rig.json')).read_text(encoding='utf-8'))
    arm = bpy.data.objects[rig['armature']]
    arm.animation_data_clear()
    with np.load(io_path(out / f'{kind}-canonical.npz'), allow_pickle=False) as data:
        targets = data['names'].tolist()
    source_rig = load_rig(Path(io_path(project)))
    source_indices = {name: i for i, name in enumerate(source_rig['names'])}
    source_rest = source_rig['rest']
    native = np.load(io_path(motion / POSE), mmap_mode='r', allow_pickle=False)
    surface = np.load(io_path(motion / MESH), mmap_mode='r', allow_pickle=False)
    mapping = dict(zip(targets, BODY))
    mapping.update(finger_mapping(kind, arm.data.bones.keys(), rig['semantic']))
    if kind == 'pmx' and '腰' in arm.data.bones:
        mapping['腰'] = 'Hips'
    root_name = body_trajectory_root(
        {b.name: b.parent.name if b.parent else None for b in arm.data.bones}, targets)
    if rig.get('trajectory_root', root_name) != root_name:
        raise ValueError('Character hierarchy changed after import. Import the character again.')
    needed = set(mapping)
    for name in mapping:
        ancestor = arm.data.bones[name].parent
        while ancestor is not None:
            needed.add(ancestor.name)
            ancestor = ancestor.parent
    bones = [bone for bone in arm.data.bones if bone.name in needed]
    rest = {bone.name: np.array(arm.matrix_world @ bone.matrix_local) for bone in bones}
    axes = {name: rotation(matrix[:3, :3]) for name, matrix in rest.items()}
    heads = {name: matrix[:3, 3] for name, matrix in rest.items()}
    left, right = (source_rest[source_indices[side + 'Leg'], :3, 3] @ A.T for side in ('Left', 'Right'))
    source_lateral = left - right
    target_lateral = heads[targets[1]] - heads[targets[2]]
    yaw = np.arctan2(target_lateral[1], target_lateral[0]) - np.arctan2(source_lateral[1], source_lateral[0])
    axes_map = np.array(Matrix.Rotation(float(yaw), 3, 'Z')) @ A
    target_length = sum(np.linalg.norm(heads[targets[a]] - heads[targets[b]]) for a, b in ((1, 4), (4, 7), (2, 5), (5, 8)))
    source_length = sum(np.linalg.norm(source_rest[source_indices[BODY[a]], :3, 3] -
                                       source_rest[source_indices[BODY[b]], :3, 3]) for a, b in ((1, 4), (4, 7), (2, 5), (5, 8)))
    if min(source_length, target_length) <= 1e-6:
        raise ValueError('Character or SOMA leg length is invalid.')
    scale = target_length / source_length

    for bone in arm.pose.bones:
        bone.matrix_basis.identity()
        if bone.name in mapping or (kind == 'pmx' and bone.name.startswith('腰キャンセル')):
            for constraint in bone.constraints:
                constraint.mute = True
    world, positions = {}, {}
    n = len(native)
    for bone in bones:
        name = bone.name
        parent = bone.parent.name if bone.parent else None
        if parent:
            pre = world[parent] @ axes[parent].T @ axes[name]
            positions[name] = positions[parent] + (world[parent] @ axes[parent].T @ (heads[name] - heads[parent]))
        else:
            pre = np.broadcast_to(axes[name], (n, 3, 3))
            positions[name] = heads[name]
        world[name] = pre
        if name not in mapping:
            continue
        j = source_indices[mapping[name]]
        delta = native[:, j, :3, :3] @ source_rest[j, :3, :3].T
        world[name] = axes_map @ delta @ axes_map.T @ axes[name]
        local = pre.swapaxes(-1, -2) @ world[name]
        q = np.array([tuple(Matrix(m.tolist()).to_quaternion().normalized()) for m in local])
        for frame in range(1, n):
            if q[frame] @ q[frame - 1] < 0:
                q[frame] *= -1
        pb = arm.pose.bones[name]
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = q[0]
        pb.keyframe_insert('rotation_quaternion', frame=1)
        key_values(arm, pb.path_from_id('rotation_quaternion'), q)

    native_root = native[:, 0, :3, 3] @ A.T
    start = native_root[0].copy()
    # Preserve the source starting X/Z in app space and its height above the
    # preview floor. Scale travel by the character's original leg proportions.
    start[2] = float(surface[0, :, 1].min()) * scale
    desired_hips = heads[targets[0]] + start + (native_root - native_root[0]) * scale
    shift = desired_hips - positions[targets[0]]
    pb = arm.pose.bones[root_name]
    if any(not c.mute and c.influence > 0 for c in pb.constraints):
        raise ValueError('A constrained character trajectory root cannot be baked safely.')
    locations = shift @ np.linalg.inv(rest[root_name][:3, :3]).T
    pb.location = locations[0]
    pb.keyframe_insert('location', frame=1)
    key_values(arm, pb.path_from_id('location'), locations)
    if any(not curve.data_path.startswith('pose.bones[') for curve in curves(arm.animation_data.action)):
        raise ValueError('Unexpected armature object animation.')
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, n
    scene.render.fps, scene.render.fps_base = FPS, 1.
    scene.frame_set(1)
    save_blend(out / 'animation.blend')
    return arm, sorted(set(mapping) | {root_name})
