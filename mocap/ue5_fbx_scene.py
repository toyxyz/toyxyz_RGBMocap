"""Blender background entry point for the UE5 mannequin skeleton and optional skinned mesh."""
from contextlib import contextmanager, nullcontext
from fractions import Fraction
import json
from pathlib import Path
import struct
import sys
import zlib

import bpy
from mathutils import Matrix
import numpy as np


def progress(current, message):
    print('@@BLEND@@' + json.dumps(dict(current=current, total=3, message=message)), flush=True)


def build_armature(names, parents, rest_positions, rest_rotations):
    data = bpy.data.armatures.new('Mannequin')
    arm = bpy.data.objects.new(names[0], data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    # Scene values are metres. Keep the edit bones in centimetres so the
    # exported FBX has a unit-scale root and centimetre child offsets.
    arm.scale = (.01, .01, .01)
    bpy.ops.object.mode_set(mode='EDIT')
    bones = [None]
    for joint, name in enumerate(names[1:], start=1):
        bone = data.edit_bones.new(name)
        bone.parent = bones[parents[joint]]
        bone.head = rest_positions[joint]
        bone.tail = rest_positions[joint] + rest_rotations[joint, :, 1] * 3.
        matrix = Matrix(rest_rotations[joint]).to_4x4()
        matrix.translation = bone.head
        bone.matrix = matrix
        bones.append(bone)
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm


def add_skin(path, arm, names, label):
    mesh = bpy.data.meshes.new(label)
    with np.load(path, allow_pickle=False) as data:
        mesh.from_pydata(data['vertices'].tolist(), [], data['triangles'].tolist())
        mesh.update()
        mesh.polygons.foreach_set('use_smooth', np.ones(len(mesh.polygons), dtype=bool))
        mesh.normals_split_custom_set(data['normals'])
        mesh.uv_layers.new(name='UVMap').uv.foreach_set('vector', data['uv'].ravel())
        indices = data['indices']
        weights = data['weights']

    skin = bpy.data.objects.new(label, mesh)
    bpy.context.collection.objects.link(skin)
    skin.parent = arm
    skin.select_set(True)
    groups = {int(joint): skin.vertex_groups.new(name=names[joint])
              for joint in np.unique(indices[:, 1])}
    for (vertex, joint), weight in zip(indices, weights):
        groups[joint].add([int(vertex)], float(weight), 'REPLACE')
    modifier = skin.modifiers.new('Mannequin Skin', 'ARMATURE')
    modifier.object = arm
    return skin


def animate(arm, names, locations, rotations):
    action = bpy.data.actions.new('Motion')
    animation = arm.animation_data_create()
    animation.action = action
    animation.action_slot = action.slots.new('OBJECT', arm.name)
    strip = action.layers.new('Motion').strips.new(type='KEYFRAME')
    channels = strip.channelbags.new(animation.action_slot)
    keys = np.zeros((len(locations), 2), dtype=np.float32)
    keys[:, 0] = np.arange(len(keys), dtype=np.float32)

    for joint, name in enumerate(names):
        owner = arm if joint == 0 else arm.pose.bones[name]
        owner.rotation_mode = 'QUATERNION'
        prefix = '' if joint == 0 else owner.path_from_id() + '.'
        for prop, values in (('location', locations[:, joint]), ('rotation_quaternion', rotations[:, joint])):
            if prop == 'location' and joint != 0 and not np.any(abs(values) > 1e-7):
                continue
            for axis in range(values.shape[1]):
                curve = channels.fcurves.new(prefix + prop, index=axis)
                keys[:, 1] = values[:, axis]
                curve.keyframe_points.add(len(keys))
                curve.keyframe_points.foreach_set('co', keys.ravel())
                for point in curve.keyframe_points:
                    point.interpolation = 'LINEAR'
                curve.update()


def element_child(element, name):
    return next(child for child in element.elems if child.id == name)


def rigid_matrix(element):
    payload, = element.props
    count, encoding, size = struct.unpack_from('<III', payload)
    if count != 16 or encoding not in (0, 1):
        raise ValueError('Invalid mannequin FBX bind matrix.')
    raw = payload[12:12+size]
    values = zlib.decompress(raw) if encoding else raw
    matrix = np.frombuffer(values, dtype='<f8').reshape(4, 4).T.copy()
    left, _, right = np.linalg.svd(matrix[:3, :3])
    matrix[:3, :3] = left @ right
    return matrix


def set_matrix(element, matrix):
    element.props.clear()
    element.props_type.clear()
    element.add_float64_array(matrix.T.ravel())


@contextmanager
def skeleton_bind_pose():
    # Blender emits BindPose records for skinned meshes. Animation-only FBX
    # needs the same record, otherwise Blender can use its animated first pose
    # as the rest pose. Reuse the exporter's world-space bone transforms and IDs.
    from io_scene_fbx import export_fbx_bin

    scene_data = export_fbx_bin.fbx_data_from_scene
    armature_elements = export_fbx_bin.fbx_data_armature_elements

    def scene_with_bind(scene, depsgraph, settings):
        data = scene_data(scene, depsgraph, settings)
        if b'BindPose' in data.templates:
            return data
        data.templates[b'BindPose'] = export_fbx_bin.fbx_template_def_pose(scene, settings, nbr_users=1)
        return data._replace(templates_users=data.templates_users + 1)

    def write_armature(root, armature, data):
        armature_elements(root, armature, data)
        if not any(element.id == b'Pose' and element.props[2] == b'BindPose' for element in root.elems):
            bones = tuple(bone for bone in armature.bones if bone in data.objects)
            export_fbx_bin.fbx_data_bindpose_element(
                root, armature, armature.bdata.data, data, bones=bones)

    export_fbx_bin.fbx_data_from_scene = scene_with_bind
    export_fbx_bin.fbx_data_armature_elements = write_armature
    try:
        yield
    finally:
        export_fbx_bin.fbx_data_from_scene = scene_data
        export_fbx_bin.fbx_data_armature_elements = armature_elements


@contextmanager
def rigid_bind_pose(with_skin):
    # Blender builds the bind from float32 matrices. FBX SDK decomposes them
    # on load, and small scale errors near right angles can invalidate the
    # entire bind. Keep all bind/cluster matrices consistently rigid.
    from io_scene_fbx import encode_bin
    original_write = encode_bin.write

    def write_with_rigid_bind(filename, root, version):
        objects = element_child(root, b'Objects')
        pose = element_child(objects, b'Pose')
        mesh = None
        if with_skin:
            mesh, = (node for node in objects.elems if node.id == b'Model' and node.props[2][4:] == b'Mesh')
        mesh_bind = None
        for node in pose.elems:
            if node.id != b'PoseNode':
                continue
            entry = element_child(node, b'Matrix')
            matrix = rigid_matrix(entry)
            set_matrix(entry, matrix)
            if mesh is not None and element_child(node, b'Node').props == mesh.props[:1]:
                mesh_bind = matrix
        if mesh is None:
            return original_write(filename, root, version)
        if mesh_bind is None:
            raise ValueError('Missing mannequin FBX mesh bind matrix.')
        for node in objects.elems:
            if node.id != b'Deformer' or node.props[2][4:] != b'Cluster':
                continue
            link = element_child(node, b'TransformLink')
            associate = element_child(node, b'TransformAssociateModel')
            bone_bind = rigid_matrix(link)
            set_matrix(link, bone_bind)
            set_matrix(associate, rigid_matrix(associate))
            set_matrix(element_child(node, b'Transform'), np.linalg.solve(bone_bind, mesh_bind))
        return original_write(filename, root, version)

    encode_bin.write = write_with_rigid_bind
    try:
        yield
    finally:
        encode_bin.write = original_write


def export(work):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.
    progress(0, 'Preparing the stock mannequin armature')
    with np.load(work / 'retarget.npz', allow_pickle=False) as data:
        names = data['names'].tolist()
        arm = build_armature(names, data['parents'], data['rest_positions'], data['rest_rotations'])
        locations, rotations = data['locations'], data['quaternions']
        fps = float(data['fps'])
        skin_label = str(data['skin_name'])
    arm['EpicAssetNotice'] = 'Reference rig from Epic Games Unreal Engine 5.7.4 Third Person template'
    arm['EpicAssetTerms'] = 'https://www.unrealengine.com/eula/unreal'
    if skin_label:
        skin = add_skin(work / 'skin.npz', arm, names, skin_label)
        skin['EpicAssetNotice'] = 'Epic Games Unreal Engine 5.7.4 Third Person template: ' + skin_label
        skin['EpicAssetTerms'] = arm['EpicAssetTerms']
    rate = Fraction(str(fps)).limit_denominator(1001)
    scene.render.fps = max(1, min(32767, round(fps)))
    scene.render.fps_base = float(scene.render.fps / rate)
    scene.frame_start, scene.frame_end = 0, len(locations)-1
    animate(arm, names, locations, rotations)
    scene.frame_set(scene.frame_start)
    progress(1, 'Writing UE5 Mannequin FBX')
    bind_context = nullcontext() if skin_label else skeleton_bind_pose()
    with bind_context, rigid_bind_pose(with_skin=bool(skin_label)):
        result = bpy.ops.export_scene.fbx(
            filepath=str(work / 'mannequin.fbx'), use_selection=True, object_types={'ARMATURE', 'MESH'} if skin_label else {'ARMATURE'},
            axis_forward='-Y', axis_up='Z', apply_unit_scale=True, apply_scale_options='FBX_SCALE_NONE',
            mesh_smooth_type='FACE' if skin_label else 'OFF',
            armature_nodetype='NULL', use_custom_props=True, add_leaf_bones=False, use_armature_deform_only=False,
            bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_all_actions=False,
            bake_anim_use_nla_strips=False, bake_anim_step=1., bake_anim_simplify_factor=0.)
    if result != {'FINISHED'}:
        raise RuntimeError('UE5 Mannequin FBX export did not finish.')
    progress(3, 'UE5 Mannequin FBX written')


if __name__ == '__main__':
    export(Path(sys.argv[sys.argv.index('--') + 1]))
