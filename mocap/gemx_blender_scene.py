"""Blender-only builder for the native rig, exact PC2 surface and reference video."""

import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mocap.retarget.blender_bake import key_values
from mocap.retarget.file_paths import io_path, save_blend


def main(folder):
    job = json.loads(Path(io_path(folder / 'blend-job.json')).read_text(encoding='utf-8'))
    with np.load(io_path(folder / 'rig.npz'), allow_pickle=False) as data:
        names, parents, rest, poses, first, faces = (data[key] for key in
            ('names', 'parents', 'rest', 'pose', 'first', 'faces'))
    n, fps = job['frames'], job['fps']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = 'SOMA Motion'
    scene.unit_settings.system = 'METRIC'
    scene.frame_start, scene.frame_end, scene.render.fps = 1, n, fps
    convert = np.array(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    rest_world = convert @ rest
    arm = bpy.data.armatures.new('SOMA Native77')
    rig = bpy.data.objects.new('SOMA Native77 · Editable Reference', arm)
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    rig.show_in_front = True
    bpy.ops.object.mode_set(mode='EDIT')
    for index, name in enumerate(names):
        bone = arm.edit_bones.new(str(name))
        bone.head = rest_world[index, :3, 3]
        children = np.flatnonzero(parents == index)
        if len(children) and np.linalg.norm(rest_world[children[0], :3, 3] - bone.head) > .001:
            bone.tail = rest_world[children[0], :3, 3]
        else:
            bone.tail = bone.head + Vector(rest_world[index, :3, 1]) * .02
        if parents[index] >= 0:
            bone.parent = arm.edit_bones[str(names[parents[index]])]
    bpy.ops.object.mode_set(mode='POSE')
    bind = np.array([np.array(arm.bones[str(name)].matrix_local) for name in names])
    posed = convert @ poses @ np.linalg.inv(rest_world) @ bind
    for index, name in enumerate(names):
        parent = parents[index]
        relative = np.linalg.inv(posed[:, parent]) @ posed[:, index] if parent >= 0 else posed[:, index]
        bind_relative = np.linalg.inv(bind[parent]) @ bind[index] if parent >= 0 else bind[index]
        local = np.linalg.inv(bind_relative) @ relative
        quaternions = np.array([tuple(Matrix(frame[:3, :3].tolist()).to_quaternion().normalized()) for frame in local])
        for i in range(1, n):
            if quaternions[i] @ quaternions[i - 1] < 0:
                quaternions[i] *= -1
        bone = rig.pose.bones[str(name)]
        bone.rotation_mode = 'QUATERNION'
        bone.rotation_quaternion = quaternions[0]
        bone.location = local[0, :3, 3]
        for path, values in (('rotation_quaternion', quaternions), ('location', local[:, :3, 3])):
            bone.keyframe_insert(path, frame=1)
            key_values(rig, bone.path_from_id(path), values)
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh = bpy.data.meshes.new('SOMA MID')
    mesh.from_pydata(first.tolist(), [], faces.tolist())
    mesh.update()
    surface = bpy.data.objects.new('SOMA · Baked Surface', mesh)
    scene.collection.objects.link(surface)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    cache = surface.modifiers.new('Exact SOMA animation', 'MESH_CACHE')
    cache.cache_format = 'PC2'
    cache.filepath = '//surface.pc2'
    cache.play_mode, cache.time_mode, cache.frame_start = 'SCENE', 'FRAME', 1
    cache.forward_axis, cache.up_axis, cache.deform_mode = 'POS_Y', 'POS_Z', 'OVERWRITE'
    points = posed[:, :, :3, 3]
    center = Vector((points.min(axis=(0, 1)) + points.max(axis=(0, 1))) / 2)
    distance = max(2., float(np.ptp(points, axis=(0, 1)).max()) * 2.)
    view = Vector((3., -5., 2.)).to_track_quat('Z', 'Y')
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                region = area.spaces.active.region_3d
                region.view_distance, region.view_location, region.view_rotation = distance, center, view

    reference = bpy.data.scenes.new('Input Video')
    reference.frame_start, reference.frame_end, reference.render.fps = 1, n, fps
    reference.render.resolution_x, reference.render.resolution_y = job['width'], job['height']
    reference.render.resolution_percentage = 100
    strips = reference.sequence_editor_create().strips
    video = strips.new_movie('Synchronized input', io_path(folder / 'video.mp4'), channel=1, frame_start=1)
    if video.frame_duration != n:
        raise ValueError('Blender decoded a different number of video frames.')
    video.filepath = '//video.mp4'
    scene.frame_set(1)
    save_blend(folder / job['target'])


main(Path(sys.argv[sys.argv.index('--') + 1]))
