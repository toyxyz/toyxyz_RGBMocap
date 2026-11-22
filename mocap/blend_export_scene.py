"""Build the portable Blender scene inside the managed Blender runtime."""
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

WORLD = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
CV = Matrix.Diagonal((1, -1, -1, 1))


def fcurves(action):
    """F-curves of an action: Blender 5 keeps them in slotted channel bags."""
    if hasattr(action, 'fcurves'):
        return list(action.fcurves)
    return [curve for layer in action.layers for strip in layer.strips for bag in strip.channelbags for curve in bag.fcurves]


def progress(current, total, message):
    print('@@BLEND@@' + json.dumps(dict(current=current, total=total, message=message)), flush=True)


def import_rig(folder, scene, take, combined, fps_base):
    before = set(scene.objects)
    bpy.ops.import_scene.fbx(filepath=str(folder / take['fbx']), anim_offset=0,
                             automatic_bone_orientation=False, use_prepost_rot=True, ignore_leaf_bones=True)
    rigs = [obj for obj in scene.objects if obj not in before and obj.type == 'ARMATURE']
    if len(rigs) != 1:
        raise ValueError('Expected one exported source armature per take')
    rig = rigs[0]
    rig.show_in_front = True
    rig.data.display_type = 'OCTAHEDRAL'
    if not combined:
        return rig

    first, end = take['first'], take['first'] + take['frames'] + int(take['prepend_tpose'])
    action = rig.animation_data.action
    for curve in fcurves(action):
        for key in curve.keyframe_points:
            key.co.x = first + key.co.x / fps_base
            key.handle_left.x = first + key.handle_left.x / fps_base
            key.handle_right.x = first + key.handle_right.x / fps_base
        curve.update()
    rig['RGBMoCap person'] = take['name']
    rig['RGBMoCap first frame'] = first
    rig['RGBMoCap last frame'] = end - 1
    for frame, hidden in ((-1, True), (first, False), (end, True)):
        for prop in ('hide_viewport', 'hide_render'):
            setattr(rig, prop, hidden)
            rig.keyframe_insert(data_path=prop, frame=frame)
    for curve in fcurves(action):
        if curve.data_path in ('hide_viewport', 'hide_render'):
            for key in curve.keyframe_points:
                key.interpolation = 'CONSTANT'
    return rig


def add_camera(folder, scene, take, name, combined):
    camera = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, camera)
    scene.collection.objects.link(obj)
    start = take['first'] + int(take['prepend_tpose'])
    frames = take['frames']
    obj.matrix_world = WORLD @ Matrix(take['world_to_camera']).inverted() @ CV
    if take['camera_track']:
        import numpy as np

        track = np.load(folder / take['camera_track'], allow_pickle=False, mmap_mode='r')
        try:
            if track.shape != (frames, 4, 4):
                raise ValueError('Invalid camera track length')
            obj.rotation_mode = 'QUATERNION'
            previous = None
            for frame, matrix in enumerate(track):
                if frame % 100 == 0:
                    progress(frame, frames, 'Baking camera animation')
                location, rotation, _ = (WORLD @ Matrix(matrix.tolist()).inverted() @ CV).decompose()
                if previous is not None and rotation.dot(previous) < 0:
                    rotation.negate()
                obj.location = location
                obj.rotation_quaternion = rotation
                previous = rotation.copy()
                for prop in ('location', 'rotation_quaternion'):
                    obj.keyframe_insert(data_path=prop, frame=frame + start)
                    if frame == 0 and take['prepend_tpose']:
                        obj.keyframe_insert(data_path=prop, frame=take['first'])
        finally:
            track._mmap.close()
        for curve in fcurves(obj.animation_data.action):
            for key in curve.keyframe_points:
                key.interpolation = 'LINEAR'

    k, width, height = take['K'], take['width'], take['height']
    ratio = k[0][0] / k[1][1]
    camera.type = 'PERSP'
    camera.sensor_fit = 'HORIZONTAL'
    camera.sensor_width = 36
    camera.lens = k[0][0] * 36 / width
    camera.shift_x = (width / 2 - k[0][2]) / width
    camera.shift_y = (k[1][2] - height / 2) * ratio / width
    camera.clip_start = .01
    camera.clip_end = 10000
    camera.show_background_images = True
    background = camera.background_images.new()
    background.source = 'IMAGE'
    path = take['video']
    image = bpy.data.images.load(str(folder / path), check_existing=True)
    image.filepath = '//' + path
    background.image = image
    background.image_user.frame_start = start
    background.image_user.frame_duration = frames
    background.image_user.use_auto_refresh = True
    background.image_user.use_cyclic = False
    background.alpha = 1
    background.display_depth = 'BACK'
    background.frame_method = 'FIT'
    if combined:
        for frame, visible in ((-1, False), (start, True), (start + frames, False)):
            camera.show_background_images = visible
            camera.keyframe_insert(data_path='show_background_images', frame=frame)
        for key in fcurves(camera.animation_data.action)[0].keyframe_points:
            key.interpolation = 'CONSTANT'
    return obj


def setup_cameras(folder, scene, takes, combined):
    reference = takes[0]
    k = reference['K']
    ratio = k[0][0] / k[1][1]
    scene.render.resolution_x = reference['width']
    scene.render.resolution_y = reference['height']
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = max(1., 1. / ratio)
    scene.render.pixel_aspect_y = max(1., ratio)
    if not combined:
        scene.camera = add_camera(folder, scene, reference, 'Source video camera', False)
        return

    cameras = {}
    for take in takes:
        first = take['first'] + int(take['prepend_tpose'])
        end = first + take['frames']
        kal = tuple(value for row in take['K'] for value in row)
        key = (take['video'], first, end, kal, take['camera_key'])
        if key not in cameras:
            name = f'Source video {len(cameras) + 1:02d}'
            cameras[key] = first, end, add_camera(folder, scene, take, name, True)
    intervals = list(cameras.values())
    current = None
    for frame in sorted({value for first, end, _ in intervals for value in (first, end)}):
        if current is not None and current[0] <= frame < current[1]:
            continue
        candidate = next((entry for entry in intervals if entry[0] <= frame < entry[1]), None)
        if candidate is not None:
            marker = scene.timeline_markers.new(candidate[2].name, frame=frame)
            marker.camera = candidate[2]
        current = candidate
    scene.camera = intervals[0][2]


def main(folder):
    job = json.loads((folder / 'blend-job.json').read_text(encoding='utf-8'))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    combined = 'takes' in job
    takes = job.get('takes', [job])
    fps = float(job['fps'])
    fps_base = round(fps) / fps
    rigs = []
    for index, take in enumerate(takes, 1):
        progress(index, len(takes), 'Importing source animation')
        rigs.append(import_rig(folder, scene, take, combined, fps_base))

    start = int(takes[0]['prepend_tpose'])
    scene.frame_start = 0
    scene.frame_end = job['frame_end'] if combined else takes[0]['frames'] - 1 + start
    scene.render.fps = round(fps)
    scene.render.fps_base = fps_base
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1
    setup_cameras(folder, scene, takes, combined)
    scene['RGBMoCap video start'] = start
    scene['RGBMoCap camera'] = takes[0].get('camera_mode', 'Stored reference camera')
    scene['RGBMoCap video'] = 'Keep the video files next to this Blender project'
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.region_3d.view_perspective = 'CAMERA'
                space.overlay.show_floor = False
                space.overlay.show_axis_x = False
                space.overlay.show_axis_y = False
    scene.frame_set(start)
    for item in bpy.context.selected_objects:
        item.select_set(False)
    rigs[0].select_set(True)
    bpy.context.view_layer.objects.active = rigs[0]
    progress(0, 0, 'Saving portable Blender project')
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / job['target']), check_existing=False)


main(Path(sys.argv[sys.argv.index('--') + 1]))
