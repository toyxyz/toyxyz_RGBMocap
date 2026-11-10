"""Blender background entry point for an armature-only FBX export."""

import json
from pathlib import Path
import sys

import bpy

# Blender starts with its own Python paths. This entry point only imports the
# pinned FBX sampler already used by character exports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mocap.retarget.selective_fbx import body_only_sampler


def progress(current, message):
    print('@@BLEND@@' + json.dumps(dict(current=current, total=3, message=message)), flush=True)


def export(work):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    progress(0, 'Loading skeleton for FBX compatibility')
    bpy.ops.import_scene.fbx(
        filepath=str(work / 'source.fbx'), anim_offset=0,
        automatic_bone_orientation=False, use_prepost_rot=True, ignore_leaf_bones=True,
    )
    scene = bpy.context.scene
    armatures = [obj for obj in scene.objects if obj.type == 'ARMATURE']
    if len(armatures) != 1:
        raise ValueError('Expected one source armature for FBX export.')
    arm = armatures[0]
    if arm.animation_data is None or arm.animation_data.action is None:
        raise ValueError('The source armature has no animation.')

    action = arm.animation_data.action
    fps_base = scene.render.fps_base
    if fps_base != 1.0:
        # Blender 4.5 imports key times using render.fps alone. Bake at the
        # original frame rate by accounting for fps_base before sampling.
        for curve in action.fcurves:
            for key in curve.keyframe_points:
                key.co.x /= fps_base
                key.handle_left.x /= fps_base
                key.handle_right.x /= fps_base
            curve.update()

    scene.unit_settings.system = 'METRIC'
    scene.frame_start, scene.frame_end = map(round, action.frame_range)
    scene.frame_set(scene.frame_start)
    bpy.ops.object.select_all(action='DESELECT')
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm

    progress(1, 'Writing compatible FBX in Blender')
    with body_only_sampler(arm, set(arm.data.bones.keys())):
        result = bpy.ops.export_scene.fbx(
            filepath=str(work / 'compatible.fbx'), use_selection=True, object_types={'ARMATURE'},
            axis_forward='-Z', axis_up='Y', add_leaf_bones=True,
            bake_anim=True, bake_anim_use_all_bones=False, bake_anim_use_all_actions=False,
            bake_anim_use_nla_strips=False, bake_anim_step=1.0, bake_anim_simplify_factor=0.0,
        )
    if result != {'FINISHED'}:
        raise RuntimeError('Blender did not finish writing the FBX.')
    progress(3, 'Compatible FBX written')


if __name__ == '__main__':
    export(Path(sys.argv[sys.argv.index('--') + 1]))
