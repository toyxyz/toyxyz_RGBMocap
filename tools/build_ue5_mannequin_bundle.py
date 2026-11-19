"""Build portable skin data from verified stock UE5 Third Person FBX exports.

Run in Blender 4.5 with -- <profiles.json> <Manny FBX> <Quinn FBX> <output directory>.
Profiles must come from read_ue5_mannequin.py for the same engine build.
"""
import json
from pathlib import Path
import sys

import bpy
from mathutils import Quaternion
import numpy as np


def extract(filename, bones, output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(filename), use_anim=False, use_image_search=False)
    arm, = (obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE')
    obj, = (obj for obj in bpy.context.scene.objects if obj.type == 'MESH')
    bind = list(arm.data.bones)
    names = [row['name'] for row in bones]
    if arm.name != names[0] or {bone.name for bone in bind} != set(names[1:]):
        raise ValueError('Stock FBX and reference have different bone names.')

    # Validate the stock bind against Unreal and place the skin in the same
    # right-handed centimetre armature space used by the animation exporter.
    mirror = np.diag([1., -1., 1.])
    positions, rotations = [], []
    for row in bones:
        x, y, z, w = row['rotation']
        rotation = np.asarray(Quaternion((w, x, y, z)).to_matrix())
        point = np.asarray(row['position'])
        parent = row['parent']
        if parent >= 0:
            point = positions[parent] + rotations[parent] @ point
            rotation = rotations[parent] @ rotation
        positions.append(point)
        rotations.append(rotation)
    for bone in bind:
        joint = names.index(bone.name)
        parent = bone.parent.name if bone.parent else arm.name
        if parent != names[bones[joint]['parent']]:
            raise ValueError('Stock FBX and reference have different parents.')
        np.testing.assert_allclose(bone.head_local, mirror @ positions[joint], atol=.002, rtol=0)

    mesh = obj.data
    mesh.calc_loop_triangles()
    to_arm = arm.matrix_world.inverted() @ obj.matrix_world
    vertices = np.asarray([to_arm @ vertex.co for vertex in mesh.vertices], dtype=np.float32)
    triangles = np.asarray([tri.vertices[:] for tri in mesh.loop_triangles], dtype=np.int32)
    loops = np.asarray([tri.loops[:] for tri in mesh.loop_triangles], dtype=np.int32).ravel()
    normals = np.asarray([normal.vector[:] for normal in mesh.corner_normals])[loops]
    normals = normals @ np.linalg.inv(np.asarray(to_arm.to_3x3()))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    uv_layer = mesh.uv_layers.active.uv if mesh.uv_layers.active else ()
    uv = np.asarray([entry.vector[:] for entry in uv_layer], dtype=np.float32)
    uv = uv[loops] if len(uv) else np.zeros((len(loops), 2), dtype=np.float32)

    groups = [names.index(group.name) for group in obj.vertex_groups]
    influences = [(vertex.index, groups[group.group], group.weight)
                  for vertex in mesh.vertices for group in vertex.groups if group.weight > 0]
    indices = np.asarray([(vertex, joint) for vertex, joint, _ in influences], dtype=np.int32)
    weights = np.asarray([weight for _, _, weight in influences], dtype=np.float32)
    totals = np.bincount(indices[:, 0], weights=weights, minlength=len(vertices))
    if (indices[:, 1] == 0).any() or not np.allclose(totals, 1., atol=.01):
        raise ValueError('Invalid skin weights or unexpected root-bound vertices.')
    np.savez_compressed(output, vertices=vertices, triangles=triangles, normals=normals.astype(np.float32),
                        uv=uv, weights=weights, indices=indices)
    print(f'{filename.name}: {len(vertices)} vertices, {len(triangles)} triangles, {len(indices)} skin influences')


def main():
    source, manny, quinn, output = map(Path, sys.argv[sys.argv.index('--') + 1:])
    profiles = json.loads(source.read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    for name, filename in (('manny', manny), ('quinn', quinn)):
        extract(filename, profiles['profiles'][name], output / (name + '.npz'))
    (output / 'reference.json').write_text(json.dumps(profiles, separators=(',', ':')) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
