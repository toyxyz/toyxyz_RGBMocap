"""Read the stock UE5 mannequin meshes using Unreal's Python commandlet."""
import json
from pathlib import Path

import unreal


def read(label):
    asset = unreal.load_asset('/Game/Characters/Mannequins/Meshes/SKM_' + label + '_Simple')
    if not isinstance(asset, unreal.SkeletalMesh) or asset.get_editor_property('skeleton').get_name() != 'SK_Mannequin':
        raise RuntimeError('Cannot load the Unreal Third Person mannequin: ' + label)
    mesh = unreal.SkeletalMeshComponent()
    mesh.set_skeletal_mesh_asset(asset)
    bones = []
    for index in range(mesh.get_num_bones()):
        name = mesh.get_bone_name(index)
        transform = mesh.get_ref_pose_transform(index)
        p, q, s = transform.translation, transform.rotation, transform.scale3d
        if max(abs(s.x-1), abs(s.y-1), abs(s.z-1)) > 1e-4:
            raise ValueError('Non-rigid mannequin reference bone: ' + str(name))
        bones.append(dict(name=str(name), parent=mesh.get_bone_index(mesh.get_parent_bone(name)),
                          position=[p.x,p.y,p.z], rotation=[q.x,q.y,q.z,q.w]))
    return bones


if __name__ == '__main__':
    path = Path(__file__).with_name('profiles.json')
    path.write_text(json.dumps(dict(version=1, profiles={label.lower():read(label)
                                                        for label in ('Manny','Quinn')})),encoding='utf-8')
