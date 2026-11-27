"""PMX motion export using the bundled MMD Tools coordinate conversion.

Stream body/finger keys at MMD's fixed 30 Hz. No model, accessory animation,
camera, morph or physics data is generated. Outputs target the same PMX rig.
"""
import json,math,struct,sys,uuid
from pathlib import Path
from .file_paths import io_path
import bpy,addon_utils
from mathutils import Matrix,Quaternion


def name_bytes(name,size):
    try:raw=name.encode('cp932')
    except UnicodeEncodeError as exc:raise ValueError(f'VMD cannot encode bone name: {name}') from exc
    if len(raw)>size:raise ValueError(f'VMD bone name exceeds {size} bytes: {name}')
    return raw.ljust(size,b'\0')


def rotation_carrier(arm, root, compensate):
    """Keep native hidden grant helpers unkeyed; carry the waist turn above them.

    MMD ignores hidden bone animation but still evaluates its grants. Blender
    does not, so baking inverse grants onto hidden waist-cancel bones is unsafe.
    Move the whole-body transform to the existing trajectory root instead.
    This changes VMD channels only, never the stored rig or baked motion.
    """
    hidden = [arm.pose.bones[n] for n in compensate if arm.data.bones[n].hide]
    if not hidden:
        return None, []
    sources = {pb.mmd_bone.additional_transform_bone for pb in hidden}
    if len(sources) != 1:
        raise ValueError('VMD export cannot safely bake these hidden PMX grant controls.')
    waist = arm.pose.bones[next(iter(sources))]
    if (waist.mmd_bone.name_j or waist.name) != '腰' or waist == root or root.bone.hide:
        raise ValueError('VMD waist cancellation requires a visible ancestor motion root.')
    for pb in hidden:
        if (not pb.name.startswith('腰キャンセル') or
                abs(pb.mmd_bone.additional_transform_influence + 1.) > 1e-6 or
                'mmd_additional_location' in compensate[pb.name]):
            raise ValueError('Unsupported hidden PMX grant; refusing an incorrect VMD export.')
    chain = []
    parent = waist.parent
    while parent is not None and parent != root:
        chain.append(parent)
        parent = parent.parent
    if parent != root or any(not c.mute and c.influence > 0 for pb in [root, waist]+chain for c in pb.constraints):
        raise ValueError('VMD waist cancellation requires an unconstrained ancestor motion root.')
    return waist, chain


def identity_basis(bone):
    basis = bone.matrix_basis
    return all(abs(basis[row][col] - float(row == col)) <= 1e-5
               for row in range(4) for col in range(4))


def center_motion_bone(arm, root, names):
    error = 'Cannot export with Center (センター). Choose Current root for this PMX model.'
    candidates = [pb for pb in arm.pose.bones
                  if not pb.is_mmd_shadow_bone and (pb.mmd_bone.name_j or pb.name) == 'センター']
    if len(candidates) != 1 or root.parent is not None:
        raise ValueError(error)
    center = candidates[0]
    if center == root or center.bone.hide:
        raise ValueError(error)
    chain = []
    parent = center.parent
    while parent is not None and parent != root:
        chain.append(parent)
        parent = parent.parent
    if parent != root:
        raise ValueError(error)
    for pb in [root, center, *chain]:
        meta = pb.mmd_bone
        if (meta.has_additional_rotation or meta.has_additional_location
                or any(not c.mute and c.influence > 0 for c in pb.constraints)):
            raise ValueError(error)
    affected = {center.name, *(pb.name for pb in center.children_recursive)}
    if names - affected - {root.name}:
        raise ValueError(error)
    # Any vertex influenced by a bone outside Center would lose the root transform.
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or not any(m.type == 'ARMATURE' and m.object == arm for m in obj.modifiers):
            continue
        outside = {group.index for group in obj.vertex_groups
                   if group.name in arm.data.bones and group.name not in affected
                   and arm.data.bones[group.name].use_deform}
        if outside and any(weight.group in outside and weight.weight > 0
                           for vertex in obj.data.vertices for weight in vertex.groups):
            raise ValueError(error)
    # Native grants read channels directly. Moving those channels would also
    # change the grant, even when its recipient is already beneath Center.
    for pb in arm.pose.bones:
        meta = pb.mmd_bone
        if (meta.additional_transform_bone in {root.name, center.name}
                and (meta.has_additional_rotation or meta.has_additional_location)
                and abs(meta.additional_transform_influence) > 1e-6):
            raise ValueError(error)
    return center, chain


def export_vmd(folder, target, transform, label, placement='original', root_bone='root'):
    if placement not in ('original', 'origin', 'in-place'):
        raise ValueError('Unsupported VMD placement option.')
    if root_bone not in ('root', 'center'):
        raise ValueError('Unsupported VMD root bone option.')
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'third_party/character_import'))
    addon_utils.enable('mmd_tools',default_set=False,persistent=False)
    from mmd_tools.core.vmd.importer import BoneConverter
    from .mapping import A
    import numpy as np
    bpy.ops.wm.open_mainfile(filepath=io_path(folder/'animation.blend'),load_ui=False)
    arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    info=json.loads(Path(io_path(folder/'bake.json')).read_text(encoding='utf8'))
    rig=json.loads(Path(io_path(folder/'pmx-rig.json')).read_text(encoding='utf8'))
    names=set(info['keyed_bones']);compensate={}
    # Native PMX grant/cancel rules still run in MMD, even when the Blender
    # bake muted them. Cancel that contribution instead of applying it twice.
    for pb in arm.pose.bones:
        muted={c.name for c in pb.constraints if c.mute}
        if 'mmd_additional_rotation' in muted or 'mmd_additional_location' in muted:
            if pb.mmd_bone.additional_transform_bone:
                names.add(pb.name);compensate[pb.name]=muted
    root=arm.pose.bones[rig['trajectory_root']]
    root_rest = root.bone.matrix_local.copy()
    inverse_root_rest = root_rest.inverted()
    inverse_root_axes = inverse_root_rest.to_3x3()
    planar_offset = None
    waist, carrier_chain = rotation_carrier(arm, root, compensate)
    if waist is not None:
        names.add(root.name)
        names.add(waist.name)
        # These controls must remain at native identity; their grant now also
        # sees identity at the waist. Do not emit ineffective hidden keys.
        for name in list(compensate):
            if arm.data.bones[name].hide:
                compensate.pop(name)
                names.discard(name)
    center = None
    if root_bone == 'center':
        center, center_chain = center_motion_bone(arm, root, names)
        center_rest = center.bone.matrix_local.copy()
        root_to_center = center_rest.inverted() @ root_rest
        center_to_root = inverse_root_rest @ center_rest
        center_has_keys = center.name in names
        names.discard(root.name)
        names.add(center.name)
    bones=[pb for pb in arm.pose.bones if pb.name in names and not pb.is_mmd_shadow_bone]
    encoded=[name_bytes(pb.mmd_bone.name_j or pb.name,15) for pb in bones]
    if len(set(encoded))!=len(encoded):raise ValueError('Duplicate PMX names cannot be exported to VMD')
    needed=set(names)
    needed.update(arm.pose.bones[n].mmd_bone.additional_transform_bone for n in compensate)
    sample_bones=[pb for pb in arm.pose.bones if pb.name in needed]
    scale=json.loads(Path(io_path(folder/'import.json')).read_text(encoding='utf8')).get('mmd_import_scale',.08)
    converters={pb.name:BoneConverter(pb,1./scale,invert=True) for pb in sample_bones}
    ik=set()
    for pb in arm.pose.bones:
        for c in pb.constraints:
            if c.type=='IK' and c.subtarget in arm.pose.bones:
                b=arm.pose.bones[c.subtarget];ik.add(b.mmd_bone.name_j or b.name)
    if center is not None:
        root_label = root.mmd_bone.name_j or root.name
        if root_label in ik or any((pb.mmd_bone.name_j or pb.name) == root_label for pb in bones):
            raise ValueError('This PMX requires keys with the original root name. Choose Current root.')
    ik_names=[name_bytes(n,20) for n in sorted(ik)]
    scene=bpy.context.scene;fps=scene.render.fps/scene.render.fps_base
    start,end=scene.frame_start,scene.frame_end
    count=max(1,math.ceil((end-start)/fps*30-1e-8)+1)
    world=arm.matrix_world.copy()
    if transform is not None:
        C=np.eye(4);C[:3,:3]=A
        world=Matrix((C@np.asarray(transform)@C.T).tolist())@world
    temp=target.with_name('.'+target.stem+'-'+uuid.uuid4().hex+'.partial.vmd')
    previous={};linear=bytes([20]*4+[20]*4+[107]*4+[107]*4)*4
    try:
        with open(io_path(temp),'wb') as f:
            f.write(b'Vocaloid Motion Data 0002'.ljust(30,b'\0'))
            # The header is informational; bone names remain strict and exact.
            title=label.encode('cp932',errors='replace')[:20]
            while title:
                try:title.decode('cp932');break
                except UnicodeDecodeError:title=title[:-1]
            f.write(title.ljust(20,b'\0'));f.write(struct.pack('<I',count*len(bones)))
            for frame in range(count):
                sample=min(end,start+frame*fps/30)
                scene.frame_set(int(sample),subframe=sample-int(sample));bpy.context.view_layer.update()
                carrier_basis = None
                if waist is not None:
                    # A static, identity chain is required so that relocating
                    # the turn preserves every body pivot, not just orientation.
                    if any(not identity_basis(pb) for pb in carrier_chain):
                        raise ValueError('Animated PMX controls between root and waist require a separate VMD solve.')
                    deform = world @ waist.matrix @ waist.bone.matrix_local.inverted()
                    carrier_basis = inverse_root_rest @ deform @ root_rest
                root_basis = carrier_basis if carrier_basis is not None else inverse_root_rest @ world @ root.matrix
                if placement != 'original':
                    # Measure the final root after scene placement and waist grant compensation.
                    if planar_offset is None or placement == 'in-place':
                        planar_offset = (root_rest @ root_basis).translation
                        planar_offset.z = 0.0
                    root_basis.translation = root_basis.translation - inverse_root_axes @ planar_offset
                if center is not None:
                    if any(not identity_basis(pb) for pb in center_chain):
                        raise ValueError('Animated PMX controls above Center prevent transfer. Choose Current root.')
                    # VMD stores translation and rotation only.
                    root_motion = Matrix.LocRotScale(root_basis.translation, root_basis.to_quaternion().normalized(), None)
                    center_basis = root_to_center @ root_motion @ center_to_root
                    if center_has_keys:
                        center_basis = center_basis @ center.matrix_basis
                values={}
                for pb in sample_bones:
                    basis=pb.matrix_basis
                    if pb==root:
                        basis=root_basis
                    elif pb==center:
                        # Re-express the root transform around the Center pivot,
                        # then compose with Center's own animation.
                        basis=center_basis
                    elif pb==waist:
                        basis=Matrix.Identity(4)
                    q=basis.to_quaternion().normalized();conv=converters[pb.name]
                    values[pb.name]=(conv.convert_location(basis.translation),conv.convert_rotation((q.x,q.y,q.z,q.w)))
                for pb,encoded_name in zip(bones,encoded):
                    loc,q=values[pb.name];loc=loc.copy();q=q.copy()
                    if pb.name in compensate:
                        meta=pb.mmd_bone;grant_loc,grant_q=values[meta.additional_transform_bone]
                        muted=compensate[pb.name];weight=meta.additional_transform_influence
                        if 'mmd_additional_rotation' in muted:
                            grant=Quaternion(grant_q.axis,grant_q.angle*weight)
                            q=q@grant.inverted()
                        if 'mmd_additional_location' in muted:loc-=grant_loc*weight
                    q.normalize()
                    if pb.name in previous and q.dot(previous[pb.name])<0:q.negate()
                    previous[pb.name]=q.copy()
                    f.write(encoded_name);f.write(struct.pack('<I3f4f',frame,*loc,q.x,q.y,q.z,q.w));f.write(linear)
                if frame%60==0:print(f'CHARACTER: Saving VMD {frame+1}/{count}',flush=True)
            f.write(struct.pack('<4I',0,0,0,0)) # morph, camera, light, shadow
            f.write(struct.pack('<IIBI',1,0,1,len(ik_names)))
            for name in ik_names:f.write(name+b'\0') # disable IK: retain baked FK contacts
        Path(io_path(temp)).replace(io_path(target))
    finally:Path(io_path(temp)).unlink(missing_ok=True)
