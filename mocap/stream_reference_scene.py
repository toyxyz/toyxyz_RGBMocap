"""Standard FBX writer for the Live Link reference; runs only in Blender."""
from pathlib import Path
import sys,json
import bpy
from mathutils import Matrix,Vector


def display_mesh(armature,schema):
    """Bind simple reference geometry in Blender, in the armature's local space."""
    bones=armature.data.bones
    lengths=dict(zip(schema['names'],schema['lengths']))
    if set(bones.keys())!=set(lengths):raise ValueError('Reference bone names changed during import.')
    # Only the imported skeleton is authoritative. Rebuild display geometry and
    # weights here so the final writer owns consistent bind matrices throughout.
    for obj in list(bpy.context.scene.objects):
        if obj.type=='MESH':bpy.data.objects.remove(obj,do_unlink=True)
    vertices=[];polygons=[];groups=[]
    faces=((0,2,3),(0,3,4),(0,4,5),(0,5,2),(1,3,2),(1,4,3),(1,5,4),(1,2,5))
    for bone in bones:
        head=bone.head_local.copy()
        direction=(sum((child.head_local for child in bone.children),Vector())/len(bone.children)-head
                   if bone.children else bone.matrix_local.to_3x3()@Vector((0.,0.,lengths[bone.name])))
        if direction.length<1e-7:direction=bone.matrix_local.to_3x3()@Vector((0.,0.,lengths[bone.name]))
        unit=direction.normalized()
        seed=min((Vector((1.,0.,0.)),Vector((0.,1.,0.)),Vector((0.,0.,1.))),key=lambda axis:abs(unit.dot(axis)))
        side=unit.cross(seed).normalized();other=unit.cross(side)
        radius=max(.04,min(direction.length*.08,1.5));ring=head+direction*.2
        offset=len(vertices)
        vertices.extend((head,head+direction,ring+side*radius,ring+other*radius,ring-side*radius,ring-other*radius))
        polygons.extend(tuple(offset+i for i in face) for face in faces)
        groups.append((bone.name,list(range(offset,offset+6))))
    mesh=bpy.data.meshes.new('ReferenceBones');mesh.from_pydata(vertices,[],polygons);mesh.update()
    obj=bpy.data.objects.new('ReferenceBones',mesh);bpy.context.collection.objects.link(obj)
    obj.parent=armature;obj.matrix_parent_inverse=Matrix.Identity(4);obj.matrix_basis=Matrix.Identity(4)
    for name,indices in groups:obj.vertex_groups.new(name=name).add(indices,1.,'REPLACE')
    modifier=obj.modifiers.new('ReferenceSkin','ARMATURE');modifier.object=armature


def export(work):
    def progress(value,text):
        print('@@BLEND@@'+json.dumps(dict(current=value,total=3,message=text)),flush=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    progress(0,'Loading exact Live Link reference rig')
    bpy.ops.import_scene.fbx(filepath=str(work/'source.fbx'),automatic_bone_orientation=False,
                            ignore_leaf_bones=False,primary_bone_axis='Z',secondary_bone_axis='X')
    armatures=[obj for obj in bpy.context.scene.objects if obj.type=='ARMATURE']
    if len(armatures)!=1:raise ValueError('Expected one reference armature.')
    armature=armatures[0]
    # Unreal recognizes this container name and preserves Hips as the sole root.
    armature.name='Armature'
    display_mesh(armature,json.loads((work/'rig.json').read_text(encoding='utf-8')))
    bpy.context.scene.unit_settings.system='METRIC'
    bpy.context.scene.unit_settings.scale_length=.01
    for obj in bpy.context.scene.objects:
        if obj.parent is None:obj.matrix_world=Matrix.Scale(100.,4)@obj.matrix_world
    progress(1,'Writing skinned reference FBX')
    result=bpy.ops.export_scene.fbx(filepath=str(work/'reference.fbx'),use_selection=False,
        object_types={'ARMATURE','MESH'},add_leaf_bones=False,bake_anim=False,
        axis_forward='-Y',axis_up='Z',primary_bone_axis='Z',secondary_bone_axis='X',
        apply_unit_scale=True,apply_scale_options='FBX_SCALE_ALL',use_armature_deform_only=False)
    if result!={'FINISHED'}:raise RuntimeError('Reference FBX export did not finish.')
    progress(3,'Reference rig FBX written')


if __name__=='__main__':
    export(Path(sys.argv[sys.argv.index('--')+1]))
