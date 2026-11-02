"""PR #18-derived Blender-only portable multi-view scene builder. No playback Python handlers."""
import json,sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Vector,Quaternion

def progress(current,total,message):
    print('@@BLEND@@'+json.dumps(dict(current=current,total=total,message=message)),flush=True)

folder=Path(sys.argv[sys.argv.index('--')+1]);job=json.loads((folder/'scene-job.json').read_text(encoding='utf-8'))
progress(0,0,'Importing source animation')
bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene
scene.frame_start=0;scene.frame_end=job['frames']-1
scene.render.fps=round(job['fps']);scene.render.fps_base=round(job['fps'])/job['fps']
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
def collection(name):
    c=bpy.data.collections.new(name);scene.collection.children.link(c);return c
bodies=collection('Animated bodies');rigs=collection('Reference skeletons');cameras=collection('Calibrated cameras');panels=collection('Video panels');stage=collection('Presentation')
def move(obj,c):
    for old in list(obj.users_collection):old.objects.unlink(obj)
    c.objects.link(obj)
def visibility(obj,first,end):
    for f,hide in [(-1,True),(first,False),(end,True)]:
        for prop in ('hide_viewport','hide_render'):
            setattr(obj,prop,hide);obj.keyframe_insert(data_path=prop,frame=f)
    if obj.animation_data:
        for curve in obj.animation_data.action.fcurves:
            if curve.data_path in ('hide_viewport','hide_render'):
                for p in curve.keyframe_points:p.interpolation='CONSTANT'
for actor_index,a in enumerate(job['actors']):
    progress(actor_index,len(job['actors']),'Building animated body: '+a['name'])
    before=set(scene.objects)
    bpy.ops.import_scene.fbx(filepath=str(folder/a['fbx']),anim_offset=a['first'],automatic_bone_orientation=False,use_prepost_rot=True,ignore_leaf_bones=True)
    for obj in set(scene.objects)-before:
        move(obj,rigs);obj.show_in_front=False;visibility(obj,a['first'],a['end'])
    with np.load(folder/a['geometry']) as z:verts=z['vertices'];faces=z['faces']
    mesh=bpy.data.meshes.new(a['name']);mesh.from_pydata(verts.tolist(),[],faces.tolist());mesh.update()
    obj=bpy.data.objects.new(a['name']+' body',mesh);bodies.objects.link(obj)
    for face in mesh.polygons:face.use_smooth=True
    color=tuple(int(a['color'][i:i+2],16)/255 for i in (1,3,5))+(1.,)
    mat=bpy.data.materials.new(a['name']+' color');mat.diffuse_color=color;mat.use_nodes=True
    mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=color
    mat.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.65
    mesh.materials.append(mat);obj.color=color
    mod=obj.modifiers.new('Exact recorded surface','MESH_CACHE');mod.cache_format='PC2';mod.filepath='//'+a['cache']
    mod.play_mode='SCENE';mod.time_mode='FRAME';mod.frame_start=a['first'];mod.frame_scale=1.;mod.forward_axis='POS_Y';mod.up_axis='POS_Z'
    mod.deform_mode='OVERWRITE';visibility(obj,a['first'],a['end'])
camera_objects={}
world=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)));cv=Matrix.Diagonal((1,-1,-1,1))
for camera_index,c in enumerate(job['cameras']):
    progress(camera_index,len(job['cameras']),'Building calibrated camera: '+c['name'])
    data=bpy.data.cameras.new(c['name']);obj=bpy.data.objects.new(c['name'],data);cameras.objects.link(obj)
    obj.matrix_world=world@Matrix(c['world_to_camera']).inverted()@cv
    camera_objects[c['name']]=obj
    k=c['K'];w=c['width'];h=c['height'];aspect=k[0][0]/k[1][1]
    obj['video_width']=w;obj['video_height']=h;obj['pixel_aspect_y']=aspect
    data.sensor_fit='HORIZONTAL';data.sensor_width=36;data.lens=k[0][0]*36/w
    data.shift_x=(w/2-k[0][2])/w;data.shift_y=(k[1][2]-h/2)*aspect/w
    data.clip_start=.01;data.clip_end=10000;data.display_size=.3;obj.show_name=True
    if scene.camera is None:scene.camera=obj
    image=bpy.data.images.load(str(folder/c['video']));image.filepath='//'+c['video']
    data.show_background_images=True;bg=data.background_images.new();bg.source='IMAGE';bg.image=image
    bg.image_user.frame_start=0;bg.image_user.frame_duration=job['frames'];bg.image_user.use_auto_refresh=True;bg.alpha=1;bg.display_depth='BACK';bg.frame_method='FIT'
    depth=.8
    corners=[((u-k[0][2])/k[0][0]*depth,-(v-k[1][2])/k[1][1]*depth,-depth) for u,v in [(0,h),(w,h),(w,0),(0,0)]]
    mesh=bpy.data.meshes.new(c['name']+' screen');mesh.from_pydata(corners,[],[(0,1,2,3)]);mesh.uv_layers.new()
    for loop,uv in zip(mesh.uv_layers.active.data,[(0,0),(1,0),(1,1),(0,1)]):loop.uv=uv
    screen=bpy.data.objects.new(c['name']+' video',mesh);panels.objects.link(screen);screen.parent=obj
    mat=bpy.data.materials.new(c['name']+' video');mat.use_nodes=True;nodes=mat.node_tree.nodes;nodes.clear()
    tex=nodes.new('ShaderNodeTexImage');tex.image=image;tex.image_user.frame_start=0;tex.image_user.frame_duration=job['frames'];tex.image_user.use_auto_refresh=True;tex.interpolation='Linear'
    emission=nodes.new('ShaderNodeEmission');output=nodes.new('ShaderNodeOutputMaterial');mat.node_tree.links.new(tex.outputs['Color'],emission.inputs['Color']);mat.node_tree.links.new(emission.outputs[0],output.inputs['Surface']);mesh.materials.append(mat)
# Linked scenes preserve each camera's own aspect ratio without playback scripts.
scene.render.resolution_percentage=100;scene.render.pixel_aspect_x=1
primary=job['cameras'][0]
scene.render.resolution_x=primary['width'];scene.render.resolution_y=primary['height']
ratio=primary['K'][0][0]/primary['K'][1][1]
scene.render.pixel_aspect_x=max(1.,1./ratio);scene.render.pixel_aspect_y=max(1.,ratio)
points=np.array(job['bounds']);lo=points.min(0);hi=points.max(0);center=(lo+hi)/2;span=max(3.,float(np.max(hi-lo)))
bpy.ops.mesh.primitive_plane_add(size=max(12.,span*3),location=(center[0],center[1],min(0.,lo[2])-.01));floor=bpy.context.object;floor.name='Reference floor (not reconstructed)';move(floor,stage)
mat=bpy.data.materials.new('Floor');mat.use_nodes=True;nodes=mat.node_tree.nodes;checker=nodes.new('ShaderNodeTexChecker');checker.inputs['Color1'].default_value=(.16,.18,.20,1);checker.inputs['Color2'].default_value=(.24,.26,.28,1);checker.inputs['Scale'].default_value=12
mat.node_tree.links.new(checker.outputs['Color'],nodes.get('Principled BSDF').inputs['Base Color']);floor.data.materials.append(mat)
world_data=bpy.data.worlds.new('Studio');scene.world=world_data;world_data.use_nodes=True;world_data.node_tree.nodes.get('Background').inputs[0].default_value=(.12,.12,.12,1)
for x,y,z,power in [(-3,-4,7,1600),(4,1,5,1000)]:
    light=bpy.data.lights.new('Studio area','AREA');light.energy=power;light.shape='DISK';light.size=5
    obj=bpy.data.objects.new(light.name,light);stage.objects.link(obj);obj.location=Vector(center)+Vector((x,y,z));obj.rotation_euler=(Vector(center)-obj.location).to_track_quat('-Z','Y').to_euler()
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            space=area.spaces.active;space.region_3d.view_perspective='PERSP';space.region_3d.view_location=Vector(center);space.region_3d.view_distance=span*2.5;space.region_3d.view_rotation=Quaternion((.88,.28,.12,.35)).normalized();space.shading.type='MATERIAL';space.overlay.show_floor=False
scene['RGBMoCap scene']='MAMMA shared world; baked surfaces and separate reference rigs; keep videos/ and meshes/'
for c in job['cameras']:
    view=bpy.data.scenes.new('Camera - '+c['name'])
    for coll in (bodies,rigs,cameras,stage):view.collection.children.link(coll)
    view.camera=camera_objects[c['name']];view.world=scene.world
    view.frame_start=0;view.frame_end=job['frames']-1
    view.render.fps=scene.render.fps;view.render.fps_base=scene.render.fps_base
    view.render.resolution_percentage=100;view.render.resolution_x=c['width'];view.render.resolution_y=c['height']
    ratio=c['K'][0][0]/c['K'][1][1]
    view.render.pixel_aspect_x=max(1.,1./ratio);view.render.pixel_aspect_y=max(1.,ratio)
    view.unit_settings.system='METRIC';view.unit_settings.scale_length=1
scene.name='Overview - all cameras'
scene.frame_set(0);bpy.context.view_layer.update()
progress(0,0,'Saving portable Blender project')
bpy.ops.wm.save_as_mainfile(filepath=str(folder/job['target']),check_existing=False)
