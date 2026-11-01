"""Blender-only builder for a portable camera/video/skeleton scene."""
import json,sys
from pathlib import Path
import bpy
from mathutils import Matrix

folder=Path(sys.argv[sys.argv.index('--')+1])
job=json.loads((folder/'blend-job.json').read_text(encoding='utf-8'))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(folder/'animation.fbx'),anim_offset=0,
                         automatic_bone_orientation=False,use_prepost_rot=True,ignore_leaf_bones=True)
scene=bpy.context.scene
fps=float(job['fps']);scene.render.fps=round(fps);scene.render.fps_base=round(fps)/fps
scene.render.resolution_x=job['width'];scene.render.resolution_y=job['height'];scene.render.resolution_percentage=100
start=int(job['prepend_tpose']);scene.frame_start=0;scene.frame_end=job['frames']-1+start
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
arms=[o for o in scene.objects if o.type=='ARMATURE']
if len(arms)!=1:raise ValueError('Expected one exported source armature')
arm=arms[0];arm.show_in_front=True;arm.data.display_type='OCTAHEDRAL'
camera=bpy.data.cameras.new('Source video camera');obj=bpy.data.objects.new(camera.name,camera)
scene.collection.objects.link(obj);scene.camera=obj
# FBX Y-up world -> Blender Z-up; CV camera (+Z forward, +Y down) -> Blender camera.
world=Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))
cv=Matrix.Diagonal((1,-1,-1,1))
obj.matrix_world=world@Matrix(job['world_to_camera']).inverted()@cv
if job.get('camera_track'):
    import numpy as np
    track=np.load(folder/job['camera_track'],allow_pickle=False,mmap_mode='r')
    if track.shape!=(job['frames'],4,4):raise ValueError('Invalid camera track length')
    obj.rotation_mode='QUATERNION';previous=None
    for frame in range(job['frames']):
        matrix=world@Matrix(track[frame].tolist()).inverted()@cv
        location,rotation,scale=matrix.decompose()
        if previous is not None and rotation.dot(previous)<0:rotation.negate()
        obj.location=location;obj.rotation_quaternion=rotation;previous=rotation.copy()
        obj.keyframe_insert(data_path='location',frame=frame+start)
        obj.keyframe_insert(data_path='rotation_quaternion',frame=frame+start)
        if frame==0 and start:
            obj.keyframe_insert(data_path='location',frame=0)
            obj.keyframe_insert(data_path='rotation_quaternion',frame=0)
    for curve in obj.animation_data.action.fcurves:
        for key in curve.keyframe_points:key.interpolation='LINEAR'
k=job['K'];width=job['width'];height=job['height']
camera.type='PERSP';camera.sensor_fit='HORIZONTAL';camera.sensor_width=36
camera.lens=k[0][0]*36/width
scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=k[0][0]/k[1][1]
camera.shift_x=(width/2-k[0][2])/width
camera.shift_y=(k[1][2]-height/2)*scene.render.pixel_aspect_y/width
camera.clip_start=.01;camera.clip_end=10000
camera.show_background_images=True
background=camera.background_images.new();background.source='IMAGE'
clip=bpy.data.images.load(str(folder/'video.mp4'));clip.filepath='//video.mp4'
background.image=clip
background.image_user.frame_start=start
background.image_user.frame_duration=job['frames']
background.image_user.use_auto_refresh=True
background.image_user.use_cyclic=False
background.alpha=1;background.display_depth='BACK';background.frame_method='FIT'
scene['RGBMoCap video start']=start
scene['RGBMoCap camera']=job.get('camera_mode','Stored reference camera')
scene['RGBMoCap video']='Keep video.mp4 next to this Blender project'
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            space=area.spaces.active;space.region_3d.view_perspective='CAMERA'
            space.overlay.show_floor=False;space.overlay.show_axis_x=False;space.overlay.show_axis_y=False
for item in bpy.context.selected_objects:item.select_set(False)
arm.select_set(True);bpy.context.view_layer.objects.active=arm
scene.frame_set(start)
bpy.ops.wm.save_as_mainfile(filepath=str(folder/job['target']),check_existing=False)
