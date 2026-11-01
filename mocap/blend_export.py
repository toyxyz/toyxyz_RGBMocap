"""Portable source-animation Blender bundles, built in a separate process."""
import json, os, shutil, subprocess
from pathlib import Path
import numpy as np
from .result_artifacts import file_stat
from .motion_tools import selected_export


def camera_record(project):
    from .scene_camera import read_camera, validate_transform
    camera=read_camera(project)
    if camera is not None:
        return camera, [project/'scene_camera.json']
    # A single-person project may not have a shared-scene receipt.
    # Use the paired native camera and world roots, without running inference.
    import torch
    from scipy.spatial.transform import Rotation
    from .preview_data import mapped_array
    from .scene_camera import paired_camera
    source=project/'hmr4d_results.pt'
    data=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    if data.get('source_model')=='flowhmr':
        raise ValueError('Shared FlowHMR camera is unavailable. Prepare Preview before exporting a Blender project.')
    world,incam=data['smpl_params_global'],data['smpl_params_incam']
    joints=mapped_array(project/'initial_joints.npz','joints')
    try:pelvis=np.asarray(joints[0,0])-world['transl'][0].numpy()
    finally:joints.close()
    rot=lambda p:Rotation.from_rotvec(p['global_orient'][0].numpy()).as_matrix()
    matrix=paired_camera(rot(world),rot(incam),world['transl'][0].numpy(),incam['transl'][0].numpy(),pelvis)
    with np.load(project/'tracking.npz',allow_pickle=False) as tracking:k=tracking['K'].tolist()
    validate_transform(matrix)
    return dict(world_to_camera=matrix.tolist(),K=k),[source,project/'initial_joints.npz',project/'tracking.npz']


def run(request,emit=lambda *a,**k:None):
    from .fbx_export import run as export_fbx
    from .scene_export import context
    from .retarget.runtime import blender
    from .scene_camera import validate_transform
    from .video import probe
    request=Path(request)
    if request.stat().st_size>4*1024**2:raise ValueError('Export request is too large')
    rows=json.loads(request.read_text(encoding='utf-8'))
    if not isinstance(rows,list) or not 1<=len(rows)<=4096:raise ValueError('Invalid export request')
    staged=[]
    try:
        for i,row in enumerate(rows):
            project=Path(row['project']);kind=row['kind'];target=Path(row['target'])
            if target.suffix.lower()!='.blend':raise ValueError('Choose a Blender destination')
            folder=target.parent
            if folder.exists():raise ValueError('Export folder already exists. Choose a new name: '+str(folder))
            if any(folder.resolve()==item[1].resolve() for item in staged):raise ValueError('Duplicate export folder')
            if selected_export(project,kind)!=Path(row['source']):raise ValueError('Selected motion changed. Export again.')
            video=project/'input.mp4'
            info=json.loads((project/(kind+'_mixamo.json')).read_text(encoding='utf-8'))
            camera,camera_paths=camera_record(project)
            track=None;track_stamps=[];camera_mode='Stored reference camera (tracking disabled)'
            alignment=None;background_report=None
            if camera.get('source_model')=='flowhmr':
                from .camera_alignment import align
                aligned,alignment_stamps,alignment=align(project,camera['world_to_camera'],kind)
                camera={**camera,'world_to_camera':aligned.tolist()}
                track_stamps+=alignment_stamps
                camera_paths += [p for p,_ in alignment_stamps]
            if row.get('animate_camera',True):
                from .blend_camera_track import load
                scale=float(row.get('camera_translation_scale',1.0)) if row.get('camera_translation',False) else 0.0
                track,tracking_stamps,camera_mode=load(project,camera['world_to_camera'],info['frames'],translation_scale=scale)
                track_stamps+=tracking_stamps
                camera_paths += [p for p,_ in tracking_stamps]
            elif row.get('camera_translation',False):raise ValueError('Enable camera tracking before experimental translation')
            if row.get('background_refine',False):
                if track is None or not row.get('animate_camera',True):raise ValueError('Background refinement requires saved camera tracking')
                from .camera_background import refine
                track,extra,background_report=refine(project,track,camera['K'])
                track_stamps+=extra;camera_paths += [p for p,_ in extra]
                camera_mode+='; '+background_report['reason']
            paths=[video,project/'project.json',project/(kind+'_mixamo.json'),Path(row['source']),*camera_paths]
            for key in ('asset_file','frames_file'):
                if Path(info[key]).name!=info[key]:raise ValueError('Invalid committed file path')
                paths.append(project/info[key])
            stamps=[file_stat(p) for p in paths]
            if any(file_stat(p)!=stamp for p,stamp in track_stamps):raise ValueError('Camera changed during export')
            media=probe(video)
            if media['frames']!=info['frames'] or abs(media['fps']-info['fps'])>.001:
                raise ValueError('Video and motion timing differ. Prepare the matching take first.')
            scene=row.get('scene')
            transform=np.eye(4) if scene is None else validate_transform(scene['transform'])
            if scene is not None and context(scene['root'],project)!=scene:raise ValueError('Scene placement changed')
            w2c=validate_transform(camera['world_to_camera'])@np.linalg.inv(transform)
            temporary=folder.with_name('.'+folder.name+'.rgbmocap-'+request.stem+'.partial')
            temporary.mkdir()  # Exclusive staging; never overwrite an existing bundle.
            staged.append((temporary,folder,project,kind,paths,stamps,scene))
            emit('progress',stage='save_blend',current=i,total=len(rows),message='Copying synchronized video')
            shutil.copy2(video,temporary/'video.mp4')
            nested=temporary/'fbx-request.json'
            nested.write_text(json.dumps([{**row,'target':str(temporary/'animation.fbx')}]),encoding='utf-8')
            export_fbx(nested,lambda *a,**k:None)
            job=dict(target=target.name,frames=info['frames'],fps=info['fps'],width=media['width'],height=media['height'],
                     world_to_camera=w2c.tolist(),K=camera['K'],prepend_tpose=bool(row.get('prepend_tpose')))
            job['camera_mode']=camera_mode
            job['camera_alignment']=alignment
            if track is not None:
                np.save(temporary/'camera.npy',track@np.linalg.inv(transform))
                job['camera_track']='camera.npy'
            (temporary/'blend-job.json').write_text(json.dumps(job),encoding='utf-8')
            emit('progress',stage='save_blend',current=i,total=len(rows),message='Building Blender project')
            command=[str(blender()),'-b','--factory-startup','--disable-autoexec','--threads','3',
                     '--python-exit-code','1','--python',str(Path(__file__).with_name('blend_export_scene.py')),'--',str(temporary)]
            with (temporary/'blender.log').open('wb') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode or not (temporary/target.name).is_file():
                detail=(temporary/'blender.log').read_text(encoding='utf-8',errors='replace')[-3000:]
                raise RuntimeError('Blender export failed: '+detail)
            (temporary/'camera_report.json').write_text(json.dumps(dict(camera_mode=camera_mode,alignment=alignment,background_refinement=background_report),indent=2),encoding='utf-8')
            for name in ('animation.fbx','blend-job.json','blender.log','camera.npy'):(temporary/name).unlink(missing_ok=True)
            (temporary/'README.txt').write_text('Open '+target.name+' in Blender 4.5 or newer.\nKeep video.mp4 beside the .blend file when moving this folder.\nNumpad 0: camera view. Space: playback. Camera background is a viewport reference, not a rendered image.\nCamera: '+camera_mode+'. See camera_report.json for refinement details.\n',encoding='utf-8')
        for _,_,project,kind,paths,stamps,scene in staged:
            if [file_stat(p) for p in paths]!=stamps or selected_export(project,kind)!=paths[3]:raise ValueError('Export inputs changed. Retry.')
            if scene is not None and context(scene['root'],project)!=scene:raise ValueError('Scene placement changed. Retry.')
        for temporary,folder,*_ in staged:temporary.rename(folder)
        return len(staged)
    finally:
        for temporary,folder,*_ in staged:
            # Only the exact staging sibling created by this request is removable.
            if temporary.parent.resolve()==folder.parent.resolve() and temporary.name.endswith('.'+ 'partial') and temporary.exists():
                shutil.rmtree(temporary)
        request.unlink(missing_ok=True)
