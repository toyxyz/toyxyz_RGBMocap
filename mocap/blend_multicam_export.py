"""Transactional MAMMA scene export; no inference and no project writes."""
import json, os, shutil, struct, subprocess
from pathlib import Path
import numpy as np
from .result_artifacts import file_stat, artifacts_current
from .motion_tools import selected_export, result_is_current


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(request, emit=lambda *a, **k: None):
    from .people_store import safe_child, load_people
    from .scene_export import context
    from .scene_camera import validate_transform, read_camera
    from .mamma_body import WORLD_ROTATION
    from .mesh_data import reference_vertices
    from .fbx_export import run as export_fbx
    from .retarget.runtime import blender
    from .video import probe
    request=Path(request);temporary=None
    if request.stat().st_size>4*1024**2:raise ValueError('Export request is too large')
    spec=read(request);root=Path(spec['root']).resolve();target=Path(spec['target']).resolve()
    if target.suffix.lower()!='.blend' or target.parent.exists():
        raise ValueError('Choose a new Blender bundle folder')
    tracked={};checks=[]
    def track(path):
        path=Path(path);tracked[path]=file_stat(path);return path
    try:
        state=read(track(root/'project.json'))
        if state.get('pipeline',{}).get('model')!='mamma':raise ValueError('Whole-scene export requires a MAMMA result')
        if (root/'mamma_pending.json').exists():raise ValueError('MAMMA publication is incomplete')
        work=(root/state['mamma_work']).resolve()
        workspace=(root/state['workspace_root']).resolve()
        if not work.is_relative_to(workspace):raise ValueError('Invalid MAMMA data path')
        cameras=read(track(work/'camera-world.json'));entries=read(track(work/'inputs.json'))
        if len(entries)<2 or set(cameras)!={e['id'] for e in entries}:raise ValueError('Camera calibration and videos differ')
        people=load_people(root);track(root/'people.json');track(root/'people_scene.json')
        ids=spec['people'];kind=spec['kind']
        if not ids or len(set(ids))!=len(ids):raise ValueError('Select at least one person')
        selected=[p for p in people['people'] if p['id'] in ids]
        if len(selected)!=len(ids):raise ValueError('Selected people changed')
        n=int(state['frames']);fps=float(state['fps']);actors=[]
        for person in selected:
            if not person['takes']:raise ValueError('Selected person has no animation')
            for take in person['takes']:
                p=safe_child(root,take['project']);meta=read(track(p/'project.json'))
                source=selected_export(p,kind)
                if source is None or not result_is_current(p,kind) or not artifacts_current(p,kind):
                    raise ValueError(person['name']+': selected result is stale. Apply Correction or Prepare Preview first.')
                track(source);scene=context(root,p)
                if not np.allclose(scene['transform'],np.eye(4),atol=1e-6):raise ValueError('MAMMA actors must share one native world')
                first,end=map(int,meta['person_frames'])
                if not 0<=first<end<=n or end-first!=meta['frames'] or abs(meta['fps']-fps)>1e-6:raise ValueError('Invalid actor timing')
                info=read(track(p/(kind+'_mesh.json')))
                if Path(info['frames_file']).name!=info['frames_file']:raise ValueError('Invalid mesh cache path')
                asset=track(Path(info['asset']));frames=track(p/info['frames_file'])
                if info.get('version')!=1 or info['frames']!=end-first:raise ValueError('Prepare the current mesh preview first')
                for suffix in ('_mixamo.json','_artifacts.json'):
                    track(p/(kind+suffix))
                checks.append((p,source,scene))
                actors.append(dict(project=p,name=person['name'],color=person['color'],first=first,end=end,asset=asset,frames=frames,source=source,scene=scene))
        primary=entries[0]['id'];camera_jobs=[]
        for i,e in enumerate(entries):
            name=e['id']
            if Path(name).name!=name or any(c in name for c in '/\\:'):raise ValueError('Invalid camera name')
            c=cameras[name];k=np.asarray(c['intrinsic_matrix'],float);ext=np.asarray(c['extrinsics_matrix'],float)
            if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0 or abs(k[0,1])>1e-6:raise ValueError('Unsupported camera intrinsics')
            if np.any(np.abs(c.get('distortion_coeffs',[0]))>1e-8):raise ValueError('This Blender scene export requires undistorted camera videos/calibration')
            if ext.shape!=(3,4):raise ValueError('Invalid camera extrinsics')
            w2c=np.eye(4);w2c[:3,:3]=ext[:,:3]@WORLD_ROTATION.T;w2c[:3,3]=ext[:,3];validate_transform(w2c)
            video=track(work/'videos'/(name+'.mp4'));media=probe(video)
            if media['frames']!=n or abs(media['fps']-fps)>.001 or [media['width'],media['height']]!=c['image_size']:raise ValueError('Camera video timing/resolution differs from calibration: '+name)
            camera_jobs.append(dict(name=name,K=k.tolist(),world_to_camera=w2c.tolist(),width=media['width'],height=media['height'],video=f'videos/camera_{i+1:03}.mp4',source=video))
        for p,_,_ in checks:
            camera=read_camera(p)
            if camera is None or not np.allclose(camera['world_to_camera'],camera_jobs[0]['world_to_camera'],atol=1e-6):raise ValueError('Primary camera changed. Prepare Preview first.')
        required=sum(file_stat(c['source'])[0] for c in camera_jobs)+sum((a['end']-a['first'])*10475*12 for a in actors)+512*1024**2
        if shutil.disk_usage(target.parent.parent).free<required:raise ValueError('Insufficient free space for mesh caches and videos')
        temporary=target.parent.with_name('.'+target.parent.name+'.rgbmocap-'+request.stem+'.partial');temporary.mkdir()
        (temporary/'videos').mkdir();(temporary/'meshes').mkdir();job_actors=[];bounds=[]
        for c in camera_jobs:shutil.copy2(c['source'],temporary/c['video'])
        for index,a in enumerate(actors):
            emit('progress',stage='save_blend_scene',current=index,total=len(actors),message='Baking exact body mesh: '+a['name'])
            stem=f'actor_{index+1:03}';cache='meshes/'+stem+'.pc2'
            with np.load(a['asset'],allow_pickle=False) as asset:
                packed=asset['packed'];faces=asset['faces'];count=int(asset['vertex_count'])
            frames=np.load(a['frames'],mmap_mode='r',allow_pickle=False)
            try:
                if frames.shape!=(a['end']-a['first'],1389):raise ValueError('Unexpected SMPL-X frame layout')
                with (temporary/cache).open('wb') as out:
                    out.write(struct.pack('<12siiffi',b'POINTCACHE2\0',1,count,0.,1.,len(frames)))
                    for f,frame in enumerate(frames):
                        vertices=reference_vertices(packed,frame,count)[:,[0,2,1]].copy();vertices[:,1]*=-1
                        if not np.isfinite(vertices).all():raise ValueError('Non-finite mesh vertices')
                        if f==0:
                            np.savez(temporary/(stem+'.npz'),vertices=vertices,faces=faces)
                        bounds.extend([vertices.min(0).tolist(),vertices.max(0).tolist()])
                        out.write(vertices.astype('<f4').tobytes())
                        if f%30==0:emit('progress',stage='save_blend_scene',current=f,total=len(frames),message='Mesh '+a['name'])
            finally:frames._mmap.close()
            nested=temporary/'fbx-request.json';nested.write_text(json.dumps([dict(project=str(a['project']),kind=kind,source=str(a['source']),target=str(temporary/(stem+'.fbx')),rig_name=a['name'],scene=a['scene'])]),encoding='utf8')
            export_fbx(nested)
            job_actors.append(dict(name=a['name'],color=a['color'],first=a['first'],end=a['end'],cache=cache,geometry=stem+'.npz',fbx=stem+'.fbx'))
        job=dict(target=target.name,frames=n,fps=fps,actors=job_actors,cameras=[{k:v for k,v in c.items() if k!='source'} for c in camera_jobs],bounds=bounds)
        (temporary/'scene-job.json').write_text(json.dumps(job),encoding='utf8')
        command=[str(blender()),'-b','--factory-startup','--disable-autoexec','--threads','3','--python-exit-code','1','--python',str(Path(__file__).with_name('blend_multicam_scene.py')),'--',str(temporary)]
        with (temporary/'blender.log').open('wb') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode or not (temporary/target.name).is_file():raise RuntimeError('Blender scene export failed: '+(temporary/'blender.log').read_text(errors='replace')[-2500:])
        for path,stamp in tracked.items():
            if file_stat(path)!=stamp:raise ValueError('Scene inputs changed during export. Retry.')
        for p,source,scene in checks:
            if selected_export(p,kind)!=source or not result_is_current(p,kind) or context(root,p)!=scene:raise ValueError('Selected motion changed during export')
        report=dict(model='mamma',result=kind,frames=n,fps=fps,cameras=job['cameras'],actors=job_actors,geometry='Exact baked SMPL-X surface, separate editable source skeletons',calibration='Saved MAMMA calibration; automatic scale/floor remain estimates')
        (temporary/'scene_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        (temporary/'README.txt').write_text('Open the .blend file in Blender 4.5+. Keep videos/ and meshes/ beside it.\nBodies use exact baked PC2 animation; armatures are separate editable reference skeletons, not mesh deformation rigs.\nAll cameras use the common MAMMA world. Video panels and camera backgrounds share the timeline.\nUse the scene selector: Overview shows all cameras; Camera - name scenes use matching image dimensions. Numpad 0 enters camera view.\nEach camera scene has its own output resolution and excludes video panels; camera backgrounds are viewport references.\nThe floor and lighting are presentation aids, not reconstructed scene geometry.\nNo T-pose is inserted in this synchronized scene export.\n',encoding='utf8')
        for a in job_actors:
            (temporary/a['geometry']).unlink();(temporary/a['fbx']).unlink()
        for name in ('scene-job.json','blender.log'):(temporary/name).unlink()
        temporary.rename(target.parent);temporary=None
        return 1
    finally:
        if temporary is not None and temporary.exists():
            expected=target.parent.with_name('.'+target.parent.name+'.rgbmocap-'+request.stem+'.partial')
            if temporary.resolve()!=expected.resolve() or temporary.parent.resolve()!=target.parent.parent.resolve():raise ValueError('Invalid export cleanup path')
            shutil.rmtree(temporary)
        request.unlink(missing_ok=True)
