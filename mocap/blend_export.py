"""PR #18-derived portable source animation, source video and camera bundles."""
import json
from pathlib import Path
import numpy as np
from .blend_support import Inputs,Bundle,read,copy_video,build,disk_check


def run(request,emit=lambda *a,**k:None):
    from .motion_tools import selected_export
    from .scene_export import context
    from .fbx_export import run as export_fbx
    from .camera_tracks import load,at
    from .camera_background_worker import refined,current
    from .video import probe
    request=Path(request);bundles=[];checks=[];cache={}
    try:
        rows=read(request)
        if not isinstance(rows,list) or not 1<=len(rows)<=4096:raise ValueError('Invalid Blender export request')
        from .gemx_result import is_project
        if any(is_project(row['project']) for row in rows):
            if len(rows)!=1:raise ValueError('Export one native SOMA result at a time.')
            from .gemx_blend import run as native_export
            return native_export(rows[0],request,emit)
        folders=set()
        for row in rows:
            folder=Path(row['target']).resolve().parent
            if str(folder).casefold() in folders:raise ValueError('Duplicate Blender bundle destination')
            folders.add(str(folder).casefold())
        for row in rows:
            project=Path(row['project']).resolve();kind=row['kind'];source=Path(row['source']).resolve();inputs=Inputs()
            if selected_export(project,kind)!=source:raise ValueError('Selected motion changed. Prepare Preview or Apply Correction first.')
            state=read(inputs.track(project/'project.json'));info=read(inputs.track(project/(kind+'_mixamo.json')))
            for key in ('asset_file','frames_file'):
                if Path(info[key]).name!=info[key]:raise ValueError('Invalid committed animation path')
                inputs.track(project/info[key])
            inputs.track(source);scene=row.get('scene')
            if scene is not None and context(scene['root'],project)!=scene:raise ValueError('Shared scene placement changed')
            group=scene['root'] if scene is not None else None
            cameras=load(project,group)
            if row.get('background_refine',False) and row.get('animate_camera',True):
                cameras=refined(cameras,project,lambda:False,
                    lambda done,total:emit('progress',stage='save_blend',current=done,total=total,message='Refining camera rotation'),cache)
            for item in cache.values():
                if not current(item['sources']):raise ValueError('Background camera inputs changed during export')
                for p,_ in item['sources']:inputs.track(p)
            if not current(list(cameras['camera_sources'].items())):raise ValueError('Camera sources changed during export')
            for p in cameras['camera_sources']:inputs.track(p)
            camera=cameras['cameras'][0];n=int(info['frames']);fps=float(info['fps'])
            if abs(camera['K'][0,1])>1e-6:raise ValueError('Blender export requires camera intrinsics without skew')
            if n!=int(state['frames']) or abs(fps-state['fps'])>1e-6:raise ValueError('Video and selected animation timing differ')
            scale=float(row.get('camera_translation_scale',1.))
            if not np.isfinite(scale) or not 0<=scale<=100:raise ValueError('Invalid camera movement scale')
            animate=row.get('animate_camera',True)
            times=state['start']+np.arange(n)/fps
            track=[at(camera,t,scale if animate else 0.) for t in times]
            if any(m is None for m in track):raise ValueError('Saved camera does not cover the selected motion')
            track=np.asarray(track)
            if not animate:track[:]=track[0]
            video=inputs.track(project/'input.mp4');media=probe(video)
            if (media['frames']!=n or abs(media['fps']-fps)>.001
                    or [media['width'],media['height']]!=list(camera['size'])):
                raise ValueError('Source video timing or resolution differs from the saved camera')
            disk_check(Path(row['target']).parent.parent,video.stat().st_size+source.stat().st_size*4)
            bundle=Bundle(row['target'],request);bundles.append(bundle);work=bundle.work
            copy_video(video,work/'video.mp4',emit,'save_blend')
            nested=work/'fbx-request.json';nested.write_text(json.dumps([{**row,'target':str(work/'animation.fbx'),'place_at_origin':False,'symmetric_rest':False}]),encoding='utf-8')
            export_fbx(nested,emit)
            np.save(work/'camera.npy',track,allow_pickle=False)
            job=dict(target=bundle.target.name,frames=n,fps=fps,width=media['width'],height=media['height'],
                world_to_camera=track[0].tolist(),K=camera['K'].tolist(),prepend_tpose=bool(row.get('prepend_tpose')),
                camera_track='camera.npy' if animate and not np.array_equal(track,np.broadcast_to(track[0],track.shape)) else None,
                camera_mode=cameras['note'])
            (work/'blend-job.json').write_text(json.dumps(job),encoding='utf-8')
            emit('progress',stage='save_blend',current=0,total=0,message='Building Blender project')
            build(work,'blend_export_scene.py',emit,'save_blend')
            report=dict(model=cameras['model'],result=kind,frames=n,fps=fps,video_start_frame=int(job['prepend_tpose']),
                camera_mode=cameras['note'],camera_warning=cameras.get('warning',''),animate_camera=bool(animate),
                camera_movement_scale=scale,background_refinement=cameras.get('background_report'),
                camera_translation='Approximate display estimate, not measured metric camera motion')
            (work/'camera_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            (work/'README.txt').write_text('Open '+bundle.target.name+' in Blender 4.5 or newer.\n'
                'Keep video.mp4 beside the .blend file when moving this folder.\n'
                'Numpad 0: camera view. Space: playback. The video is a viewport background, not a render background.\n'
                'The source armature is editable. An optional T-pose is one extra animation frame; the video starts after it.\n'
                'Camera movement may include reconstruction error and arbitrary scale; it is not measured calibration.\n'
                'See camera_report.json. Source motion and original camera files were not changed.\n',encoding='utf-8')
            for name in ('animation.fbx','blend-job.json','blender.log','camera.npy'):(work/name).unlink(missing_ok=True)
            checks.append((inputs,project,kind,source,scene))
        for inputs,project,kind,source,scene in checks:
            inputs.verify()
            if selected_export(project,kind)!=source:raise ValueError('Motion changed during export')
            if scene is not None and context(scene['root'],project)!=scene:raise ValueError('Scene placement changed during export')
        for bundle in bundles:bundle.publish()
        return len(bundles)
    finally:
        for bundle in bundles:bundle.close()
        request.unlink(missing_ok=True)
