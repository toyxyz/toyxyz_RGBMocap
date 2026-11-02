"""Read saved cameras without inference or body-based camera refitting."""
from pathlib import Path
import json,zipfile
import numpy as np
from .scene_camera import read_camera,validate_transform

MAX_BYTES=32*1024**2

def read(path):
    path=Path(path)
    if path.stat().st_size>2*1024**2:raise ValueError('Camera metadata exceeds the preview limit.')
    return json.loads(path.read_text(encoding='utf-8'))

def load_array(path,key=None):
    if key is not None:
        with zipfile.ZipFile(path) as z:
            entry=z.getinfo(key+'.npy')
            if entry.file_size>MAX_BYTES:raise ValueError('Camera track exceeds the preview memory limit.')
        with np.load(path,allow_pickle=False) as z:return np.array(z[key],dtype=float)
    a=np.load(path,mmap_mode='r',allow_pickle=False)
    try:
        if a.nbytes>MAX_BYTES:raise ValueError('Camera track exceeds the preview memory limit.')
        return np.array(a,dtype=float)
    finally:a._mmap.close()

def valid_rotations(r):
    if r.ndim!=3 or r.shape[1:]!=(3,3) or not len(r) or not np.isfinite(r).all():raise ValueError('Invalid camera rotations.')
    if not np.allclose(r.swapaxes(-1,-2)@r,np.eye(3),atol=2e-4) or not np.allclose(np.linalg.det(r),1,atol=2e-4):raise ValueError('Camera rotations are not rigid.')

def record(name,k,size,matrices,start,fps,description,displacement=None):
    k=np.asarray(k,float);m=np.asarray(matrices,float)
    if m.shape==(4,4):m=m[None]
    valid_rotations(m[:,:3,:3])
    if m.shape[1:]!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[:,3],[0,0,0,1]):raise ValueError('Invalid camera transforms.')
    if (k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0
            or not np.allclose(k[2],[0,0,1]) or abs(np.linalg.det(k))<1e-9):raise ValueError('Invalid camera intrinsics.')
    if len(size)!=2 or not np.isfinite(size).all() or min(size)<=0 or not np.isfinite(start) or not np.isfinite(fps) or fps<=0:raise ValueError('Invalid camera timing or image size.')
    return dict(name=name,K=k,size=tuple(size),matrices=m,start=float(start),fps=float(fps),description=description,displacement=displacement)

def at(camera,time,translation_scale=0.):
    m=camera['matrices'];index=round((time-camera['start'])*camera['fps'])
    if len(m)>1 and not 0<=index<len(m):return None
    index=max(0,min(len(m)-1,index));result=m[index]
    if translation_scale and camera['displacement'] is not None:
        result=result.copy();result[:3,3]-=result[:3,:3]@(camera['displacement'][index]*translation_scale)
    return result

def centers(camera,scale=0.):
    m=camera['matrices'];ids=np.unique(np.linspace(0,len(m)-1,min(len(m),256),dtype=int))
    result=-np.einsum('tji,tj->ti',m[ids,:3,:3],m[ids,:3,3])
    if scale and camera['displacement'] is not None:result+=scale*camera['displacement'][ids]
    return result

def geometry(camera,matrix,depth=.35):
    """CV +Z forward/+Y down, transported into the application's Y-up world."""
    k=camera['K'];w,h=camera['size']
    pixels=np.array([[0,0],[w,0],[w,h],[0,h],[w*.32,0],[w*.5,-h*.22],[w*.68,0]])
    local=np.c_[pixels,np.ones(7)]@np.linalg.inv(k).T
    local*=depth/local[:,2,None]
    local=np.vstack(([0.,0.,0.],local))
    world=(local-matrix[:3,3])@matrix[:3,:3]
    edges=np.array([(0,i) for i in range(1,5)]+[(1,2),(2,3),(3,4),(4,1),(5,6),(6,7),(7,5)])
    return world[edges],world[0]

def paired_reference(project,cancelled,cache):
    """Keep tensor deserialization outside the editor, including its loader thread."""
    import os,subprocess,time
    from .paths import worker_command
    from .result_artifacts import file_stat,artifacts_current
    if not artifacts_current(project,'initial'):raise ValueError('Original motion preview is stale. Use Prepare Preview.')
    prior=cache.get(str(project)) if cache is not None else None
    if prior and prior.get('schema')==2 and all(file_stat(project/name)==stamp for name,stamp in prior['sources'].items()):return prior
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    process=subprocess.Popen(worker_command('camera_reference','--project',str(project)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        env=environment,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    deadline=time.monotonic()+60
    try:
        while True:
            if cancelled():raise ValueError('Camera load cancelled.')
            if time.monotonic()>deadline:raise ValueError('Saved camera loading timed out. Retry after other processing finishes.')
            try:output,error=process.communicate(timeout=.1);break
            except subprocess.TimeoutExpired:pass
        if process.returncode:raise ValueError('Saved original camera is unavailable. Use Prepare Preview; check the original motion and body data.')
        if len(output)>MAX_BYTES:raise ValueError('Saved camera metadata exceeds the preview limit.')
        value=json.loads(output.decode('utf-8'))
        if value.get('centers') is not None:
            value['centers']=np.asarray(value['centers'],float)
            if value['centers'].ndim!=2 or value['centers'].shape[1]!=3 or value['centers'].nbytes>MAX_BYTES or not np.isfinite(value['centers']).all():
                raise ValueError('Invalid estimated camera centers.')
        if cache is not None:
            if len(cache)>=8:cache.pop(next(iter(cache)))
            cache[str(project)]=value
        return value
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.communicate(timeout=2)
            except subprocess.TimeoutExpired:process.kill();process.communicate()


def load(project,group=None,cancelled=lambda:False,reference_cache=None):
    from .result_artifacts import file_stat
    project=Path(project).resolve();state=read(project/'project.json');dependencies={}
    def tracked(p):
        p=Path(p);dependencies[p]=file_stat(p);return p
    tracked(project/'project.json')
    model=state.get('pipeline',{}).get('model');group=Path(group).resolve() if group else None
    had_scene=(project/'scene_camera.json').exists()
    camera=read_camera(project)
    if camera is None and group is None and model=='flowhmr' and not had_scene:
        # Single-person FlowHMR does not publish the multi-person scene receipt.
        # Reuse its committed WV basis/height, never refit to the corrected body.
        from .flow_placement import current
        if not current(project):raise ValueError('FlowHMR placement is stale. Use Prepare Preview.')
        world=read(tracked(project/'flow_world.json'))
        raw=load_array(tracked(project/'flow_camera.npz'),'camera_RT_yup')
        matrix=np.eye(4);matrix[:3,:3]=raw[0,:3,:3]@np.asarray(world['scene_basis'],float).T
        matrix[:3,3]=-matrix[:3,:3]@np.array([0.,world['camera_height'],0.])
        camera=dict(world_to_camera=validate_transform(matrix),K=world['intrinsics'])
    if camera is None and group is None and model in ('exmr','gvhmr') and not had_scene:
        camera=paired_reference(project,cancelled,reference_cache)
        for name,stamp in camera['sources'].items():dependencies[project/name]=stamp
    if camera is None:raise ValueError('Saved camera is unavailable or stale. Use Prepare Preview.')
    if had_scene:tracked(project/'scene_camera.json')
    reference=validate_transform(camera['world_to_camera'])
    transform=np.eye(4)
    if group is not None:
        from .scene_export import context
        placement=context(group,project);transform=validate_transform(placement['transform'])
        tracked(group/'people_scene.json')
    base=reference@np.linalg.inv(transform);items=[];note='';warning='';static_camera=False
    if model=='mamma':
        from .mamma_body import WORLD_ROTATION
        root=group or project;meta=read(tracked(root/'project.json'))
        work=(root/meta['mamma_work']).resolve();workspace=(root/meta['workspace_root']).resolve()
        if not work.is_relative_to(workspace):raise ValueError('Invalid MAMMA camera data path.')
        saved=read(tracked(work/'camera-world.json'));entries=read(tracked(work/'inputs.json'))
        if set(saved)!={e['id'] for e in entries}:raise ValueError('MAMMA camera identities differ.')
        for e in entries:
            c=saved[e['id']];ext=np.asarray(c['extrinsics_matrix'],float)
            if ext.shape!=(3,4):raise ValueError('Invalid MAMMA camera matrix.')
            m=np.eye(4);m[:3,:3]=ext[:,:3]@WORLD_ROTATION.T;m[:3,3]=ext[:,3]
            if np.any(np.abs(c.get('distortion_coeffs',[0]))>1e-8):note=warning='Distorted camera videos may not align with the pinhole overlay.'
            items.append(record(e['id'],c['intrinsic_matrix'],c['image_size'],m,meta['start'],meta['fps'],'Saved fixed MAMMA calibration'))
        if not np.allclose(items[0]['matrices'][0],base,atol=1e-6):raise ValueError('MAMMA camera world changed. Prepare Preview first.')
    elif model in ('flowhmr','exmr'):
        root=group or project;meta=read(tracked(root/'project.json'))
        if model=='flowhmr':
            from .flow_camera import saved_interval
            static_camera=saved_interval(project,state)==-1
            from .flow_placement import current
            if not current(project):raise ValueError('FlowHMR placement is stale. Use Prepare Preview.')
            raw=load_array(tracked(root/'flow_camera.npz'),'camera_RT_yup');r=raw[:,:3,:3];valid_rotations(r)
            rotations=r@r[0].T@base[:3,:3]
            original_centers=-np.einsum('tji,tj->ti',r,raw[:,:3,3])
            basis=base[:3,:3].T@r[0]
            displacement=(original_centers-original_centers[0])@basis.T
            note='Estimated VGGT camera path; movement scale is approximate. Motion scale 0 keeps the saved camera center. Gizmos only; video overlays keep the calibrated reference.'
        else:
            receipt=read(tracked(project/'exmr_camera_sources.json'));root=(project/receipt['root']).resolve()
            if not project.is_relative_to(root):raise ValueError('Invalid ExMR camera root.')
            p=tracked(root/'exmr_camera.npy')
            if file_stat(p)!=receipt['root_stat']:raise ValueError('ExMR camera changed. Prepare Preview first.')
            r=load_array(p);valid_rotations(r)
            rotations=r@np.asarray(receipt['reference_rotation'],float).T@base[:3,:3]
            meta=read(tracked(root/'project.json'));displacement=None
            note='ExMR camera rotation; fixed camera position.'
            policy=read(tracked(root/'exmr_camera.json'))
            static_camera=bool(policy['identity']['static'])
            if not policy['identity']['static']:
                anchor=project if group is None else (group/read(group/'people_scene.json')['anchor']).resolve()
                if group is not None and not anchor.is_relative_to(group):raise ValueError('Invalid camera reference take.')
                try:path=paired_reference(anchor,cancelled,reference_cache)
                except (OSError,ValueError,RuntimeError):path=None
                if path is not None:
                    for name,stamp in path['sources'].items():dependencies[anchor/name]=stamp
                points=path.get('centers') if path is not None else None
                if (points is not None and len(points)==len(r) and abs(path['start']-meta['start'])<1e-6 and path['fps']==meta['fps']):
                    displacement=points-points[0]
                    note='Estimated ExMR camera movement from original paired world/camera roots. Approximate, not measured motion. Gizmos only; video overlays keep the calibrated reference.'
                else:warning='Camera movement reference is unavailable or incomplete; showing rotation only.'
        m=np.tile(np.eye(4),(len(rotations),1,1));m[:,:3,:3]=rotations
        center=-base[:3,:3].T@base[:3,3];m[:,:3,3]=-np.einsum('tij,j->ti',rotations,center)
        items=[record('Camera 1',camera['K'],[state['width'],state['height']],m,meta['start'],meta['fps'],note,displacement)]
    else:
        items=[record('Camera 1',camera['K'],[state['width'],state['height']],base,state['start'],state['fps'],'Saved reference camera; no dynamic camera track exposed')]
        note='Saved reference camera only; no dynamic camera translation is available.'
    if cancelled():return None
    if (project/'scene_camera.json').exists()!=had_scene:raise ValueError('Camera inputs changed while loading.')
    if model=='flowhmr' and not current(project):raise ValueError('FlowHMR placement changed while loading. Use Prepare Preview.')
    if any(file_stat(p)!=stamp for p,stamp in dependencies.items()):raise ValueError('Camera inputs changed while loading.')
    return dict(cameras=items,model=model,note=note,warning=warning,project=str(project),static_camera=static_camera,camera_sources={str(p):stamp for p,stamp in dependencies.items()})
