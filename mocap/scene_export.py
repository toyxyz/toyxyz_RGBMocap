"""Shared viewport placement at export time; original bind and take caches stay intact."""
import json
from pathlib import Path
import numpy as np
from .scene_camera import VERSION,read_camera,validate_transform
from .people_store import root_for,load_people,safe_child,media_receipt
from .result_artifacts import file_stat


def context(root,project):
    root,project=Path(root).resolve(),Path(project).resolve()
    state=load_people(root)
    if root_for(project)!=root or not state or not any(
            safe_child(root,t['project'])==project for p in state['people'] for t in p.get('takes',[])):
        raise ValueError('The export take does not belong to this people scene')
    path=root/'people_scene.json';reference=json.loads(path.read_text(encoding='utf-8'))
    anchor_project=safe_child(root,reference['anchor']);anchor=read_camera(anchor_project);camera=read_camera(project)
    if (reference.get('version')!=VERSION or reference.get('input')!=media_receipt(root)
            or anchor is None or camera is None or reference.get('sources')!=anchor['sources']):
        raise ValueError('Shared scene cameras are stale. Use Prepare Preview')
    reference_matrix=validate_transform(reference['world_to_camera'])
    if not np.allclose(reference_matrix,anchor['world_to_camera'],rtol=0,atol=1e-8):
        raise ValueError('Shared scene anchor changed. Use Prepare Preview')
    if not np.allclose(reference['K'],camera['K'],rtol=0,atol=1e-4):
        raise ValueError('Scene camera intrinsics differ. Use Prepare Preview')
    transform=validate_transform(np.linalg.inv(reference_matrix)@validate_transform(camera['world_to_camera']))
    if camera.get('source_model')=='flowhmr':
        if camera.get('placement_version')!=2 or not np.allclose(transform,np.eye(4),rtol=0,atol=1e-7):
            raise ValueError('FlowHMR placement needs updating. Use Prepare Preview')
        transform=np.eye(4) # Native Flow tensors already inhabit the common world.
    return dict(root=str(root),transform=transform.tolist(),reference=file_stat(path),
                camera=file_stat(project/'scene_camera.json'),anchor=file_stat(anchor_project/'scene_camera.json'),
                sources=camera['sources'],anchor_sources=anchor['sources'],model=camera.get('model'))


def transform_motion(inp,out,transform,progress=lambda:None,prepend_tpose=False):
    """Stream only root channels in bounded blocks; keep the hierarchy byte-for-byte."""
    from scipy.spatial.transform import Rotation
    import warnings
    transform=validate_transform(transform);rotation=transform[:3,:3];translation=transform[:3,3]
    channels=0;root_seen=False;root_order=None
    for _ in range(4096):
        line=inp.readline(65537)
        if not line or len(line)>65536:raise ValueError('Invalid BVH hierarchy')
        out.write(line);tokens=line.split()
        if tokens[:1]==[b'ROOT']:root_seen=True
        if tokens[:1]==[b'CHANNELS']:
            if channels==0:
                if tokens[:5]!=[b'CHANNELS',b'6',b'Xposition',b'Yposition',b'Zposition'] or sorted(tokens[5:])!=[b'Xrotation',b'Yrotation',b'Zrotation']:
                    raise ValueError('Unsupported BVH root channels')
                root_order=''.join(token[:1].decode('ascii') for token in tokens[5:])
            channels+=int(tokens[1])
        if line.strip()==b'MOTION':break
    else:raise ValueError('BVH hierarchy is too large')
    if not root_seen or channels!=159:raise ValueError('Expected the committed 52-bone BVH')
    count_line=inp.readline(256);time_line=inp.readline(256)
    if not count_line.startswith(b'Frames:') or not time_line.startswith(b'Frame Time:'):
        raise ValueError('Invalid BVH motion header')
    count=int(count_line.split(b':',1)[1]);step=float(time_line.split(b':',1)[1])
    if count<1 or not np.isfinite(step) or step<=0:raise ValueError('Invalid BVH timing')
    out.write(f'Frames: {count+1}\n'.encode('ascii') if prepend_tpose else count_line)
    out.write(time_line);previous=np.zeros(3)
    for first in range(0,count,128):
        rows=[]
        for _ in range(min(128,count-first)):
            line=inp.readline(65537)
            if len(line)>65536:raise ValueError('BVH frame is too large')
            fields=line.split()
            if len(fields)!=channels:raise ValueError('Invalid BVH motion channels')
            values=np.asarray(fields,dtype=float)
            if not np.isfinite(values).all():raise ValueError('Non-finite BVH motion')
            rows.append((fields,values[:6]))
        values=np.asarray([v for _,v in rows]);positions=values[:,:3]@rotation.T+translation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Gimbal lock detected')
            angles=Rotation.from_matrix(rotation@Rotation.from_euler(root_order,values[:,3:6],degrees=True).as_matrix()).as_euler(root_order)
        if first==0 and prepend_tpose:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Gimbal lock detected')
                reference=Rotation.from_matrix(rotation).as_euler(root_order)
            pose=np.zeros(channels);pose[:3]=positions[0];pose[3:6]=np.rad2deg(reference)
            out.write((' '.join(f'{v:.9f}' for v in pose)+'\n').encode('ascii'))
            previous=reference
        for index,(fields,_) in enumerate(rows):
            aa=angles[index].copy();bb=aa.copy();bb[0]+=np.pi;bb[1]=np.pi-bb[1];bb[2]+=np.pi
            aa+=2*np.pi*np.round((previous-aa)/(2*np.pi));bb+=2*np.pi*np.round((previous-bb)/(2*np.pi))
            previous=bb if np.sum((bb-previous)**2)<np.sum((aa-previous)**2) else aa
            root=np.r_[positions[index],np.rad2deg(previous)]
            out.write((' '.join(f'{v:.9f}' for v in root)).encode('ascii')+b' '+b' '.join(fields[6:])+b'\n')
        progress()
    if inp.read(1):raise ValueError('Unexpected trailing BVH data')
