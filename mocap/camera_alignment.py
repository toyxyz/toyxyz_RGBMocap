"""Constant per-take camera alignment from existing SAM pelvis anchors."""
import json
import numpy as np
from .result_artifacts import file_stat

def align(project,reference,kind):
    paths=[project/'flow_placement.json',project/(kind+'_mixamo.json'),project/'flow_camera.npz']
    stamps=[file_stat(p) for p in paths]
    anchors=np.asarray(json.loads(paths[0].read_text())['camera_pelvis'],float)
    info=json.loads(paths[1].read_text());name=info['frames_file']
    from pathlib import Path
    if Path(name).name!=name:raise ValueError('Invalid motion path')
    paths.append(project/name);stamps.append(file_stat(paths[-1]))
    frames=np.load(paths[-1],mmap_mode='r',allow_pickle=False)
    if anchors.ndim!=2 or anchors.shape[1]!=4 or not len(anchors) or not np.isfinite(anchors).all():raise ValueError('Invalid camera anchors')
    ids=anchors[:,0].astype(int)
    if not np.array_equal(ids,anchors[:,0]) or np.any(ids<0) or np.any(ids>=min(30,len(frames))):raise ValueError('Invalid camera anchor frames')
    points=np.array(frames[ids,0,:3,3],float);del frames
    # Track loader already transports rotations into the shared world.
    from .blend_camera_track import load
    track,extra,_=load(project,reference,info['frames'])
    r=track[ids,:3,:3];t=track[ids,:3,3]
    residual=np.einsum('tij,tj->ti',r,points)+t-anchors[:,1:]
    shift=np.median(np.einsum('tji,tj->ti',r,residual),axis=0)
    result=np.asarray(reference,float).copy();result[:3,3]-=result[:3,:3]@shift
    after=residual-np.einsum('tij,j->ti',r,shift)
    if any(file_stat(p)!=s for p,s in zip(paths,stamps)):raise ValueError('Alignment inputs changed')
    return result,list(zip(paths,stamps))+extra,dict(method='Constant camera-center fit to saved SAM pelvis anchors for this exported take',
        anchor_error_before_m=float(np.linalg.norm(residual,axis=1).mean()),anchor_error_after_m=float(np.linalg.norm(after,axis=1).mean()),center_offset_m=shift.tolist())
