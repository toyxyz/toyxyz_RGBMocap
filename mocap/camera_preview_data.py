"""Prepare a small camera-space overlay in a worker, never in the GUI."""
import json
from pathlib import Path
import numpy as np
from .blend_export import camera_record
from .blend_camera_track import load
from .camera_alignment import align
from .motion_tools import selected_export
from .result_artifacts import file_stat

def run(project,kind,out,background_refine=False):
    project=Path(project);out=Path(out)
    source=selected_export(project,kind)
    if source is None:raise ValueError('Apply Correction or Prepare Preview for the selected take first.')
    meta=project/(kind+'_mixamo.json');info=json.loads(meta.read_text())
    state=json.loads((project/'project.json').read_text());model=None
    paths=[meta,source,project/'project.json']
    for key in ('frames_file','asset_file'):
        if Path(info[key]).name!=info[key]:raise ValueError('Invalid preview path')
        paths.append(project/info[key])
    stamps=[file_stat(p) for p in paths]
    camera,extra=camera_record(project);model=camera.get('source_model');dependencies=[(p,file_stat(p)) for p in extra]
    if model=='flowhmr':
        reference,extra,_=align(project,camera['world_to_camera'],kind);dependencies+=extra
    else:reference=np.asarray(camera['world_to_camera'])
    track,extra,description=load(project,reference,info['frames']);dependencies+=extra
    if track is None:track=np.broadcast_to(reference,(info['frames'],4,4))
    delta=np.zeros((info['frames'],3))
    if model=='flowhmr':
        moving,extra,_=load(project,reference,info['frames'],translation_scale=1);dependencies+=extra
        delta=moving[:,:3,3]-track[:,:3,3]
    background_report=None
    if background_refine:
        from .camera_background import refine
        refined,extra,background_report=refine(project,track,camera['K']);dependencies+=extra
        correction=refined[:,:3,:3]@track[:,:3,:3].swapaxes(-1,-2)
        delta=np.einsum('tij,tj->ti',correction,delta);track=refined
    frames=np.load(paths[-2],mmap_mode='r',allow_pickle=False)
    points=np.lib.format.open_memmap(out/'points.npy',mode='w+',dtype='float32',shape=(len(frames),52,3))
    for start in range(0,len(frames),128):
        end=min(start+128,len(frames));points[start:end]=np.einsum('tij,tkj->tki',track[start:end,:3,:3],frames[start:end,:,:3,3])+track[start:end,None,:3,3]
    points.flush();del points,frames
    with np.load(paths[-1],allow_pickle=False) as asset:parents=asset['parents'].tolist()
    np.save(out/'translation.npy',delta)
    if [file_stat(p) for p in paths]!=stamps or any(file_stat(p)!=s for p,s in dependencies) or selected_export(project,kind)!=source:
        raise ValueError('Motion or camera changed. Refresh the camera preview.')
    from .video import select_preview
    data=dict(K=camera['K'],parents=parents,frames=info['frames'],fps=info['fps'],start=state.get('start',0),
              width=state['width'],height=state['height'],video=str(select_preview(project)),model=model,description=description,background_refinement=background_report)
    (out/'ready.json').write_text(json.dumps(data),encoding='utf-8')
