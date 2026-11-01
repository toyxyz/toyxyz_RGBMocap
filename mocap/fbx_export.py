"""Cancellable, transactional FBX export of the selected committed generation."""
import json
import os
from pathlib import Path
import numpy as np
from .motion_tools import selected_export
from .result_artifacts import file_stat


def run(request, emit=lambda *a, **k:None):
    from .fbx_writer import write
    from .scene_export import context
    request=Path(request)
    if request.stat().st_size>4*1024**2:raise ValueError('Export request is too large')
    rows=json.loads(request.read_text(encoding='utf-8'))
    if not isinstance(rows,list) or not 1<=len(rows)<=4096:raise ValueError('Invalid export request')
    staged=[]
    try:
        for i,row in enumerate(rows):
            project=Path(row['project']);kind=row['kind'];target=Path(row['target'])
            source=selected_export(project,kind)
            if source is None or source.resolve()!=Path(row['source']).resolve():
                raise ValueError('Selected target motion changed. Prepare Preview and export again.')
            if target.suffix.lower()!='.fbx':raise ValueError('Choose an FBX destination')
            if any(target.resolve()==old[1].resolve() for old in staged):raise ValueError('Duplicate export destination')
            meta=project/(kind+'_mixamo.json');info=json.loads(meta.read_text(encoding='utf-8'))
            paths=[meta,source]
            for key in ('asset_file','frames_file'):
                if Path(info[key]).name!=info[key]:raise ValueError('Invalid committed file path')
                paths.append(project/info[key])
            if target.resolve() in [p.resolve() for p in paths]:raise ValueError('Choose a different export destination')
            stamps=[file_stat(p) for p in paths]
            scene=row.get('scene')
            if scene is not None and context(scene['root'],project)!=scene:raise ValueError('Scene placement changed. Export again.')
            temporary=target.with_name('.'+target.name+'.rgbmocap-'+request.stem+'.partial')
            staged.append((temporary,target,project,kind,paths,stamps,scene))
            with np.load(project/info['asset_file'],allow_pickle=False) as data:
                needed=('parents','fbx_rest','fbx_rest_frames','fbx_display_frames','fbx_tip_local')
                if not all(k in data for k in needed):raise ValueError('FBX rest pose is unavailable. Use Prepare Preview first.')
                a={k:data[k] for k in needed}
            frames=np.load(project/info['frames_file'],mmap_mode='r',allow_pickle=False)
            try:
                def progress(done,total):
                    emit('progress',stage='save_fbx',current=i*52+done,total=len(rows)*52,message='Export FBX: '+kind)
                write(temporary,info['names'],a['parents'],a['fbx_rest'],a['fbx_rest_frames'],
                      a['fbx_display_frames'],a['fbx_tip_local'],frames,info['fps'],
                      transform=None if scene is None else scene['transform'],progress=progress,
                      rig_name=row.get('rig_name','SourceAnimation'),prepend_tpose=row.get('prepend_tpose',False))
            finally:frames._mmap.close()
        for _,_,project,kind,paths,stamps,scene in staged:
            if [file_stat(p) for p in paths]!=stamps or selected_export(project,kind)!=paths[1]:
                raise ValueError('Motion generation changed during export. Retry.')
            if scene is not None and context(scene['root'],project)!=scene:raise ValueError('Scene placement changed during export. Retry.')
        for temporary,target,*_ in staged:os.replace(temporary,target)
        return len(staged)
    finally:
        for temporary,*_ in staged:temporary.unlink(missing_ok=True)
        request.unlink(missing_ok=True)
