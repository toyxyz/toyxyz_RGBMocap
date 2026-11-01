"""Bounded BVH file copying in the existing cancellable worker process."""
import json,os
import numpy as np
from pathlib import Path
from .motion_tools import selected_export
from .result_artifacts import file_stat


def run(request,emit=lambda *a,**k:None):
    request=Path(request)
    if request.stat().st_size>4*1024**2:raise ValueError('Export request is too large')
    rows=json.loads(request.read_text(encoding='utf-8'))
    if not isinstance(rows,list) or not 1<=len(rows)<=4096:raise ValueError('Invalid export request')
    staged=[]
    try:
        for i,row in enumerate(rows):
            project=Path(row['project']);source=selected_export(project,row['kind']);target=Path(row['target'])
            if source is None or source.resolve()!=Path(row['source']).resolve():raise ValueError('Selected target motion changed. Export again.')
            if target.suffix.lower()!='.bvh':raise ValueError('Choose a BVH destination')
            if source.resolve()==target.resolve():raise ValueError('Choose a different export destination')
            if any(target.resolve()==old[1].resolve() for old in staged):raise ValueError('Duplicate export destination')
            stamp=file_stat(source)
            scene=row.get('scene')
            if scene is not None:
                from .scene_export import context
                if context(scene['root'],project)!=scene:raise ValueError('Scene placement changed. Export again.')
            temporary=target.with_name('.'+target.name+'.rgbmocap-'+request.stem+'.partial')
            staged.append((temporary,target,project,row['kind'],source,stamp,scene))
            with source.open('rb') as inp,temporary.open('xb') as out:
                def progress():emit('progress',stage='copy_bvh',current=i,total=len(rows),message='Saving BVH in shared scene' if scene else 'Saving BVH')
                if row.get('bvh_profile','generic')=='biped':
                    from tempfile import TemporaryFile
                    from .scene_export import transform_motion
                    from .biped_bvh import convert
                    with TemporaryFile() as prepared:
                        transform_motion(inp,prepared,np.eye(4) if scene is None else scene['transform'],progress,prepend_tpose=row.get('prepend_tpose',False))
                        prepared.seek(0)
                        convert(prepared,out,progress)
                elif row.get('prepend_tpose',False) or (scene is not None and not np.allclose(scene['transform'],np.eye(4),rtol=0,atol=1e-12)):
                    from .scene_export import transform_motion
                    transform_motion(inp,out,np.eye(4) if scene is None else scene['transform'],progress,prepend_tpose=row.get('prepend_tpose',False))
                else:
                    while True:
                        chunk=inp.read(1024*1024)
                        if not chunk:break
                        out.write(chunk);progress()
            if file_stat(source)!=stamp:raise ValueError('Motion changed during export. Retry.')
        for _,_,project,kind,source,stamp,scene in staged:
            if scene is not None and context(scene['root'],project)!=scene:raise ValueError('Scene placement changed during export. Retry.')
            if selected_export(project,kind)!=source or file_stat(source)!=stamp:raise ValueError('Target result changed during export. Retry.')
        for temporary,target,*_ in staged:os.replace(temporary,target)
        return len(staged)
    finally:
        for temporary,*_ in staged:temporary.unlink(missing_ok=True)
        request.unlink(missing_ok=True)
