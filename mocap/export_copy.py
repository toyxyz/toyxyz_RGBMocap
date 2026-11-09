"""Bounded BVH file copying in the existing cancellable worker process."""
import json,os
import numpy as np
from pathlib import Path
from .motion_tools import selected_export
from .gemx_result import is_project
from .result_artifacts import file_stat


def run(request,emit=lambda *a,**k:None):
    request=Path(request)
    if request.stat().st_size>4*1024**2:raise ValueError('Export request is too large')
    rows=json.loads(request.read_text(encoding='utf-8'))
    if not isinstance(rows,list) or not 1<=len(rows)<=4096:raise ValueError('Invalid export request')
    staged=[];origins={};committed=[];native_checks=[]
    try:
        destinations = {Path(row['target']).resolve() for row in rows}
        if len(destinations) != len(rows):raise ValueError('Duplicate export destination')
        for i,row in enumerate(rows):
            profile=row.get('bvh_profile','generic')
            if profile not in ('generic','biped'):raise ValueError('Unsupported BVH export profile')
            project=Path(row['project']);source=selected_export(project,row['kind']);target=Path(row['target'])
            if target.suffix.lower()!='.bvh':raise ValueError('Choose a BVH destination')
            if source is None or source.resolve()!=Path(row['source']).resolve():raise ValueError('Selected target motion changed. Export again.')
            if source.resolve() in destinations:raise ValueError('An export destination overlaps a committed source file.')
            temporary=target.with_name('.'+target.name+'.rgbmocap-'+request.stem+'.partial')
            if is_project(project):
                from .gemx_skeleton_export import write as write_soma

                def progress(done,total):
                    emit('progress',stage='copy_bvh',current=i*total+done,total=len(rows)*total,message='Export SOMA BVH')

                staged.append((temporary,target))
                native_checks.append(write_soma(temporary,row,source,destinations,'bvh',progress))
                continue
            stamp=file_stat(source)
            scene=row.get('scene')
            if scene is not None:
                from .scene_export import context
                if context(scene['root'],project)!=scene:raise ValueError('Scene placement changed. Export again.')
            staged.append((temporary,target))
            committed.append((project,row['kind'],source,stamp,scene))
            with source.open('rb') as inp,temporary.open('xb') as out:
                def progress():emit('progress',stage='copy_bvh',current=i,total=len(rows),message='Saving BVH in shared scene' if scene else 'Saving BVH')
                transform=np.eye(4) if scene is None else np.asarray(scene['transform'],float)
                if row.get('place_at_origin',False):
                    from .export_options import centered_transform,first_bvh_root
                    transform=centered_transform(transform,first_bvh_root(inp),origins,row.get('origin_group',str(project)))
                symmetric=row.get('symmetric_rest',False)
                prepare=row.get('prepend_tpose',False) or symmetric or not np.allclose(transform,np.eye(4),rtol=0,atol=1e-12)
                if profile=='biped':
                    from .biped_bvh import convert
                    if prepare:
                        from tempfile import TemporaryFile
                        from .scene_export import transform_motion
                        with TemporaryFile() as prepared:
                            transform_motion(inp,prepared,transform,progress,prepend_tpose=row.get('prepend_tpose',False),symmetric_rest=symmetric)
                            prepared.seek(0);convert(prepared,out,progress)
                    else:convert(inp,out,progress)
                elif prepare:
                    from .scene_export import transform_motion
                    transform_motion(inp,out,transform,progress,prepend_tpose=row.get('prepend_tpose',False),symmetric_rest=symmetric)
                else:
                    while True:
                        chunk=inp.read(1024*1024)
                        if not chunk:break
                        out.write(chunk);progress()
            if file_stat(source)!=stamp:raise ValueError('Motion changed during export. Retry.')
        for verify in native_checks:verify()
        for project,kind,source,stamp,scene in committed:
            if scene is not None and context(scene['root'],project)!=scene:raise ValueError('Scene placement changed during export. Retry.')
            if selected_export(project,kind)!=source or file_stat(source)!=stamp:raise ValueError('Target result changed during export. Retry.')
        for temporary,target,*_ in staged:os.replace(temporary,target)
        return len(staged)
    finally:
        for temporary,*_ in staged:temporary.unlink(missing_ok=True)
        request.unlink(missing_ok=True)
