"""Bounded BVH file copying in the existing cancellable worker process."""
import json,os
import numpy as np
from pathlib import Path
from .motion_tools import selected_export
from .gemx_result import is_project
from .result_artifacts import file_stat


def _reference_clearance(asset):
    """Reuse the sole-to-toe distance from the committed FBX reference."""
    with np.load(asset, allow_pickle=False) as data:
        rest = data.get('fbx_rest')
        if rest is None or rest.shape != (52, 3) or not np.isfinite(rest).all():
            raise ValueError('Invalid BVH reference geometry. Use Prepare Preview.')
    return float(rest[[10, 11], 1].min())


def run(request,emit=lambda *a,**k:None):
    from .export_options import align_auto_floor, centered_transform, first_bvh_root, recorded_auto_floor
    from .first_person.rig import is_project as is_first_person_take
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
            if is_first_person_take(project):raise ValueError('Export the ViDiHand2 first-person skeleton as FBX.')
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
            kind = row['kind']
            metadata = project / f'{kind}_mixamo.json'
            metadata_stamp = file_stat(metadata)
            info = json.loads(metadata.read_text(encoding='utf-8'))
            asset_name = info['asset_file']
            if Path(asset_name).name != asset_name:
                raise ValueError('Invalid committed file path')
            asset = project / asset_name
            floor_receipt = project / f'{kind}_correction.json'
            paths = [source, metadata, asset]
            if kind.startswith('corrected_') and floor_receipt.is_file():
                paths.append(floor_receipt)
            if destinations.intersection(path.resolve() for path in paths):
                raise ValueError('An export destination overlaps a committed source file.')
            stamps = [file_stat(path) for path in paths]
            if stamps[1] != metadata_stamp:
                raise ValueError('Motion changed during export. Retry.')
            clearance = _reference_clearance(asset)
            floor = recorded_auto_floor(project, kind) if floor_receipt in paths else None
            scene=row.get('scene')
            if scene is not None:
                from .scene_export import context
                if context(scene['root'],project)!=scene:raise ValueError('Scene placement changed. Export again.')
            staged.append((temporary,target))
            committed.append((project,kind,source,paths,stamps,scene))
            with source.open('rb') as inp,temporary.open('xb') as out:
                def progress():emit('progress',stage='copy_bvh',current=i,total=len(rows),message='Saving BVH in shared scene' if scene else 'Saving BVH')
                transform = align_auto_floor(None if scene is None else scene['transform'], floor)
                transform = np.eye(4) if transform is None else np.asarray(transform, dtype=float)
                if row.get('place_at_origin',False):
                    transform=centered_transform(transform,first_bvh_root(inp),origins,row.get('origin_group',str(project)))
                symmetric=row.get('symmetric_rest',False)
                prepare = (clearance != 0. or row.get('prepend_tpose', False) or symmetric
                           or not np.allclose(transform, np.eye(4), rtol=0, atol=1e-12))
                if profile=='biped':
                    from .biped_bvh import convert
                    if prepare:
                        from tempfile import TemporaryFile
                        from .scene_export import transform_motion
                        with TemporaryFile() as prepared:
                            transform_motion(inp,prepared,transform,progress,prepend_tpose=row.get('prepend_tpose',False),symmetric_rest=symmetric,sole_clearance=clearance)
                            prepared.seek(0);convert(prepared,out,progress)
                    else:convert(inp,out,progress)
                elif prepare:
                    from .scene_export import transform_motion
                    transform_motion(inp,out,transform,progress,prepend_tpose=row.get('prepend_tpose',False),symmetric_rest=symmetric,sole_clearance=clearance)
                else:
                    while True:
                        chunk=inp.read(1024*1024)
                        if not chunk:break
                        out.write(chunk);progress()
            if [file_stat(path) for path in paths] != stamps:
                raise ValueError('Motion changed during export. Retry.')
        for verify in native_checks:verify()
        for project,kind,source,paths,stamps,scene in committed:
            if scene is not None and context(scene['root'],project)!=scene:raise ValueError('Scene placement changed during export. Retry.')
            if selected_export(project,kind)!=source or [file_stat(path) for path in paths]!=stamps:raise ValueError('Target result changed during export. Retry.')
        from .workflow_jobs import publish_output
        for (temporary, target), row in zip(staged, rows):
            publish_output(temporary, target, replace=row.get('replace_existing', True))
        return len(staged)
    finally:
        for temporary,*_ in staged:temporary.unlink(missing_ok=True)
        request.unlink(missing_ok=True)
