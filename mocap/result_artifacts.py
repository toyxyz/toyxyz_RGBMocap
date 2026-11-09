"""Small generation receipts; keep interrupted output out of preview/export."""
import json
from pathlib import Path
from .storage import atomic_json


def file_stat(path):
    value=Path(path).stat()
    return [value.st_size,value.st_mtime_ns]


def pending_path(project,kind,stage):
    return Path(project)/f'{kind}_{stage}_pending.json'


def begin_update(project,kind,stage):
    # Deliberately retained after failure/cancellation. A successful retry is
    # the only operation that may clear an unfinished generation.
    atomic_json(pending_path(project,kind,stage),{'stage':stage})


def finish_update(project,kind,stage):
    pending_path(project,kind,stage).unlink(missing_ok=True)


def commit_artifacts(project,kind,source_stat,fps):
    from .motion_tools import RESULTS
    project=Path(project)
    if file_stat(project/RESULTS[kind][0])!=source_stat:
        raise ValueError('Motion changed during export. Run export again.')
    state_path=project/'project.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    if state.get('fps',30)!=fps:raise ValueError('Frame rate changed during export. Run export again.')
    mesh=json.loads((project/f'{kind}_mesh.json').read_text(encoding='utf-8'))
    if [mesh['source_size'],mesh['source_mtime_ns']]!=source_stat:
        raise ValueError('Mesh and exported motion belong to different results.')
    frame_file=mesh['frames_file']
    if Path(frame_file).name!=frame_file:raise ValueError('Invalid mesh frame path')
    outputs=[f'{kind}_joints.npz',f'{kind}_motion.bvh',f'{kind}_mesh.json',frame_file]
    atomic_json(project/f'{kind}_artifacts.json',{'version':1,'source_stat':source_stat,'fps':fps,
        'model_stat':[mesh['model_size'],mesh['model_mtime_ns']],
        'outputs':{name:file_stat(project/name) for name in outputs}})
    finish_update(project,kind,'export')


def artifacts_current(project,kind):
    """Only stat small metadata here; never read motion tensors in the GUI."""
    from .motion_tools import RESULTS
    project=Path(project)
    from .gemx_result import is_project
    if (project/'soma-result.json').exists() and is_project(project):
        from .motion_tools import result_is_current
        return result_is_current(project, kind, ignore_contacts=True)
    if pending_path(project,kind,'export').exists() or pending_path(project,kind,'correct').exists():return False
    try:
        receipt=project/f'{kind}_artifacts.json'
        source=project/RESULTS[kind][0]
        if receipt.exists():
            record=json.loads(receipt.read_text(encoding='utf-8'))
            if record.get('version')!=1 or file_stat(source)!=record['source_stat']:return False
            state_path=project/'project.json'
            state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
            if state.get('fps',30)!=record['fps']:return False
            outputs=record['outputs']
            if not all(name in outputs for name in (f'{kind}_joints.npz',f'{kind}_motion.bvh',f'{kind}_mesh.json')):return False
            mesh=json.loads((project/f'{kind}_mesh.json').read_text(encoding='utf-8'))
            if mesh['frames_file'] not in outputs:return False
            if not all(Path(name).name==name and file_stat(project/name)==stamp for name,stamp in outputs.items()):return False
            from .models import BY_KEY
            if state.get('pipeline',{}).get('model')=='mamma':
                from .mamma_body import body_path
                model=body_path(project)
            else:model=BY_KEY['smplh' if mesh.get('representation')=='smplh' else 'smplx'].path
            return not model.is_file() or file_stat(model)==record['model_stat']
        # Existing baked projects remain usable. Their mesh metadata already
        # ties the generation to its source tensor. Version 3 corrections must
        # always finish a new export before they can be displayed or exported.
        if kind.startswith('corrected_'):
            report=project/f'{kind}_correction.json'
            if report.exists() and json.loads(report.read_text(encoding='utf-8')).get('version',1)>=3:return False
        mesh=project/f'{kind}_mesh.json'
        if source.exists() and mesh.exists():
            info=json.loads(mesh.read_text(encoding='utf-8'))
            if file_stat(source)!=[info['source_size'],info['source_mtime_ns']]:return False
            from .models import BY_KEY
            from .mamma_body import body_path
            state_path=project/'project.json'
            mamma=state_path.exists() and json.loads(state_path.read_text(encoding='utf-8')).get('pipeline',{}).get('model')=='mamma'
            model=body_path(project) if mamma else BY_KEY['smplh' if info.get('representation')=='smplh' else 'smplx'].path
            return not model.is_file() or file_stat(model)==[info['model_size'],info['model_mtime_ns']]
        return True  # Legacy joint-only preview projects without motion tensors.
    except (OSError,ValueError,KeyError,TypeError):return False


def stage_results_current(stage,project,selected):
    """A cache hit must be recoverable when a payload/receipt was lost."""
    from .motion_tools import result_sources,result_is_current,resolve_motion_result
    kinds=result_sources(resolve_motion_result(project,selected))
    if stage=='correct':
        return all(result_is_current(project,kind) for kind in kinds if kind.startswith('corrected_'))
    if stage=='export':
        from .mixamo_data import current
        return all(result_is_current(project,kind) and artifacts_current(project,kind) and current(project,kind) for kind in kinds)
    return True
