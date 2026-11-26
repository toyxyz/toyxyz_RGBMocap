"""Presentation-only worker entry; inference/cache identities stay unchanged."""
import json,runpy,sys
from contextlib import contextmanager,nullcontext

for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):
        stream.reconfigure(encoding='utf-8',errors='backslashreplace')


def emit(kind,**value):
    print('@@MOCAP@@'+json.dumps(dict(type=kind,**value)),flush=True)


@contextmanager
def download_progress(callback):
    """Preserve the downloader signature, including recursive metadata calls."""
    from . import models
    original=models.download
    def wrapped(key,progress=None):
        return original(key,progress=callback if progress is None else progress)
    try:
        models.download=wrapped
        yield
    finally:models.download=original


def main():
    stage=sys.argv[1] if len(sys.argv)>1 else ''
    if stage in {'people','person','track','pva','features','gvhmr','refine','footmr','handflow','rig','mesh','export','correct'} and '--project' in sys.argv:
        from .gemx_result import is_project
        if is_project(sys.argv[sys.argv.index('--project')+1]):
            emit('failed',stage=stage,message='This operation requires a SMPL result. GEM-X uses native SOMA.')
            return 1
    if stage in ('save_blend','save_blend_scene'):
        from .blend_export_worker import main as save_main
        return save_main(emit)
    if stage=='stream_reference':
        from .stream_reference import main as reference_main
        return reference_main(emit)
    if stage=='camera_background':
        from .camera_background_worker import main as camera_main
        return camera_main()
    if stage=='camera_reference':
        from .camera_reference import main as camera_main
        return camera_main()
    if stage in ('copy_bvh','save_fbx'):
        from .source_export_worker import main as save_main
        return save_main(emit)
    if stage=='flow_mamma_gravity':
        from .mamma_gravity_worker import main as gravity_main
        return gravity_main()
    if stage in ('prompthmr_prepare','prompthmr_publish'):
        from .compute_policy import configure_worker_cpu
        configure_worker_cpu(stage)
        from .prompthmr_pipeline import main as phmr_main
        return phmr_main(sys.argv[1:])
    if stage in ('flow_batch','exmr_batch','gvhmr_batch'):
        from .batch_worker import main as batch_main
        from .progress_events import tqdm_events
        with tqdm_events(emit):return batch_main(sys.argv[1:],emit)
    if stage in ('mamma_prepare','mamma_publish'):
        from .compute_policy import configure_worker_cpu
        configure_worker_cpu(stage)
        from .mamma_pipeline import main as mamma_main
        return mamma_main(sys.argv[1:])
    if stage == 'gemx_contact_solve':
        from .gemx_contact_solver import main as solve
        return solve(sys.argv[2:], emit)
    if stage=='character':
        from .retarget.worker import main as character_main
        return character_main(sys.argv[2:])
    if stage=='vidihand_publish':
        from .first_person.publish import main as publish_first_person
        return publish_first_person(sys.argv[2:],emit)
    if stage=='vidihand_arms':
        from .first_person.arms_worker import main as estimate_first_person_arms
        return estimate_first_person_arms(sys.argv[2:],emit)
    if stage=='flow_vidihand_camera':
        from pathlib import Path
        import time
        from .flow_backend import activate, camera
        from .compute_policy import configure_worker_cpu
        from .model_session import set_memory_budget
        from .motion_tools import require_supported_project
        configure_worker_cpu(stage)
        started=time.monotonic()
        try:
            import torch
            project=Path(sys.argv[sys.argv.index('--project')+1])
            if require_supported_project(project).key!='vidihand2':
                raise ValueError('Expected a ViDiHand2 project for camera estimation.')
            if not torch.cuda.is_available():
                raise RuntimeError('Moving camera estimation needs CUDA.')
            set_memory_budget(*torch.cuda.mem_get_info())
            activate()
            output=camera(project,emit)
        except (OSError,ValueError,RuntimeError,ImportError) as error:
            emit('failed',stage=stage,message=str(error))
            return 1
        emit('complete',stage=stage,output=str(output),elapsed=time.monotonic()-started)
        return 0
    if stage=='save_project':
        from .project_save import main as save_main
        sys.argv.pop(1)
        return save_main(emit)
    if stage.startswith('exmr') or stage=='download':
        from .progress_events import tqdm_events
        with tqdm_events(emit), (download_progress(emit) if stage=='download' else nullcontext()):
            runpy.run_module('mocap.worker',run_name='__main__')
    else:runpy.run_module('mocap.worker',run_name='__main__')


if __name__=='__main__':raise SystemExit(main())
