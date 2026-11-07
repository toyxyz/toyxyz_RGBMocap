"""Sequential actor work within one stage-scoped resident model session."""
from pathlib import Path
import argparse
import gc
import json
import os
import sys
import time
import traceback


def main(argv, emit):
    parser = argparse.ArgumentParser()
    parser.add_argument('stage')
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--project')  # Explicit routing context, validated below.
    parser.add_argument('--result')
    args = parser.parse_args(argv)
    from .queue_groups import validate,MAX_BYTES,PHASES
    if args.request.stat().st_size>MAX_BYTES:raise ValueError('Batch request is too large')
    phase,jobs = validate(json.loads(args.request.read_text(encoding='utf-8')),args.stage)
    from .compute_policy import configure_worker_cpu
    threads=configure_worker_cpu(PHASES[args.stage][phase])
    from .model_session import ModelSession,set_memory_budget
    from .storage import atomic_json
    import torch
    torch.set_num_threads(threads)
    os.environ['RGB_MOCAP_STAGE']=PHASES[args.stage][phase]
    started=time.monotonic();all_cached=True;active=None
    try:
        phases=({'gvhmr_batch':('pose','features','gvhmr'),
                 'flow_batch':('flow_features','flowhmr'),
                 'exmr_batch':('exmr_preprocess','body','hand','exmr')}[args.stage]
                if phase=='all' else (phase,))
        for phase in phases:
            os.environ['RGB_MOCAP_STAGE']=PHASES[args.stage][phase]
            with ModelSession().activate() as session:
                for index,job in enumerate(jobs,1):
                    active=job['project'];stage=PHASES[args.stage][phase]
                    emit('batch_item_start',stage=stage,phase=phase,index=index,count=len(jobs),project=str(active))
                    cached=False
                    def observe(kind,**value):
                        nonlocal cached
                        if kind=='complete':
                            cached=bool(value.get('cached',False));value['batch_inner']=True
                        if kind=='progress':value['phase']=f'{phase}:{index}:'+str(value.get('phase',value.get('message','')))
                        emit(kind,**value)
                    options=dict(force=job['force'])
                    # Reset per-actor randomness, not the loaded model or its weights.
                    torch.manual_seed(0)
                    if args.stage=='exmr_batch':
                        from . import exmr_backend as backend
                        backend.run_stage(active,stage,job['result'],observe,
                            feature_phase=phase if phase in ('body','hand') else None,**options)
                    elif args.stage=='flow_batch':
                        from . import flow_backend as backend
                        backend.run_stage(active,stage,job['result'],observe,**options)
                    elif phase in ('pose','features'):
                        import psutil
                        from .memory_mode import select_stage
                        if not torch.cuda.is_available():raise RuntimeError('GVHMR requires a supported NVIDIA CUDA GPU')
                        free,total=torch.cuda.mem_get_info();set_memory_budget(free,total)
                        select_stage('features',free+session.resident_bytes,total,0,observe)
                        from . import engines
                        # Per-file receipts permit interrupted observation phases to resume.
                        output=engines.features(active,observe,phase=phase)
                    else:
                        from . import worker
                        saved_argv=sys.argv;saved_emit=worker.emit
                        sys.argv=['mocap.worker','gvhmr','--project',str(active),'--result',job['result']]+(['--force'] if job['force'] else [])
                        worker.emit=observe
                        try:
                            code=worker.main()
                            if code:return code
                        finally:sys.argv=saved_argv;worker.emit=saved_emit
                    all_cached &= cached
                    gc.collect()
                    if torch.cuda.is_initialized():torch.cuda.empty_cache()
                    import psutil
                    from .process_memory import open_handles
                    emit('batch_item_done',stage=stage,project=str(active),index=index,count=len(jobs),
                         resident_models=int(session.value is not None),resident_allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0,
                         peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
                         rss_bytes=psutil.Process().memory_info().rss,handles=open_handles(psutil.Process()))
            emit('batch_phase_released',phase=phase,allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0)
        emit('complete',stage=args.stage,output=str(jobs[-1]['project']),cached=all_cached,elapsed=time.monotonic()-started)
        return 0
    except Exception as error:
        details=traceback.format_exc();print(details,file=sys.stderr,flush=True)
        if active is not None:
            state=json.loads((active/'project.json').read_text(encoding='utf-8'))
            state.setdefault('stages',{})[PHASES[args.stage][phase]]=dict(status='failed',error=str(error))
            atomic_json(active/'project.json',state)
            (active/(args.stage+'_error.log')).write_text(details,encoding='utf-8')
        message=str(error)
        if isinstance(error,MemoryError) or 'out of memory' in message.lower():
            from .memory_mode import GUIDANCE
            message+=' '+GUIDANCE
        emit('failed',message=message)
        return 1
