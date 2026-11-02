"""Worker resource budgets. GUI imports stay free of Torch and CUDA."""
from __future__ import annotations
import os
GIB=1024**3
MODES=('auto','low_memory')


def ram_reserve(total):
    return max(GIB,min(4*GIB,int(total*.08)))


def gpu_reserve(total):
    """ComfyUI's external-use margin, separate from inference working memory."""
    mib=1024**2
    return (600+(100 if total>15*GIB else 0))*mib if os.name=='nt' else 400*mib


def gpu_memory_fraction(free,total):
    """Budget the CUDA allocator from current free VRAM, without a second cap.

    Model weights, activations and temporary inference buffers share this budget.
    It does not cap non-PyTorch allocations or reserve memory against other apps.
    """
    from .resource_safety import enabled
    if not enabled():return 1.
    return max(0.,min(1.,(free-gpu_reserve(total))/total))


def worker_threads(stage, logical=None):
    logical=max(1,logical or os.cpu_count() or 1)
    cap=4 if stage in ('refine','handflow') else 16 if stage=='footmr' else 8
    default=2 if stage=='correct' else min(cap,max(1,logical//2))
    try:requested=int(os.environ.get('RGB_MOCAP_WORKER_THREADS',default))
    except ValueError:requested=default
    return max(1,min(requested,max(1,logical-2)))


def configure_worker_cpu(stage):
    # Do not inherit arbitrary desktop BLAS pools; large pools slow preprocessing.
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    return worker_threads(stage)


def prefetch_plan(available_ram,total_ram,logical=None,mode=None):
    """Overlap CPU crops only when the desktop has spare CPU and RAM."""
    logical=logical or os.cpu_count() or 1
    from .memory_mode import effective_mode
    mode=mode or effective_mode()
    return 2 if (mode!='low_memory' and logical>=8 and
                 available_ram-ram_reserve(total_ram)>=6*GIB) else 0


def gvhmr_plan(frames,free_vram,total_vram):
    """Use dense attention whenever it fits. Split queries, never keys/time context.

    Conservative estimates include model weights and whole-sequence intermediates.
    The validated query block is fixed at 2048; smaller unvalidated blocks are
    deliberately not used as an automatic escape hatch.
    """
    frames=max(1,int(frames))
    budget=max(0,free_vram-gpu_reserve(total_vram))
    fixed=.5*GIB+frames*196608
    dense=max(3*GIB,fixed+frames*frames*72)
    split=max(3*GIB,fixed+frames*min(frames,2048)*72)
    use_split=frames>2048 and dense>budget
    return dict(attention='query_split' if use_split else 'dense',query_block=2048,
                estimated_bytes=int(split if use_split else dense),
                dense_estimated_bytes=int(dense),fits=(split if use_split else dense)<=budget,
                precision='float32',full_key_context=True)


def image_plan(stage,free_vram,available_ram,mode=None):
    from .memory_mode import effective_mode
    mode=mode or effective_mode()
    if mode not in MODES:mode='auto'
    import sys
    torch=sys.modules.get('torch')
    initialized=bool(torch and torch.cuda.is_initialized())
    total=torch.cuda.get_device_properties(0).total_memory if initialized else free_vram
    budget=max(0,free_vram-gpu_reserve(total))
    if stage=='pva':
        # Called AFTER loading the 3.5GiB model. Measured total allocation peaks
        # at 5.08GiB even with batch 120; the temporal network sets that floor.
        batch=120 if budget>=2*GIB else 60 if budget>=1.7*GIB else 24 if budget>=1.5*GIB else 8
        if mode=='low_memory':batch=min(batch,24)
    else:
        # Real 5090 sweeps: WholeBody batch 8 is faster than 16/32, not just smaller.
        measured_gpu=initialized and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 5090'
        preferred=8 if stage=='footmr' and measured_gpu else 16
        batch=preferred if budget>=3.5*GIB else 8 if budget>=3*GIB else 4 if budget>=2.8*GIB else 1
        if mode=='low_memory':batch=min(batch,4)
    batch=min(batch,max(1,int(max(0,available_ram-GIB)/(64*1024**2))))
    return {'image_batch':batch,'core_frames':256 if mode=='low_memory' or budget<3*GIB else 1024,
            'halo_frames':720,'precision':'float32','flip_test':True,'mode':mode}


def memory_pressure_message(available,total,commit=None):
    """Report measured system pressure, never a predicted stage allocation."""
    reserve=ram_reserve(total)
    if available>=reserve and (commit is None or commit>=GIB):return None
    detail=f'Available RAM {available/GIB:.2f} GiB (system reserve {reserve/GIB:.2f} GiB)'
    if commit is not None:detail+=f'; Windows allocation headroom {commit/GIB:.2f} GiB'
    from .memory_mode import GUIDANCE
    return ('Processing stopped because available system memory is low. '+detail+
            '. '+GUIDANCE+' Completed stages can be reused.')


def retry_cuda(operation,batch,on_retry,oom_type=None,cleanup=None):
    """Leave the exception scope before cleanup; traceback must not retain CUDA tensors."""
    if oom_type is None:
        import torch
        oom_type=torch.cuda.OutOfMemoryError
    while True:
        failed=False
        try:
            return operation(batch)
        except oom_type:
            if batch<=1:raise RuntimeError('Not enough GPU memory even at batch 1. Close GPU apps or trim the clip.') from None
            failed=True
        if failed:
            batch=max(1,batch//2)
            if cleanup is None:
                import gc,torch
                gc.collect();torch.cuda.empty_cache()
            else:cleanup()
            from .memory_mode import effective_mode,announce
            if effective_mode()!='low_memory':
                os.environ['RGB_MOCAP_EFFECTIVE_MODE']='low_memory'
                announce('GPU inference','low_memory','Automatic recovery: reducing the image batch after CUDA memory pressure.')
            on_retry(batch)


def idle_timeout(last_output,last_cpu_change,now,limit=180):
    return now-max(last_output,last_cpu_change)>limit
