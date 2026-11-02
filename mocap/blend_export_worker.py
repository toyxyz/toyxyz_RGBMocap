"""Derived Blender export worker; does not change inference cache identities."""
def main(emit):
    import argparse,sys,time
    from pathlib import Path
    from .compute_policy import configure_worker_cpu
    stage=sys.argv[1];configure_worker_cpu(stage)
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[2:]);started=time.monotonic()
    if stage=='save_blend_scene':from .blend_multicam_export import run
    elif stage=='save_blend':from .blend_export import run
    else:raise ValueError('Unknown Blender export stage')
    count=run(args.output,emit)
    emit('complete',stage=stage,files=count,elapsed=time.monotonic()-started)
    return 0
