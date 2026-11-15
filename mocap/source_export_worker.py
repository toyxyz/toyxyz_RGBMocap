"""Derived FBX/BVH saving with optional face sidecars."""
def main(emit):
    import argparse,sys,time
    from pathlib import Path
    from .compute_policy import configure_worker_cpu
    stage=sys.argv[1]
    configure_worker_cpu(stage)
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[2:]);started=time.monotonic()
    from .face_export import plan, publish
    try:
        sidecars=plan(args.output,stage)
        if stage=='save_fbx':
            from .fbx_export import run
        else:
            from .export_copy import run
        count=run(args.output,emit)
        publish(sidecars,emit)
    except Exception as error:
        import traceback
        traceback.print_exc()
        emit('failed',stage=stage,message=str(error))
        return 1
    emit('complete',stage=stage,files=count,elapsed=time.monotonic()-started)
    return 0
