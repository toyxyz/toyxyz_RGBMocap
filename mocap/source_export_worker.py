"""Derived BVH save entry without changing AI worker/cache fingerprints."""
def main(emit):
    import argparse,sys,time
    from pathlib import Path
    from .compute_policy import configure_worker_cpu
    configure_worker_cpu('copy_bvh')
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[2:]);started=time.monotonic()
    from .export_copy import run
    count=run(args.output,emit)
    emit('complete',stage='copy_bvh',files=count,elapsed=time.monotonic()-started)
    return 0
