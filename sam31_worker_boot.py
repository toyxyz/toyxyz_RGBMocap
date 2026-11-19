"""Supervise SAM 3.1 attempts in separate verified ExMR processes."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from mocap.runtime_registry import activate
from mocap.sam31_discovery import RETRY_WITH_OFFLOAD, main
from mocap.worker_progress import emit


def run_worker(arguments, force_offload):
    try:
        activate(root, 'exmr')
    except (OSError, RuntimeError, ValueError) as error:
        emit('failed', stage='sam31_people', message=str(error))
        return 1
    return main(arguments, emit, force_offload=force_offload)


def supervise(arguments):
    command = [sys.executable, '-I', '-X', 'utf8', '-B', '-u', str(Path(__file__).resolve())]
    try:
        code = subprocess.run([*command, '--tracking-attempt', *arguments], check=False).returncode
        if code == RETRY_WITH_OFFLOAD:
            code = subprocess.run([*command, '--offload-attempt', *arguments], check=False).returncode
        return code
    except OSError as error:
        emit('failed', stage='sam31_people', message=f'Could not start SAM 3.1 tracking. {error}')
        return 1


if __name__ == '__main__':
    arguments = sys.argv[1:]
    if arguments and arguments[0] in ('--tracking-attempt', '--offload-attempt'):
        raise SystemExit(run_worker(arguments[1:], arguments[0] == '--offload-attempt'))
    raise SystemExit(supervise(arguments))
