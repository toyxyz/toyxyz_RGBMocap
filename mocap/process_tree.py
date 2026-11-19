"""POSIX counterpart of `taskkill /T /F`: stop a worker and its descendants."""
import os
import signal


def kill_tree(pid):
    if not pid:
        return
    try:
        import psutil
        root = psutil.Process(int(pid))
        processes = [*root.children(recursive=True), root]
    except Exception:
        processes = []
    for process in processes:
        try:
            process.kill()
        except Exception:
            pass
    if not processes:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass
