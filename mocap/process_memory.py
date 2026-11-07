"""Process resource figures for logs and receipts on every platform (several psutil fields are Windows-only)."""
import os
import sys


def peak_ram_bytes():
    """Lifetime peak resident memory of this process: Windows peak working set, POSIX ru_maxrss."""
    if os.name == 'nt':
        import psutil
        return psutil.Process().memory_info().peak_wset
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == 'darwin' else peak * 1024  # macOS reports bytes, Linux KiB


def open_handles(process):
    """Kernel objects held by `process` (a psutil.Process): Windows handles, POSIX file descriptors."""
    return process.num_handles() if os.name == 'nt' else process.num_fds()


def private_bytes(process):
    """Memory owned only by `process` (a psutil.Process): Windows private bytes, POSIX unique set size."""
    return process.memory_info().private if os.name == 'nt' else process.memory_full_info().uss
