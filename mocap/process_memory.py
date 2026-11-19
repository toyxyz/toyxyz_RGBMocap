"""Peak process memory on Windows (peak working set) and Linux (peak resident set)."""


def peak_ram_bytes(process=None):
    import psutil
    process = process or psutil.Process()
    info = process.memory_info()
    peak = getattr(info, 'peak_wset', None)
    if peak is not None:
        return peak
    try:
        with open(f'/proc/{process.pid}/status', encoding='ascii', errors='replace') as stream:
            for line in stream:
                if line.startswith('VmHWM:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return info.rss


def private_bytes(process=None):
    import psutil
    info = (process or psutil.Process()).memory_info()
    return getattr(info, 'private', None) or max(0, info.rss - getattr(info, 'shared', 0))
