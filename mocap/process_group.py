"""Cancel POSIX workers together with their children; Windows uses taskkill /T."""
import os
import signal


def own_session(process):
    """Start a QProcess as a session leader so its process group holds every descendant."""
    if os.name != 'nt':
        from PySide6.QtCore import QProcess
        process.setUnixProcessParameters(QProcess.UnixProcessFlag.CreateNewSession)


def kill_group(pid):
    """SIGKILL the process group led by pid; an already-exited group is ignored."""
    if os.name != 'nt' and pid:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
