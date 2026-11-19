"""Cooperate across app, installation and updates; never break another process's lock."""
from contextlib import contextmanager
from pathlib import Path
import os

if os.name == 'nt':
    import msvcrt

    def _lock(stream): msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    def _unlock(stream): msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
else:
    # Linux port: advisory byte-range lock with the same non-blocking behaviour.
    import fcntl

    def _lock(stream): fcntl.lockf(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1)
    def _unlock(stream): fcntl.lockf(stream.fileno(), fcntl.LOCK_UN, 1)

@contextmanager
def app_lock(root):
    folder=Path(root)/'.setup';folder.mkdir(exist_ok=True)
    fd=os.open(folder/'app.lock',os.O_CREAT|os.O_RDWR,0o600)
    with os.fdopen(fd,'r+b') as stream:
        if os.fstat(stream.fileno()).st_size==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:_lock(stream)
        except OSError:raise RuntimeError('Close RGB MoCap or the other installation/update before continuing.') from None
        try:yield
        finally:stream.seek(0);_unlock(stream)
