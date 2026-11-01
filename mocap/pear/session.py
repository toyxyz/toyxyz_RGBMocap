"""Pipe ownership and bounded mailboxes; no blocking work on the GUI thread."""
import subprocess,threading,json,time,os,uuid
from pathlib import Path
from ..paths import ROOT,worker_command
from .wire import receive

class Session:
    def __init__(self,stage="pear_live"):
        self.lock=threading.Condition();self.latest=None;self.command=None;self.closed=False;self.topology=None
        folder=ROOT/'cache/pear/sessions';folder.mkdir(parents=True,exist_ok=True)
        # One bounded log per session; retain only the most recent eight sessions.
        for old in sorted(folder.glob('*.log'),key=lambda p:p.stat().st_mtime)[:-7]:
            try:old.unlink()
            except OSError:pass
        self.log=folder/(uuid.uuid4().hex+'.log')
        log=self.log.open('wb');env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUNBUFFERED']='1'
        try:self.process=subprocess.Popen(worker_command(stage),cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        finally:log.close()
        from ..processing_guard import ProcessingGuard
        self.guard=ProcessingGuard(self.process.pid)
        threading.Thread(target=self.read,daemon=True).start();threading.Thread(target=self.write,daemon=True).start()
    def read(self):
        try:
            while not self.closed:
                meta,arrays=receive(self.process.stdout)
                self.guard.output()
                with self.lock:
                    if meta['type']=='topology':self.topology=(meta,arrays)
                    else:self.latest=(meta,arrays)
        except Exception as exc:
            with self.lock:
                if not self.closed and not (self.latest and self.latest[0]['type'] in ('error','alignment')):
                    self.latest=({'type':'error','text':'PEAR worker stopped. See '+str(self.log)}, {})
        finally:self.process.stdout.close()
    def write(self):
        try:
            while True:
                with self.lock:
                    while not self.command and not self.closed:self.lock.wait()
                    if self.closed:break
                    value=self.command;self.command=None
                self.process.stdin.write((json.dumps(value,ensure_ascii=False)+'\n').encode('utf8'));self.process.stdin.flush()
        except (OSError,ValueError):pass
        finally:
            try:self.process.stdin.close()
            except OSError:pass
    def submit(self,value):
        with self.lock:self.command=value;self.lock.notify_all()
    def poll(self):
        with self.lock:
            value=self.latest;self.latest=None;top=self.topology;self.topology=None
            return top,value
    def close(self):
        with self.lock:self.closed=True;self.command=None;self.lock.notify_all()
        def reap():
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        threading.Thread(target=reap,daemon=False).start()
