"""Bounded, transactional helpers for portable Blender exports."""
import json,os,shutil,subprocess,time
from pathlib import Path
from .result_artifacts import file_stat


def read(path,limit=4*1024**2):
    path=Path(path)
    if path.stat().st_size>limit:raise ValueError('Blender export metadata exceeds its size limit')
    return json.loads(path.read_text(encoding='utf-8'))


class Inputs:
    def __init__(self):self.files={}
    def track(self,path):
        path=Path(path);stamp=file_stat(path)
        if path in self.files and self.files[path]!=stamp:raise ValueError('Export input changed while reading')
        self.files[path]=stamp
        return path
    def verify(self):
        if any(file_stat(p)!=s for p,s in self.files.items()):raise ValueError('Export inputs changed. Retry without changing the project.')


def partial(folder,request):
    folder=Path(folder)
    return folder.with_name('.'+folder.name+'.rgbmocap-'+Path(request).stem+'.partial')


def cleanup(path,request):
    """Remove only an owned staging directory, never a finished/user folder."""
    path=Path(path);request=Path(request)
    if not path.exists():return
    if (not path.name.startswith('.') or not path.name.endswith('.rgbmocap-'+request.stem+'.partial')
            or path.is_symlink() or getattr(path,'is_junction',lambda:False)() or path.resolve().parent!=path.parent.resolve()):
        raise ValueError('Invalid Blender export staging directory')
    marker=path/'.rgbmocap-owner'
    if not marker.is_file() or marker.read_text(encoding='utf-8')!=request.stem:
        raise ValueError('Unowned Blender export staging directory')
    shutil.rmtree(path)


class Bundle:
    def __init__(self,target,request):
        self.target=Path(target).resolve();self.folder=self.target.parent;self.request=Path(request)
        if self.target.suffix.lower()!='.blend' or self.folder.exists() or not self.folder.parent.is_dir():
            raise ValueError('Choose a new Blender bundle folder in an existing directory')
        self.work=partial(self.folder,self.request);self.work.mkdir()
        try:(self.work/'.rgbmocap-owner').write_text(self.request.stem,encoding='utf-8')
        except Exception:self.work.rmdir();raise
    def close(self):cleanup(self.work,self.request)
    def publish(self):
        if self.folder.exists():raise ValueError('Blender bundle destination was created during export; existing files were retained')
        if not (self.work/self.target.name).is_file():raise ValueError('Blender did not create the project')
        self.work.rename(self.folder)
        (self.folder/'.rgbmocap-owner').unlink()


def copy_video(source,target,emit,stage):
    size=Path(source).stat().st_size;done=0;last=0.
    with Path(source).open('rb') as inp,Path(target).open('xb') as out:
        while True:
            chunk=inp.read(4*1024**2)
            if not chunk:break
            out.write(chunk);done+=len(chunk);now=time.monotonic()
            if now-last>.2 or done==size:
                emit('progress',stage=stage,current=done,total=size,message='Copying synchronized video');last=now


def build(work,script,emit,stage):
    from .retarget.runtime import blender
    work=Path(work);logpath=work/'blender.log'
    cmd=[str(blender()),'-b','--factory-startup','--disable-autoexec','--threads','3','--python-exit-code','1',
         '--python',str(Path(__file__).with_name(script)),'--',str(work)]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    p=None
    try:
        with logpath.open('wb') as log,logpath.open('rb') as reader:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            last=0.;carry=b''
            while True:
                try:p.wait(timeout=.2);done=True
                except subprocess.TimeoutExpired:done=False
                data=reader.read(256*1024)
                if data:
                    lines=(carry+data).split(b'\n');carry=lines.pop()[-16384:]
                    for line in lines:
                        if line.startswith(b'@@BLEND@@'):
                            event=json.loads(line[len(b'@@BLEND@@'):]);emit('progress',stage=stage,**event);last=time.monotonic()
                if not done and time.monotonic()-last>2:
                    emit('progress',stage=stage,current=0,total=0,message='Building Blender project');last=time.monotonic()
                if done:break
        if p.returncode:
            with logpath.open('rb') as log:log.seek(max(0,logpath.stat().st_size-3000));detail=log.read().decode('utf-8',errors='replace')
            raise RuntimeError('Blender export failed: '+detail)
    finally:
        if p is not None and p.poll() is None:
            p.terminate()
            try:p.wait(timeout=2)
            except subprocess.TimeoutExpired:p.kill();p.wait()


def disk_check(folder,required):
    if shutil.disk_usage(Path(folder)).free<required+256*1024**2:
        raise ValueError('Insufficient free disk space for the Blender bundle, videos and mesh caches')
