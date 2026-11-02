"""Isolated, cancellable CPU background-camera work and a bounded session cache."""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,time
import numpy as np
from .paths import HOME,worker_command
from .result_artifacts import file_stat

MAX_BYTES=32*1024**2


def current(sources):
    try:return all(file_stat(Path(path))==stamp for path,stamp in sources)
    except OSError:return False


def refined(data,project,cancelled,progress,cache):
    """Return a display-only camera copy; never rewrite original camera receipts."""
    if data['model'] not in ('flowhmr','exmr') or data.get('static_camera',False):
        return dict(data,background_report=dict(skipped='Static camera: background rotation refinement skipped.'))
    project=Path(project);state=json.loads((project/'project.json').read_text(encoding='utf-8'))
    camera=data['cameras'][0];fps=float(state['fps']);count=int(state['frames'])
    first=round((float(state['start'])-camera['start'])*fps);matrices=camera['matrices']
    if abs(camera['fps']-fps)>1e-6 or first<0 or first+count>len(matrices):
        raise ValueError('Saved camera does not cover the selected background video')
    track=np.asarray(matrices[first:first+count],float)
    if track.nbytes>MAX_BYTES:raise ValueError('Background camera exceeds its preview size limit')
    digest=hashlib.sha256(track.tobytes()+camera['K'].tobytes())
    for name in ('camera_background.py','camera_background_worker.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    key=(str(project.resolve()),digest.hexdigest(),first,count)
    prior=cache.get(key)
    if prior and current(prior['sources']):
        track_result=prior['matrices'];report=dict(prior['report'],cache_hit=True)
    else:
        folder=HOME/'cache'/'camera_background';folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='job-',dir=folder) as directory:
            work=Path(directory);np.save(work/'input.npy',track,allow_pickle=False)
            (work/'job.json').write_text(json.dumps(dict(project=str(project.resolve()),K=camera['K'].tolist())),encoding='utf-8')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            p=subprocess.Popen(worker_command('camera_background','--work',str(work)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            deadline=time.monotonic()+180;last=(-1,-1)
            try:
                while True:
                    if cancelled():raise InterruptedError('Background camera refinement cancelled')
                    if time.monotonic()>deadline:raise ValueError('Background camera refinement timed out; original camera retained')
                    try:output,error=p.communicate(timeout=.1);done=True
                    except subprocess.TimeoutExpired as wait:
                        output,error=wait.output or b'',wait.stderr or b'';done=False
                    if len(output)+len(error)>512*1024:raise ValueError('Background camera worker log exceeded its limit')
                    event=None
                    progress_path=work/'progress.json'
                    try:
                        if progress_path.stat().st_size<=1024:event=json.loads(progress_path.read_text(encoding='utf-8'))
                    except (OSError,ValueError):pass
                    if event is not None:
                        value=(event.get('current',0),event.get('total',0))
                        if all(type(x) is int for x in value) and 0<=value[0]<=value[1]<=600:
                            progress(*value);last=value
                    elif not done:progress(0,0)
                    for line in reversed(output.splitlines()):
                        if not line.startswith(b'@@BACKGROUND@@'):continue
                        try:event=json.loads(line[len(b'@@BACKGROUND@@'):]);value=(event['current'],event['total'])
                        except (ValueError,KeyError):continue
                        if value!=last:progress(*value);last=value
                        break
                    if done:break
                if p.returncode:raise ValueError(error.decode('utf-8',errors='replace').strip()[-800:] or 'Background camera refinement failed')
            finally:
                if p.poll() is None:
                    p.terminate()
                    try:p.communicate(timeout=1)
                    except subprocess.TimeoutExpired:p.kill();p.communicate()
            if cancelled():raise InterruptedError('Background camera refinement cancelled')
            report_path=work/'report.json'
            if report_path.stat().st_size>1024**2:raise ValueError('Background report exceeds its size limit')
            record=json.loads(report_path.read_text(encoding='utf-8'));sources=record['sources']+list(data.get('camera_sources',{}).items())
            if not current(sources):raise ValueError('Camera/background inputs changed during refinement')
            mapped=np.load(work/'result.npy',mmap_mode='r',allow_pickle=False)
            try:
                if mapped.shape!=track.shape or mapped.dtype!=np.float64 or mapped.nbytes>MAX_BYTES or not np.isfinite(mapped).all():raise ValueError('Invalid refined camera output')
                track_result=None if np.array_equal(mapped,track) else np.array(mapped)
            finally:mapped._mmap.close()
            report=dict(record['report'],cache_hit=False)
        while cache and (len(cache)>=4 or sum(v['matrices'].nbytes for v in cache.values() if v['matrices'] is not None)+(0 if track_result is None else track_result.nbytes)>MAX_BYTES):
            cache.pop(next(iter(cache)))
        cache[key]=dict(matrices=track_result,report=report,sources=sources)
    if cancelled():raise InterruptedError('Background camera refinement cancelled')
    result=dict(data);cam=dict(camera)
    if track_result is not None:
        updated=matrices.copy();updated[first:first+count]=track_result;cam['matrices']=updated
    result['cameras']=[cam];result['background_report']=report
    result['note']+=' Background rotation refinement: '+str(report.get('accepted',0))+'/'+str(len(report.get('samples',[])))+' samples accepted; source motion and camera centers unchanged.'
    return result


def main():
    import argparse
    import sys
    parser=argparse.ArgumentParser();parser.add_argument('--work',type=Path,required=True);args=parser.parse_args(sys.argv[2:])
    work=args.work
    try:
        if (work/'job.json').stat().st_size>16*1024:raise ValueError('Invalid camera job size')
        job=json.loads((work/'job.json').read_text(encoding='utf-8'))
        import psutil
        if psutil.virtual_memory().available<512*1024**2:raise MemoryError('Insufficient available RAM for background camera refinement')
        import cv2
        cv2.setNumThreads(4)
        from .camera_background import refine
        from .camera_tracks import valid_rotations
        track=np.load(work/'input.npy',mmap_mode='r',allow_pickle=False)
        try:
            if track.ndim!=3 or track.shape[1:]!=(4,4) or track.nbytes>MAX_BYTES:raise ValueError('Invalid input camera dimensions')
            valid_rotations(track[:,:3,:3])
            start=time.monotonic()
            last_progress=[0.,-1]
            def progress(current,total):
                now=time.monotonic()
                if total==last_progress[1] and current!=total and now-last_progress[0]<.2:return
                event=dict(current=current,total=total);temporary=work/'progress.tmp'
                temporary.write_text(json.dumps(event),encoding='utf-8');os.replace(temporary,work/'progress.json')
                print('@@BACKGROUND@@'+json.dumps(event),flush=True);last_progress[:]=[now,total]
            result,sources,report=refine(Path(job['project']),track,np.asarray(job['K'],float),progress=progress)
            valid_rotations(result[:,:3,:3])
            before=-np.einsum('tji,tj->ti',track[:,:3,:3],track[:,:3,3]);after=-np.einsum('tji,tj->ti',result[:,:3,:3],result[:,:3,3])
            if not np.allclose(before,after,rtol=0,atol=1e-7):raise ValueError('Background refinement changed camera centers')
            report['elapsed_seconds']=time.monotonic()-start
            np.save(work/'result.npy',result,allow_pickle=False)
            (work/'report.json').write_text(json.dumps(dict(sources=[(str(path),stamp) for path,stamp in sources],report=report)),encoding='utf-8')
        finally:track._mmap.close()
        return 0
    except Exception as error:
        import sys
        print(str(error),file=sys.stderr,flush=True);return 1
