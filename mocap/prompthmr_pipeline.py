"""Atomic, cancellable PromptHMR pipeline for one video scene."""
from pathlib import Path
import argparse,gc,hashlib,json,math,os,subprocess,sys,time,uuid
from .storage import atomic_json,sha256
from .result_artifacts import file_stat

STAGES=('prompthmr_prepare','prompthmr_track','prompthmr_pose','prompthmr_world','prompthmr_publish')


def emit(kind,**data):
    print('@@MOCAP@@'+json.dumps(dict(type=kind,**data),ensure_ascii=False),flush=True)


def options(value):
    count=value.get('people',2)
    if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=8:raise ValueError('PromptHMR People must be 0 (automatic) or 1–8.')
    interaction=value.get('interaction',True)
    if not isinstance(interaction,bool):raise ValueError('Invalid PromptHMR interaction option.')
    camera=value.get('camera','static')
    if camera not in ('static','moving'):raise ValueError('Invalid PromptHMR camera mode.')
    result=dict(people=count,interaction=interaction)
    if camera=='moving':result['camera']=camera
    return result


def request(path):
    path=Path(path)
    if path.stat().st_size>2*1024**2:raise ValueError('PromptHMR request is too large.')
    job=json.loads(path.read_text(encoding='utf8'))
    if job.get('version')!=1:raise ValueError('Unsupported PromptHMR request.')
    if not all(math.isfinite(float(job[k])) for k in ('start','end')) or not 0<=job['start']<job['end']:
        raise ValueError('Invalid PromptHMR trim range.')
    source=Path(job['video']).resolve()
    if not source.is_file() or source.suffix.lower() not in ('.mp4','.mov','.avi','.mkv','.webm','.m4v'):raise ValueError('Open a supported video file first.')
    job['root']=Path(job['root']).resolve();job['video']=str(source);job['options']=options(job.get('options',{}))
    from .correction_settings import correction_options
    correction_options(job['correction'])
    if not isinstance(job.get('correction_enabled'),bool):raise ValueError('Invalid correction selection.')
    return job


def work_for(job):
    work=(job['root']/job.get('work','')).resolve()
    if not job.get('work') or not work.is_relative_to(job['root']/'prompthmr/inference'):raise ValueError('Invalid PromptHMR generation path.')
    return work


def validate_inputs(job):
    from .models import BY_KEY
    if file_stat(job['video'])!=job['identity']['input']['stat']:raise ValueError('The video changed during processing. Previous results were preserved.')
    for key,stamp in job['identity']['models'].items():
        if file_stat(BY_KEY[key].path)!=stamp:raise ValueError('A PromptHMR model changed during processing. Previous results were preserved.')
    if job['options'].get('camera')=='moving':
        from .prompthmr_camera_runtime import configuration as camera_configuration
        cfg=camera_configuration()
        if {k:cfg[k] for k in ('support_id','base_id','compute_capability','files')}!=job['identity'].get('camera_runtime'):
            raise ValueError('PromptHMR camera support changed during processing. Previous motion was preserved.')


def prepare(path,job):
    from .prompthmr_cache import inference_code_hash
    from .prompthmr_runtime import configuration
    from .prompthmr_models import KEYS
    from .models import BY_KEY,missing
    from .video import probe,executable
    from .project_workspace import create
    from .paths import ROOT
    cfg=configuration()
    if missing('prompthmr_assets'):raise ValueError('Prepare the missing PromptHMR files in Model Manager first.')
    info=probe(Path(job['video']));n=round((job['end']-job['start'])*30)
    if job['end']>info['duration']+1e-3 or not 4<=n<=1800:
        raise ValueError('Use a trim of 4–1,800 frames (at 30 fps) within the source video. No frames were dropped.')
    identity=dict(version=1,input=dict(path=job['video'],stat=file_stat(job['video'])),start=job['start'],frames=n,
        options=job['options'],models={k:file_stat(BY_KEY[k].path) for k in KEYS},runtime=cfg['managed_id'],engine=cfg['engine_files'],
        code={name:inference_code_hash(ROOT/'mocap'/name) for name in ('prompthmr_pipeline.py','image_io.py')})
    if job['options'].get('camera')=='moving':
        from .prompthmr_camera_runtime import configuration as camera_configuration
        from .prompthmr_models import CAMERA_KEYS
        if missing('prompthmr_camera_assets'):raise ValueError('Prepare the Moving Camera files in Model Manager first.')
        camera_cfg=camera_configuration(verify=True)
        identity['camera_runtime']={k:camera_cfg[k] for k in ('support_id','base_id','compute_capability','files')}
        identity['models'].update({k:file_stat(BY_KEY[k].path) for k in CAMERA_KEYS})
        identity['code'].update({name:sha256(ROOT/'mocap'/name) for name in ('prompthmr_pipeline.py','prompthmr_camera.py','prompthmr_camera_runtime.py')})
    token=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    work=job['root']/'prompthmr/inference'/token;cached=False
    try:
        receipt=json.loads((work/'complete.json').read_text(encoding='utf8'))
        cached=receipt['identity']==identity and all(file_stat(work/k)==v for k,v in receipt['outputs'].items())
    except (OSError,ValueError,KeyError,TypeError):pass
    if work.exists() and not cached:work=work.with_name(token+'-'+uuid.uuid4().hex[:12])
    create(job['root'],Path(job['video']),info,'prompthmr')
    work.mkdir(parents=True,exist_ok=True)
    job.update(work=work.relative_to(job['root']).as_posix(),frames=n,identity=identity,cached=cached,source=info)
    atomic_json(path,{**job,'root':str(job['root'])})
    if cached:
        emit('progress',current=1,total=1,message='Reusing verified PromptHMR inference.');return
    atomic_json(work/'pending.json',identity)
    vf=f"trim=start={job['start']:.9f}:end={job['end']:.9f},setpts=PTS-STARTPTS,fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2"
    subprocess.run([executable('ffmpeg'),'-v','error','-i',job['video'],'-an','-vf',vf,'-frames:v',str(n),'-fps_mode','cfr',
        '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','2',str(work/'input.mp4')],check=True,creationflags=0x08000000)
    result=probe(work/'input.mp4')
    if result['frames']!=n or abs(result['fps']-30)>.001:raise ValueError('Working video frame count does not match the selected trim.')
    atomic_json(work/'media.json',result)


def link_assets(engine,links=None):
    from .models import BY_KEY,ready
    from .prompthmr_models import LINKS
    import shutil
    for key,name in (LINKS if links is None else links).items():
        model=BY_KEY[key]
        if not ready(model):raise ValueError('Missing or unverified model: '+model.name)
        p=engine/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.is_file() and p.samefile(model.path):continue
        if p.exists():p.unlink()
        try:os.link(model.path,p)
        except OSError:
            # An alternate model drive may disallow hard links. Explicit local
            # model copies are bounded to this profile, never automatic downloads.
            print('PromptHMR: cross-drive model copy required for '+model.path.name,flush=True)
            shutil.copy2(model.path,p)


def save_result(value,target):
    import joblib
    tmp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
    try:joblib.dump(value,tmp,compress=0);os.replace(tmp,target)
    finally:tmp.unlink(missing_ok=True)


def numpyify(value):
    import torch
    if isinstance(value,dict):return {k:numpyify(v) for k,v in value.items()}
    if isinstance(value,torch.Tensor):return value.detach().cpu().numpy()
    return value


def inference(stage,job,engine):
    import torch,joblib,numpy as np
    from .compute_policy import gpu_memory_fraction
    from .memory_mode import announce,requested_mode
    from .progress_events import tqdm_events
    work=work_for(job);link_assets(engine)
    torch.set_num_threads(4);torch.manual_seed(0)
    free,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction(free,total))
    announce(stage,requested_mode(),'PromptHMR keeps the validated one-image batch and full temporal context in both modes. Additional block offloading is not implemented; Resource Safety controls the RAM/VRAM stop limits.',emit)
    os.chdir(engine);sys.path.insert(0,str(engine))
    from .prompthmr_cache import configure_downloads
    configure_downloads(work)
    os.environ.update(MPLCONFIGDIR=str(work/'mpl-cache'),YOLO_CONFIG_DIR=str(work/'yolo-cache'))
    from pipeline import Pipeline
    pipe=Pipeline(static_cam=job['options'].get('camera')!='moving');pipe.cfg.num_max_people=job['options']['people'] or 8
    pipe.images,pipe.seq_folder=pipe.load_frames(str(work/'input.mp4'),str(work/'frames'))
    if len(pipe.images)!=job['frames']:raise ValueError('PromptHMR decoded a different frame count.')
    pipe.cfg.seq_folder=pipe.seq_folder
    pipe.results=dict(camera={},people={},timings={},masks=None,has_tracks=False,has_hps_cam=False,has_hps_world=False,has_slam=False,has_2d_kpts=False,has_post_opt=False)
    previous={'prompthmr_pose':'tracking.pkl','prompthmr_world':'camera-motion.pkl'}
    if stage in previous:pipe.results=joblib.load(work/previous[stage])
    with tqdm_events(emit):
        if stage=='prompthmr_track':
            pipe.run_detect_track()
            if not pipe.results['people']:raise ValueError('No people were found. Use a trim with the intended people visible near the beginning.')
            save_result(pipe.results,work/'tracking.pkl')
        elif stage=='prompthmr_pose':
            from pipeline.spec import run_cam_calib
            emit('progress',current=0,total=0,message='Estimate camera intrinsics and gravity')
            pipe.results['spec_calib']=run_cam_calib(pipe.images,out_folder=str(work/'spec'),save_res=True,stride=max(1,len(pipe.images)//30),method='spec',first_frame_idx=0)
            if job['options'].get('camera')=='moving':
                from .prompthmr_camera import estimate
                estimate(pipe,emit)
            else:pipe.camera_motion_estimation()
            pipe.estimate_2d_keypoints()
            from pipeline.phmr_vid import PromptHMRVideoDataset
            original=PromptHMRVideoDataset.__getitem__
            def item(dataset,index):
                batch=original(dataset,index);batch['interaction']=job['options']['interaction'];return batch
            PromptHMRVideoDataset.__getitem__=item
            emit('progress',current=0,total=0,message='PromptHMR image and full-context video motion')
            try:pipe.hps_estimation()
            finally:PromptHMRVideoDataset.__getitem__=original
            save_result(numpyify(pipe.results),work/'camera-motion.pkl')
        elif stage=='prompthmr_world':
            emit('progress',current=0,total=0,message='Shared world placement and native post-optimization')
            pipe.world_hps_estimation();pipe.results=numpyify(pipe.results);save_result(pipe.results,work/'world-before-postopt.pkl')
            pipe.post_optimization()
            result=numpyify(pipe.results)
            # Masks/features are upstream intermediates, not UI playback data.
            for person in result['people'].values():
                for key in ('masks','prhmr_img_feats'):person.pop(key,None)
            save_result(result,work/'results.pkl')
            validate_inputs(job)
            atomic_json(work/'complete.json',dict(identity=job['identity'],outputs={name:file_stat(work/name) for name in ('results.pkl','input.mp4','media.json')}))
            (work/'pending.json').unlink(missing_ok=True)
        else:raise ValueError('Unknown PromptHMR inference stage.')
    del pipe;gc.collect();torch.cuda.empty_cache()


def _run(argv=None,engine=None):
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=STAGES);parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args(argv);started=time.monotonic();job=request(args.request)
    if args.stage=='prompthmr_prepare':prepare(args.request,job)
    else:
        validate_inputs(job)
        if args.stage=='prompthmr_publish':
            from .prompthmr_publish import publish
            publish(job,emit)
        elif not job.get('cached'):inference(args.stage,job,engine)
        else:emit('progress',current=1,total=1,message='Verified PromptHMR inference cache')
    emit('complete',stage=args.stage,cached=job.get('cached',False),elapsed=time.monotonic()-started)
    return 0


def main(argv=None,engine=None):
    try:
        from .image_io import unicode_image_paths
        with unicode_image_paths():
            return _run(argv,engine)
    except Exception as error:
        import traceback
        traceback.print_exc()
        message=str(error)
        if isinstance(error,MemoryError) or 'out of memory' in message.lower():
            message='PromptHMR ran out of memory and stopped. Shorten the trim, reduce People or close other GPU applications. Both resource modes retain one-image batches; additional PromptHMR offloading is not implemented.'
        emit('failed',message=message)
        return 1
