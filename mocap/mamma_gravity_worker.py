"""GeoCalib in the pinned Python 3.10 Flow overlay; never import it into MAMMA."""
from pathlib import Path
import json,sys,time
import numpy as np
from .storage import atomic_json,sha256


def main():
    from .mamma_pipeline import emit
    path=Path(sys.argv[-1]).resolve()
    if path.name!='gravity-request.json' or path.stat().st_size>2*1024**2:
        raise ValueError('Invalid MAMMA gravity request.')
    work=path.parent
    if work.parent.name!='inference' or work.parent.parent.name!='mamma':
        raise ValueError('Gravity work must belong to a MAMMA inference generation.')
    job=json.loads(path.read_text('utf8'));digest=sha256(path)
    from .mamma_gravity import support_identity,estimate
    if job.get('version')!=1 or job['support']!=support_identity():
        raise ValueError('Camera alignment support changed. Close RGB MoCap and run install.cmd.')
    cameras=job['cameras'];frames=job['frames'];names=[c['id'] for c in cameras]
    if len(cameras)<2 or len(set(names))!=len(names) or not 1<=len(frames)<=3 or len(set(frames))!=len(frames):
        raise ValueError('Invalid camera/gravity samples.')
    if any(not isinstance(f,int) or not 0<=f<1800 for f in frames):raise ValueError('Invalid gravity sample time.')
    for c in cameras:
        if not isinstance(c['id'],str) or Path(c['id']).name!=c['id'] or len(c['id'])>120 or any(x in c['id'] for x in '/\\:'):
            raise ValueError('Invalid camera identity.')
        r=np.asarray(c['rotation']);k=np.asarray(c['intrinsics'])
        if r.shape!=(3,3) or k.shape!=(3,3) or not np.isfinite(r).all() or not np.isfinite(k).all() or np.max(np.abs(r@r.T-np.eye(3)))>1e-3 or np.linalg.det(r)<.999 or min(k[0,0],k[1,1])<=0:
            raise ValueError('Invalid VGGT camera matrices.')
    from .pear.alignment_support import activate
    activate()
    import cv2,torch
    from geocalib import GeoCalib
    from .models import BY_KEY
    from .result_artifacts import file_stat
    from .compute_policy import gpu_memory_fraction
    from .image_io import read_image
    weight=BY_KEY['pear_geocalib'];stamp=file_stat(weight.path)
    if sha256(weight.path)!=weight.expected_sha256:raise ValueError('Prepare Scene Gravity / GeoCalib in Model Manager.')
    torch.set_num_threads(4);cv2.setNumThreads(1);free,total=torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction(free,total));torch.cuda.reset_peak_memory_stats()
    emit('progress',current=0,total=0,message='MAMMA: loading GeoCalib background gravity')
    start=time.perf_counter();model=GeoCalib(str(weight.path)).cuda().eval();active=None
    def exclude_actor(module,args,out):
        for key in ('up_confidence','latitude_confidence'):
            mask=torch.nn.functional.interpolate(active,size=out[key].shape[-2:],mode='nearest')
            out[key]=out[key]*(mask[:,0] if out[key].ndim==3 else mask)
        return out
    hook=model.model.perspective_decoder.register_forward_hook(exclude_actor);observations=[];done=0
    try:
        with torch.inference_mode():
            for frame in frames:
                for ci,c in enumerate(cameras):
                    video=work/'videos'/(c['id']+'.mp4')
                    if not video.resolve().is_relative_to(work):raise ValueError('Linked gravity video leaves its generation.')
                    cap=cv2.VideoCapture(str(video))
                    try:
                        width=cap.get(cv2.CAP_PROP_FRAME_WIDTH);height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        if not 0<width<=1920 or not 0<height<=1080:raise ValueError('Gravity sample exceeds working-video bounds.')
                        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read()
                    finally:cap.release()
                    if not ok:raise ValueError('Cannot read a GeoCalib camera frame.')
                    background=read_image(work/'camera-images'/f'{frame:06}-{ci}.png')
                    if background is None or background.shape!=im.shape:raise ValueError('Invalid GeoCalib background sample.')
                    mask=(background.max(-1)>0).astype('float32')
                    tensor=torch.from_numpy(im[...,::-1].copy()).permute(2,0,1).float().cuda()/255
                    active=model.image_processor(torch.from_numpy(mask).cuda()[None,None])['image']
                    result=model.calibrate(tensor,priors={'focal':tensor.new_tensor(float(c['intrinsics'][1][1]))})
                    up=result['gravity'].vec3d[0].cpu().numpy();world=np.asarray(c['rotation']).T@up
                    uncertainty={key:float(value.detach().reshape(-1)[0].cpu()) for key,value in result.items() if 'uncertainty' in key}
                    observations.append(dict(camera=c['id'],frame=frame,up_saved_world=world.tolist(),up_camera=up.tolist(),uncertainty_radians=uncertainty,background_fraction=float(mask.mean())))
                    done+=1;emit('progress',current=done,total=len(frames)*len(cameras),message=f'MAMMA: GeoCalib background gravity {done}/{len(frames)*len(cameras)}')
        try:consensus=estimate(observations,names)
        except ValueError as error:raise ValueError('Cannot determine reliable background gravity. Import manual camera calibration or use clearer background views. '+str(error)) from error
        if sha256(path)!=digest or stamp!=file_stat(weight.path) or support_identity()!=job['support']:
            raise ValueError('GeoCalib inputs or support changed during processing.')
        atomic_json(work/'gravity-result.json',dict(version=1,token=job['token'],request_sha256=digest,support=job['support'],observations=observations,consensus=consensus,seconds=time.perf_counter()-start,peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/2**30))
        return 0
    finally:
        hook.remove();del model


if __name__=='__main__':raise SystemExit(main())
