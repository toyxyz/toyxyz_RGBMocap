"""Transport saved visual-odometry rotations into the committed source world."""
import json
from pathlib import Path
import numpy as np
from .result_artifacts import file_stat


def load(project, reference, frames, translation_scale=0.0):
    project=Path(project)
    state=json.loads((project/'project.json').read_text(encoding='utf-8'))
    model=state.get('pipeline',{}).get('model')
    paths=[]
    translation_scale=float(translation_scale)
    if not np.isfinite(translation_scale) or not 0<=translation_scale<=100:raise ValueError('Camera translation scale must be between 0 and 100')
    if translation_scale and model!='flowhmr':raise ValueError('Experimental camera translation requires FlowHMR')
    centers=None
    def read(path):
        paths.append((path,file_stat(path)))
        return json.loads(path.read_text(encoding='utf-8'))
    if model=='exmr':
        receipt=read(project/'exmr_camera_sources.json')
        root=(project/receipt['root']).resolve()
        if not project.resolve().is_relative_to(root):raise ValueError('Invalid camera root')
        path=root/'exmr_camera.npy';paths.append((path,file_stat(path)))
        if paths[-1][1]!=receipt['root_stat']:raise ValueError('ExMR camera changed. Prepare Preview again.')
        full=np.load(path,allow_pickle=False,mmap_mode='r')
        begin=int(receipt['first']);rotations=np.array(full[begin:begin+frames],dtype=float)
        baseline=np.asarray(receipt['reference_rotation'],float)
        if not np.allclose(full[0],baseline,atol=1e-6):raise ValueError('ExMR camera reference changed')
        description='ExMR saved camera rotation; fixed camera position (translation was not estimated)'
    elif model=='flowhmr':
        from .flow_placement import current
        if not current(project):raise ValueError('FlowHMR placement is out of date')
        placement=read(project/'flow_world.json')
        path=project/'flow_camera.npz';paths.append((path,file_stat(path)))
        with np.load(path,allow_pickle=False) as data:
            raw=np.asarray(data['camera_RT_yup'],float)
            rotations=raw[:,:3,:3]
            centers=-np.einsum('tji,tj->ti',rotations,raw[:,:3,3])
            if not np.isfinite(centers).all():raise ValueError('Invalid VO camera translation')
        # The committed shared reference already applies the WV scene basis
        # and metric camera height. Native monocular translation is not metric.
        root=(project/state.get('people_root','.')).resolve()
        if not project.resolve().is_relative_to(root):raise ValueError('Invalid FlowHMR camera root')
        path=root/'flow_camera.npz';paths.append((path,file_stat(path)))
        with np.load(path,allow_pickle=False) as data:baseline=np.array(data['camera_RT_yup'][0,:3,:3],float)
        description='FlowHMR saved camera rotation; fixed metric camera position (unscaled VO translation omitted)'
    else:
        return None,[], 'Stored reference camera; supported saved rotation tracking unavailable for this model'
    if rotations.shape!=(frames,3,3) or not np.isfinite(rotations).all():raise ValueError('Camera tracking does not match the take')
    for array in (rotations,baseline):
        if not np.allclose(array.swapaxes(-1,-2)@array,np.eye(3),atol=2e-4) or not np.allclose(np.linalg.det(array),1,atol=2e-4):
            raise ValueError('Invalid saved camera rotations')
    delta=rotations@baseline.T
    result=np.broadcast_to(np.eye(4),(frames,4,4)).copy()
    result[:,:3,:]=delta@np.asarray(reference,float)[:3,:]
    if translation_scale:
        # Convert displacement of camera CENTERS, not extrinsic t, to the
        # committed world. Anchor at take frame zero to retain alignment.
        basis=np.asarray(reference,float)[:3,:3].T@baseline
        displacement=(centers-centers[0])@basis.T*translation_scale
        result[:,:3,3]-=np.einsum('tij,tj->ti',result[:,:3,:3],displacement)
        description='Experimental FlowHMR camera translation, scale '+str(translation_scale)+'; saved rotation; translation anchored at take start'
    if any(file_stat(p)!=stamp for p,stamp in paths):raise ValueError('Camera changed during export')
    return result,paths,description
