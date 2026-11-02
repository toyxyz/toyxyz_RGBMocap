"""Read an original paired SMPL-X camera in an isolated CPU process; no writes."""
from pathlib import Path
import json,zipfile
import numpy as np


def reference(project):
    from .result_artifacts import artifacts_current,file_stat
    from .scene_camera import paired_camera,validate_transform
    project=Path(project).resolve()
    state=json.loads((project/'project.json').read_text(encoding='utf-8'))
    model=state.get('pipeline',{}).get('model')
    if model not in ('exmr','gvhmr'):raise ValueError('Paired source camera is unsupported for this model.')
    if not artifacts_current(project,'initial'):raise ValueError('Original motion preview is stale. Use Prepare Preview.')
    names=['hmr4d_results.pt','initial_joints.npz','initial_mesh.json','tracking.npz']
    if (project/'initial_artifacts.json').exists():names.append('initial_artifacts.json')
    saved=cam_root=None
    if model=='exmr':
        names.append('exmr_camera_sources.json')
        saved=json.loads((project/'exmr_camera_sources.json').read_text(encoding='utf-8'))
        cam_root=(project/saved['root']).resolve()
        if not project.is_relative_to(cam_root):raise ValueError('Invalid ExMR camera root.')
        import os
        names.extend(os.path.relpath(cam_root/name,project) for name in ('exmr_camera.json','exmr_camera.npy'))
    sources={name:file_stat(project/name) for name in names}
    import psutil
    if psutil.virtual_memory().available<512*1024**2:raise MemoryError('Free at least 512 MiB RAM to read the saved camera.')
    import torch
    torch.set_num_threads(1)
    from scipy.spatial.transform import Rotation
    motion=torch.load(project/'hmr4d_results.pt',map_location='cpu',weights_only=True,mmap=True)
    world,incam=motion['smpl_params_global'],motion['smpl_params_incam']
    # Read only one joint directly from the archive; do not extract a preview cache.
    with zipfile.ZipFile(project/'initial_joints.npz') as archive:
        with archive.open('joints.npy') as stream:
            version=np.lib.format.read_magic(stream)
            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            shape,fortran,dtype=reader(stream)
            if fortran or dtype.kind!='f' or len(shape)!=3 or shape[0]!=len(world['transl']) or shape[1:]!=(22,3):
                raise ValueError('Original source joints do not match the saved motion.')
            root=np.frombuffer(stream.read(3*dtype.itemsize),dtype=dtype).astype(float)
    pelvis=root-world['transl'][0].numpy()
    if pelvis.shape!=(3,) or not np.isfinite(pelvis).all():raise ValueError('Invalid original pelvis pivot.')
    rotation=lambda params:Rotation.from_rotvec(params['global_orient'][0].numpy()).as_matrix()
    matrix=paired_camera(rotation(world),rotation(incam),world['transl'][0].numpy(),incam['transl'][0].numpy(),pelvis)
    centers=None
    if model=='exmr':
        align=np.asarray(saved['reference_rotation'])@np.asarray(saved['first_rotation']).T
        matrix[:3,:3]=align@matrix[:3,:3];matrix[:3,3]=align@matrix[:3,3]
        policy=json.loads((cam_root/'exmr_camera.json').read_text(encoding='utf-8'))
        if not policy['identity']['static']:
            rotations=np.load(cam_root/'exmr_camera.npy',mmap_mode='r',allow_pickle=False)
            try:
                count=len(world['transl']);first=saved['first']
                if count*3*8>32*1024**2:raise ValueError('Camera path exceeds the preview memory limit.')
                r=np.asarray(rotations[first:first+count],float)@np.asarray(saved['reference_rotation']).T@matrix[:3,:3]
                if len(r)!=count:raise ValueError('ExMR camera does not cover the original take.')
                # Relative camera-center estimates from ORIGINAL paired roots.
                # Body/inference errors remain visible; no pose refitting occurs.
                centers=world['transl'].numpy()+pelvis-np.einsum('tji,tj->ti',r,incam['transl'].numpy()+pelvis)
                if not np.isfinite(centers).all():raise ValueError('Invalid estimated camera path.')
            finally:rotations._mmap.close()
    with np.load(project/'tracking.npz',allow_pickle=False) as data:k=data['K'].tolist()
    if not artifacts_current(project,'initial') or any(file_stat(project/name)!=stamp for name,stamp in sources.items()):
        raise ValueError('Original camera inputs changed while loading.')
    return dict(schema=2,world_to_camera=validate_transform(matrix).tolist(),K=k,sources=sources,
        centers=None if centers is None else centers.tolist(),start=state['start'],fps=state['fps'])


def main():
    import sys
    from .compute_policy import configure_worker_cpu
    configure_worker_cpu('camera_reference')
    try:
        print(json.dumps(reference(Path(sys.argv[sys.argv.index('--project')+1]))),flush=True)
        return 0
    except Exception as error:
        print('Saved original camera could not be read: '+str(error),file=sys.stderr,flush=True)
        return 1
