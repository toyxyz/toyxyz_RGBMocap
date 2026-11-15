"""Bind landmark positions to the locally registered body surface."""
import numpy as np

from .face_data import POINTS, POINT_INDEX, STABLE


def similarity(source, target):
    src, dst = source.mean(0), target.mean(0)
    a, b = source-src, target-dst
    left, singular, right = np.linalg.svd(a.T @ b)
    sign = np.array([1., 1., np.sign(np.linalg.det(left @ right))])
    scale = (singular*sign).sum()/max(float((a*a).sum()), 1e-8)
    matrix = scale*(left*sign)@right
    return matrix, dst-src@matrix


def read_canonical(path):
    with path.open(encoding='utf8') as stream:
        result = np.array([[float(v) for v in line.split()[1:4]]
                           for line in stream if line.startswith('v ')], dtype=np.float32)
    if result.shape != (468, 3):
        raise ValueError('Invalid MediaPipe canonical face.')
    return result


def surface_binding(info, canonical):
    from .models import BY_KEY
    model = info.get('pipeline', {}).get('model')
    if model == 'gemx':
        name = 'soma'
        with np.load(BY_KEY['gemx_soma_neutral'].path, allow_pickle=False) as z:
            vertices, faces = z['mean']/100., z['triangles']
    elif model == 'flowhmr':
        name = 'smplh'
        with np.load(BY_KEY['smplh'].path, allow_pickle=False) as z:
            vertices, faces = z['v_template'], z['f']
    else:
        name = 'smplx'
        with np.load(BY_KEY['smplx'].path, allow_pickle=False) as z:
            vertices, faces = z['v_template'], z['f']
            landmark_faces = z['lmk_faces_idx']
            bary = z['lmk_bary_coords']
    height = np.ptp(vertices[:,1])
    head = vertices[:,1] >= vertices[:,1].max()-.155*height
    narrow = head & (np.abs(vertices[:,0])<.12*height)
    face_triangles = faces[np.all(narrow[faces], axis=1)]
    samples = vertices[narrow]
    width = np.ptp(samples[:,0])
    top = samples[:,1].max()
    central = samples[(np.abs(samples[:,0])<.18*width) &
                      (samples[:,1]>top-.105*height) & (samples[:,1]<top-.037*height)]
    nose = central[np.argmax(central[:,2])]
    scale = .87*width/np.ptp(canonical[:,0])
    coords = (canonical[POINTS,:2]-canonical[4,:2])*scale+nose[:2]
    if name == 'smplx' and len(landmark_faces)==51:
        # Native SMPL-X provides the 51 interior 68-point facial anchors.
        sparse_dlib = [27, 30, 31, 35, 36, 39, 42, 45, 48, 54, 51, 57]
        sparse_mp = [168, 4, 98, 327, 33, 133, 362, 263, 61, 291, 0, 17]
        tri = vertices[faces[landmark_faces[np.array(sparse_dlib)-17]]]
        target = np.einsum('pvc,pv->pc',tri,bary[np.array(sparse_dlib)-17])
        if np.ptp(target[:,0]) < .18 and np.ptp(target[:,1]) < .18:
            tangent, *_ = np.linalg.lstsq(np.c_[canonical[sparse_mp,:2], np.ones(len(sparse_mp))],
                                           target[:,:2],rcond=None)
            coords = np.c_[canonical[POINTS,:2],np.ones(len(POINTS))]@tangent
    return name, *ray_anchors(vertices, face_triangles, coords)


def ray_anchors(vertices, faces, xy):
    triangles = vertices[faces]
    origin, v1, v2 = triangles[:,0,:2], triangles[:,1,:2], triangles[:,2,:2]
    a, b = v1-origin, v2-origin
    denom = a[:,0]*b[:,1]-a[:,1]*b[:,0]
    valid = np.abs(denom)>1e-9
    faces, triangles = faces[valid], triangles[valid]
    origin, a, b, denom = origin[valid],a[valid],b[valid],denom[valid]
    indices, weights = [], []
    for point in xy:
        delta = point-origin
        u = (delta[:,0]*b[:,1]-delta[:,1]*b[:,0])/denom
        v = (a[:,0]*delta[:,1]-a[:,1]*delta[:,0])/denom
        bary = np.stack((1-u-v,u,v),axis=1)
        inside = np.min(bary,axis=1) >= -1e-5
        if inside.any():
            slot = np.argmax(np.where(inside,(bary*triangles[:,:,2]).sum(1),-np.inf))
            weight = bary[slot]
        else:
            safe = np.maximum(bary,0)
            safe /= safe.sum(1)[:,None]
            pos = np.einsum('pvc,pv->pc',triangles,safe)
            slot = np.argmin(((pos[:,:2]-point)**2).sum(1)-.005*(pos[:,2]-pos[:,2].max())*.001)
            weight = safe[slot]
        indices.append(faces[slot])
        weights.append(weight)
    return np.array(indices,dtype=np.int32),np.array(weights,dtype=np.float32)


def expression_offsets(raw, reference, canonical):
    idx = [POINT_INDEX[k] for k in STABLE]
    target = canonical[POINTS]
    span = np.linalg.norm(canonical[263]-canonical[33])
    conv = np.array((1.,-1.,-1.))
    ref = reference*conv
    matrix, bias = similarity(ref[idx],target[idx])
    base = ref@matrix+bias
    output = []
    for sample in raw:
        points = sample*conv
        matrix, bias = similarity(points[idx],target[idx])
        delta = (points@matrix+bias-base)[:,:2]/span
        output.append(np.clip(delta,-.24,.24))
    return np.asarray(output,dtype=np.float32)


class FaceSurface:
    def __init__(self, archive, asset, native):
        self.bary = archive.weights
        self.used, self.lookup = np.unique(archive.indices, return_inverse=True)
        self.lookup = self.lookup.reshape(archive.indices.shape)
        n = int(asset['vertex_count'])
        expected = {'smplx':10475,'smplh':6890,'soma':18056}.get(archive.binding_type)
        if n != expected or self.used.min()<0 or self.used.max()>=n:
            raise ValueError('Face overlay topology does not match this surface.')
        self.native = native
        if not native:
            packed = asset['packed']
            xyz = n*3
            self.template = packed[:xyz].reshape(n,3)[self.used]
            self.shape = packed[xyz:21*xyz].reshape(20,n,3)[:,self.used].reshape(20,-1).copy()
            self.pose = packed[21*xyz:507*xyz].reshape(486,n,3)[:,self.used].reshape(486,-1).copy()
            self.weights = packed[507*xyz:].reshape(n,55)[self.used]

    def points(self, frame, offset):
        if self.native:
            vertices = frame[self.used]
        else:
            rest = self.template+(frame[1366:1386]@self.shape+frame[880:1366]@self.pose).reshape(-1,3)
            t = (self.weights@frame[:880].reshape(55,16)).reshape(-1,4,4)
            vertices = np.einsum('kij,kj->ki',t[:,:3,:3],rest)+t[:,:3,3]+frame[-3:]
        anchors = np.einsum('kvc,kv->kc',vertices[self.lookup],self.bary)
        le,re,m1,m2,n = anchors[[POINT_INDEX[k] for k in (33,263,61,291,4)]]
        across = re-le
        width = max(np.linalg.norm(across),1e-7)
        right = across/width
        up = (le+re-m1-m2)*.5
        up -= (up@right)*right
        up /= max(np.linalg.norm(up),1e-7)
        front = np.cross(right,up)
        if (n-(le+re)*.5)@front < 0:
            front = -front
        points = anchors+width*(offset @ np.stack((right,up)))
        return points, front
