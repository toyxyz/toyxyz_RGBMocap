"""Bounded, non-pickle packets for the private live worker pipe."""
import io,json,struct
import numpy as np

LIMIT=16*1024**2

def send(stream,meta,**arrays):
    out=io.BytesIO()
    np.savez(out,**arrays)
    header=json.dumps(meta,ensure_ascii=False).encode('utf8')
    data=struct.pack('<I',len(header))+header+out.getvalue()
    if len(data)>LIMIT:raise ValueError('Live packet too large')
    stream.write(struct.pack('<I',len(data))+data);stream.flush()

def exact(stream,n):
    chunks=[]
    while n:
        part=stream.read(n)
        if not part:raise EOFError()
        chunks.append(part);n-=len(part)
    return b''.join(chunks)

def receive(stream):
    size=struct.unpack('<I',exact(stream,4))[0]
    if not 4<=size<=LIMIT:raise ValueError('Invalid live packet size')
    data=exact(stream,size);h=struct.unpack('<I',data[:4])[0]
    if h>65536 or h>size-4:raise ValueError('Invalid live header')
    meta=json.loads(data[4:4+h]);arrays={}
    import zipfile
    payload=io.BytesIO(data[4+h:])
    with zipfile.ZipFile(payload) as z:
        if len(z.infolist())>10 or sum(i.file_size for i in z.infolist())>LIMIT:raise ValueError('Invalid live arrays')
    payload.seek(0)
    with np.load(payload,allow_pickle=False) as z:
        for key in z.files:
            a=z[key]
            if a.dtype not in (np.dtype('uint8'),np.dtype('uint32'),np.dtype('int32'),np.dtype('float32')) or a.size>4_000_000:raise ValueError('Invalid live array')
            if a.dtype.kind=='f' and not np.isfinite(a).all():raise ValueError('Nonfinite live array')
            arrays[key]=a
    kind=meta.get('type') if isinstance(meta,dict) else None
    if kind not in ('status','topology','frame','missing','error','alignment','camera_preview'):raise ValueError('Unknown live packet')
    if kind=='topology':
        p=arrays['parents']
        if p.dtype!=np.int32 or p.shape!=(55,) or p[0]!=-1 or np.any(p[1:]<0) or np.any(p[1:]>=np.arange(1,55)):raise ValueError('Invalid PEAR hierarchy')
    if kind=='topology' and meta.get('render_mode')=='bones':
        if set(arrays)!={'parents'}:raise ValueError('Invalid bone topology payload')
    elif kind=='topology':
        n=int(meta['vertices']);f=arrays['faces'];o=arrays['offsets'];b=arrays['neighbors'];p=arrays['parents']
        if not 1<=n<=200000 or f.ndim!=1 or len(f)%3 or o.shape!=(n+1,) or b.ndim!=1 or p.ndim!=1 or len(p)>512:raise ValueError('Invalid live topology')
        if f.dtype!=np.uint32 or o.dtype!=np.uint32 or b.dtype!=np.uint32 or p.dtype!=np.int32:raise ValueError('Invalid topology type')
        if p.shape!=(55,) or p[0]!=-1 or np.any(p[1:]<0) or np.any(p[1:]>=np.arange(1,55)):raise ValueError('Invalid PEAR hierarchy')
        if not len(f) or f.max()>=n or (len(b) and b.max()>=n) or o[0]!=0 or int(o[-1])*2!=len(b) or np.any(o[1:]<o[:-1]):raise ValueError('Invalid topology indices')
        for key,shape in (('neutral_joints',(55,3)),('neutral_surface',(8,4,3))):
            a=arrays[key]
            if a.dtype!=np.float32 or a.shape!=shape:raise ValueError('Invalid live neutral calibration')
    if kind=='frame':
        bones=meta.get('render_mode')=='bones'
        for k,minimum,limit in (('bounds' if bones else 'vertices',22,128 if bones else 200000),('joints',55,512)):
            a=arrays[k]
            if a.dtype!=np.float32 or a.ndim!=2 or a.shape[1]!=3 or not minimum<=len(a)<=limit:raise ValueError('Invalid live pose')
        if bones and (int(meta.get('pose_id',0))<1 or int(meta.get('bone_revision',0))<1):raise ValueError('Invalid fixed-rig generation')
        r=arrays['rotations']
        if r.dtype!=np.float32 or r.shape!=(55,3,3) or np.max(abs(r.swapaxes(-1,-2)@r-np.eye(3)))>.005 or np.linalg.det(r).min()<.99:
            raise ValueError('Invalid live source rotations')
        r=arrays['bone_facing']
        if r.dtype!=np.float32 or r.shape!=(52,3,3) or np.max(abs(r.swapaxes(-1,-2)@r-np.eye(3)))>.005 or np.linalg.det(r).min()<.99:
            raise ValueError('Invalid live bone facing')
    if 'thumbnails' in arrays:
        a=arrays['thumbnails']
        if a.dtype!=np.uint8 or a.ndim!=4 or not 1<=len(a)<=32 or a.shape[1:]!=(72,128,3):raise ValueError('Invalid camera thumbnails')
    if 'image' in arrays:
        a=arrays['image']
        if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 or not 0<a.shape[0]<=720 or not 0<a.shape[1]<=1280:raise ValueError('Invalid camera image')
    return meta,arrays

def topology(faces,count):
    f=np.asarray(faces,dtype=np.uint32).reshape(-1,3)
    centers=f.reshape(-1)
    pairs=f[:,[[1,2],[2,0],[0,1]]].reshape(-1,2)
    order=np.argsort(centers,kind='stable')
    offsets=np.r_[0,np.cumsum(np.bincount(centers,minlength=count))].astype('uint32')
    return dict(faces=f.reshape(-1),offsets=offsets,neighbors=pairs[order].reshape(-1))
