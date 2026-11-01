"""Small original binary FBX 7.4 skeleton/animation writer.

No SDK, Blender runtime, third-party implementation or body data is embedded.
Node defaults carry the registered rest pose; PreRotation carries its axes.
Only Hips has animated translation. End markers preserve terminal lengths.
"""
import struct
import warnings
import zlib
import numpy as np
from scipy.spatial.transform import Rotation
from .source_bvh import continuous_angles

TICKS = 46186158000


def node(name, *props, children=()):
    return name, props, children


def prop(name, kind, *values, flags=''):
    subtype={'int':'Integer','double':'Number'}.get(kind,'')
    if kind in ('int','enum','bool'):values=tuple(np.int32(v) for v in values)
    return node('P', name, kind, subtype, flags, *values)


def properties(*items):
    return node('Properties70', children=items)


def _property(value):
    if isinstance(value, str):
        raw=value.encode('utf-8'); return b'S'+struct.pack('<I',len(raw))+raw
    if isinstance(value, (bool, np.bool_)):
        return b'C'+struct.pack('<?',value)
    if isinstance(value,np.int32):
        return b'I'+struct.pack('<i',value)
    if isinstance(value, (int, np.integer)):
        return b'L'+struct.pack('<q',value)
    if isinstance(value, (float, np.floating)):
        return b'D'+struct.pack('<d',value)
    value=np.ascontiguousarray(value)
    codes={np.dtype('int64'):b'l',np.dtype('int32'):b'i',np.dtype('float64'):b'd',np.dtype('float32'):b'f'}
    if value.dtype not in codes:raise ValueError('Unsupported FBX array')
    raw=zlib.compress(value.tobytes(),1)
    return codes[value.dtype]+struct.pack('<III',value.size,1,len(raw))+raw


def _element(out, item):
    name, props, children=item; name=name.encode('ascii')
    start=out.tell();out.write(b'\0'*13+name)
    prop_start=out.tell()
    for value in props:out.write(_property(value))
    prop_size=out.tell()-prop_start
    for child in children:_element(out,child)
    if children:out.write(b'\0'*13)
    end=out.tell()
    if end>=2**32:raise ValueError('FBX exceeds the 4 GiB format limit; split the clip')
    out.seek(start);out.write(struct.pack('<IIIB',end,len(props),prop_size,len(name)));out.seek(end)


def _euler(matrix):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Rotation.from_matrix(matrix).as_euler('xyz',degrees=True)


def write(path, names, parents, rest, rest_frames, display, tip_local, frames, fps,
          transform=None, progress=lambda *args:None, rig_name='SourceAnimation', prepend_tpose=False):
    """Write one committed mapped take; working rotation arrays are one bone wide."""
    from .person_names import validate_name
    rig_name=validate_name(rig_name)
    n=len(frames);parents=np.asarray(parents);rest=np.asarray(rest,float)
    if n<1 or frames.shape!=(n,52,4,4) or not np.isfinite(fps) or fps<=0:
        raise ValueError('Invalid FBX frame dimensions/timing')
    if len(names)!=52 or len(set(names))!=52 or names[0]!='Hips' or parents.shape!=(52,) or parents[0]!=-1:
        raise ValueError('Expected plain 52-bone Hips hierarchy')
    if any(not name.isascii() or any(c in name for c in ':\0\1') for name in names):
        raise ValueError('Invalid FBX bone name')
    if rest.shape!=(52,3) or np.shape(rest_frames)!=(52,3,3) or np.shape(display)!=(52,3,3) or np.shape(tip_local)!=(52,3):
        raise ValueError('Invalid FBX rest dimensions')
    if any(not 0<=parents[j]<j for j in range(1,52)):
        raise ValueError('Invalid FBX parents')
    for a in (rest,rest_frames,display,tip_local):
        if not np.isfinite(a).all():raise ValueError('Invalid FBX rest data')
    for a in (rest_frames,display):
        if np.max(abs(a.swapaxes(-1,-2)@a-np.eye(3)))>1e-5 or np.linalg.det(a).min()<.99999:
            raise ValueError('Invalid FBX rest rotation')
    from .scene_camera import validate_transform
    transform=validate_transform(np.eye(4) if transform is None else transform)
    ids=np.arange(1000,1052);objects=[];connections=[]
    link=lambda a,b,p=None:connections.append(node('C','OO' if p is None else 'OP',int(a),int(b),*(() if p is None else (p,))))
    # Empty parent is an FBX armature container, not an extra animated bone.
    objects.append(node('Model',900,rig_name+'\0\1Model','Null',children=[node('Version',232),properties()]))
    link(900,0)
    local_rest=[];offsets=[]
    for j,p in enumerate(parents):
        R=rest_frames[j] if p<0 else rest_frames[p].T@rest_frames[j]
        t=rest[j] if p<0 else rest_frames[p].T@(rest[j]-rest[p])
        local_rest.append(R);offsets.append(t)
        children=[node('Version',232),properties(
            prop('Lcl Translation','Lcl Translation',*map(float,t),flags='A'),
            prop('Lcl Rotation','Lcl Rotation',0.,0.,0.,flags='A'),
            prop('Lcl Scaling','Lcl Scaling',1.,1.,1.,flags='A'),
            prop('PreRotation','Vector3D',*map(float,_euler(R))),
            prop('RotationActive','bool',True),prop('RotationOrder','enum',0),
            prop('InheritType','enum',1))]
        objects.append(node('Model',int(ids[j]),names[j]+'\0\1Model','LimbNode',children=children))
        objects.append(node('NodeAttribute',2000+j,names[j]+'\0\1NodeAttribute','LimbNode',children=[node('TypeFlags','Skeleton')]))
        link(2000+j,ids[j]);link(ids[j],900 if p<0 else ids[p])
    for j in range(52):
        if j in parents:continue
        marker=3000+j
        objects.append(node('Model',marker,names[j]+'_end\0\1Model','LimbNode',children=[
            node('Version',232),properties(prop('Lcl Translation','Lcl Translation',*map(float,tip_local[j]),flags='A'))]))
        objects.append(node('NodeAttribute',4000+j,names[j]+'_end\0\1NodeAttribute','LimbNode',children=[node('TypeFlags','Skeleton')]))
        link(4000+j,marker);link(marker,ids[j])
    # Store an explicit bind-pose table in addition to the node defaults.
    pose_nodes=[node('PoseNode',children=[node('Node',900),node('Matrix',np.eye(4).T.reshape(-1))])]
    for j in range(52):
        M=np.eye(4);M[:3,:3]=rest_frames[j];M[:3,3]=rest[j]
        pose_nodes.append(node('PoseNode',children=[node('Node',int(ids[j])),node('Matrix',M.T.reshape(-1))]))
    objects.append(node('Pose',5000,'BindPose\0\1Pose','BindPose',children=[node('Type','BindPose'),node('Version',100),node('NbPoseNodes',len(pose_nodes)),*pose_nodes]))
    # Zero-based file times import at frame 1 with Blender's default offset 1.
    key_count=n+int(bool(prepend_tpose))
    times=np.rint(np.arange(key_count)*TICKS/fps).astype(np.int64)
    stop=int(times[-1]);stack=6000;layer=6001
    objects.append(node('AnimationStack',stack,'Motion\0\1AnimStack','',children=[properties(
        prop('LocalStart','KTime',0),prop('LocalStop','KTime',stop),prop('ReferenceStart','KTime',0),prop('ReferenceStop','KTime',stop))]))
    objects.append(node('AnimationLayer',layer,'BaseLayer\0\1AnimLayer','',children=[properties()]))
    link(layer,stack)
    settings=properties(*(prop(k,'int',v) for k,v in [('UpAxis',1),('UpAxisSign',1),('FrontAxis',2),('FrontAxisSign',1),('CoordAxis',0),('CoordAxisSign',1)]),
        prop('UnitScaleFactor','double',100.),prop('OriginalUnitScaleFactor','double',100.),
        prop('TimeMode','enum',14),prop('CustomFrameRate','double',float(fps)),
        prop('TimeSpanStart','KTime',0),prop('TimeSpanStop','KTime',stop))
    # Stream nodes/curves individually so serialized animation never accumulates.
    with open(path,'xb') as out:
        out.write(b'Kaydara FBX Binary  \0\x1a\0'+struct.pack('<I',7400))
        _element(out,node('FBXHeaderExtension',children=[node('FBXHeaderVersion',1003),node('FBXVersion',7400),node('Creator','RGB MoCap')]))
        _element(out,node('GlobalSettings',children=[node('Version',1000),settings]))
        _element(out,node('Documents',children=[node('Count',1),node('Document',800,'Scene','Scene',children=[properties(),node('RootNode',0)])]))
        _element(out,node('References'))
        _element(out,node('Definitions',children=[node('Version',100),node('Count',len(objects))]))
        start=out.tell();out.write(b'\0'*13+b'Objects')
        for item in objects:_element(out,item)
        nextid=10000
        def curves(j,kind,values):
            nonlocal nextid
            cid=nextid;nextid+=1
            _element(out,node('AnimationCurveNode',cid,('T' if kind.endswith('Translation') else 'R')+'\0\1AnimCurveNode','',children=[properties(*(prop('d|'+a,'Number',0.) for a in 'XYZ'))]))
            link(cid,layer);link(cid,ids[j],kind)
            for axis,a in enumerate('XYZ'):
                uid=nextid;nextid+=1
                _element(out,node('AnimationCurve',uid,'\0\1AnimCurve','',children=[node('Default',float(values[0,axis])),node('KeyVer',4008),
                    node('KeyTime',times),node('KeyValueFloat',values[:,axis].astype(np.float32)),
                    node('KeyAttrFlags',np.array([4],np.int32)),node('KeyAttrDataFloat',np.zeros(4,np.float32)),node('KeyAttrRefCount',np.array([key_count],np.int32))]))
                link(uid,cid,'d|'+a)
        for j,p in enumerate(parents):
            world=np.asarray(frames[:,j,:3,:3],float)@display[j]
            if p<0:
                local=transform[:3,:3]@world
                positions=np.asarray(frames[:,0,:3,3],float)@transform[:3,:3].T+transform[:3,3]
                if not np.isfinite(positions).all():raise ValueError('Non-finite FBX root')
                if prepend_tpose:positions=np.concatenate((positions[:1],positions))
                curves(j,'Lcl Translation',positions)
            else:
                parent=np.asarray(frames[:,p,:3,:3],float)@display[p]
                local=parent.swapaxes(-1,-2)@world
                actual=np.einsum('tji,tj->ti',parent,np.asarray(frames[:,j,:3,3],float)-frames[:,p,:3,3])
                if not np.isfinite(actual).all() or np.max(abs(actual-offsets[j]))>2e-5:
                    raise ValueError('Non-rigid bone offset in committed source animation')
            if not np.isfinite(local).all():raise ValueError('Non-finite FBX animation')
            if prepend_tpose:
                # Identity BVH world frames describe the straight-arm export bind.
                # Display axes transport that T-pose into the FBX bone basis.
                pose=transform[:3,:3]@display[j] if p<0 else display[p].T@display[j]
                local=np.concatenate((pose[None],local))
            angles=np.degrees(continuous_angles(np.radians(_euler(np.asarray(local_rest[j]).T@local))))
            curves(j,'Lcl Rotation',angles);progress(j+1,52)
        out.write(b'\0'*13);end=out.tell();out.seek(start);out.write(struct.pack('<IIIB',end,0,0,7));out.seek(end)
        _element(out,node('Connections',children=connections))
        _element(out,node('Takes',children=[node('Current','Motion')]))
        out.write(b'\0'*13)
        out.write(bytes.fromhex('fabcab09d0c8d466b176fb831cf7267e'))
        out.write(b'\0'*((-out.tell())%16));out.write(struct.pack('<I',7400));out.write(b'\0'*120)
        out.write(bytes.fromhex('f85a8c6adef5d97eece90ce3758f290b'))

