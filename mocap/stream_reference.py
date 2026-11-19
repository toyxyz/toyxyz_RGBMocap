"""Reference-only, skinned FBX matching the Unreal Live Link rig exactly."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def validate(schema):
    if not isinstance(schema, dict):
        raise ValueError('Invalid reference rig.')
    names, parents, rest, lengths = (schema[key] for key in ('names','parents','rest','lengths'))
    n = len(names)
    if (not 1 <= n <= 256 or any(not isinstance(name,str) or not name
            or len(name.encode('utf-16-le'))>240 or any(c in name for c in ':\\/\x00\x01') for name in names)
            or len({name.casefold() for name in names}) != n):
        raise ValueError('Invalid reference bone names.')
    if (len(parents)!=n or any(type(p) is not int for p in parents) or parents[0]!=-1
            or any(not 0<=p<i for i,p in enumerate(parents[1:],1))):
        raise ValueError('Invalid reference hierarchy.')
    rest, lengths = np.asarray(rest,dtype=float), np.asarray(lengths,dtype=float)
    if (rest.shape!=(n,7) or lengths.shape!=(n,) or not np.isfinite(rest).all()
            or not np.isfinite(lengths).all() or np.any(lengths<=0)
            or not np.allclose(np.linalg.norm(rest[:,3:],axis=1),1.,atol=1e-5)):
        raise ValueError('Invalid reference transforms/lengths.')
    return names, parents, rest, lengths


def write(path, schema):
    """Private intermediate for Blender; convert Unreal LH to FBX RH by reflecting Y."""
    from scipy.spatial.transform import Rotation
    from .fbx_writer import node, prop, properties, _element

    names, parents, rest, lengths = validate(schema)
    n=len(names)
    flip=np.diag((1.,-1.,1.))
    rotations=flip @ Rotation.from_quat(rest[:,3:]).as_matrix() @ flip
    translations=(flip @ rest[:,:3,None])[...,0]
    angles=Rotation.from_matrix(rotations).as_euler('xyz',degrees=True)
    local=np.tile(np.eye(4),(n,1,1))
    local[:,:3,:3]=rotations;local[:,:3,3]=translations
    world=local.copy()
    for i,p in enumerate(parents):
        if p>=0:world[i]=world[p]@local[i]
    ids=np.arange(1000,1000+n)
    objects=[];connections=[]
    def link(a,b):connections.append(node('C','OO',int(a),int(b)))
    for i,p in enumerate(parents):
        objects.append(node('Model',int(ids[i]),names[i]+'\0\1Model','LimbNode',children=[
            node('Version',232),properties(
                prop('Lcl Translation','Lcl Translation',*map(float,translations[i]),flags='A'),
                prop('Lcl Rotation','Lcl Rotation',*map(float,angles[i]),flags='A'),
                prop('Lcl Scaling','Lcl Scaling',1.,1.,1.,flags='A'),
                prop('RotationActive','bool',True),prop('RotationOrder','enum',0),
                prop('InheritType','enum',1))]))
        objects.append(node('NodeAttribute',2000+i,names[i]+'\0\1NodeAttribute','LimbNode',children=[
            node('TypeFlags','Skeleton'),properties(prop('Size','double',max(.2,lengths[i]*.05)))]))
        link(2000+i,ids[i]);link(ids[i],0 if p<0 else ids[p])

    # A small octahedron for every joint ensures all bones are used by the mesh.
    # Child-centred directions are display geometry only; joint axes stay untouched.
    verts=[];polygons=[];normals=[]
    faces=((0,2,3),(0,3,4),(0,4,5),(0,5,2),(1,3,2),(1,4,3),(1,5,4),(1,2,5))
    for i in range(n):
        head=world[i,:3,3]
        children=np.flatnonzero(np.asarray(parents)==i)
        direction=(world[children,:3,3].mean(0)-head if len(children)
                   else world[i,:3,2]*lengths[i])
        if np.linalg.norm(direction)<1e-7:direction=world[i,:3,2]*lengths[i]
        size=np.linalg.norm(direction)
        unit=direction/size
        seed=np.eye(3)[np.argmin(np.abs(unit))]
        side=np.cross(unit,seed);side/=np.linalg.norm(side)
        other=np.cross(unit,side)
        radius=max(.04,min(size*.08,1.5))
        ring=head+direction*.2
        points=np.array((head,head+direction,ring+side*radius,ring+other*radius,
                         ring-side*radius,ring-other*radius))
        offset=len(verts);verts.extend(points)
        for face in faces:
            a,b,c=points[list(face)]
            normal=np.cross(b-a,c-a);normal/=np.linalg.norm(normal)
            normals.extend((normal,normal,normal))
            polygons.extend((offset+face[0],offset+face[1],-(offset+face[2]+1)))
    objects.append(node('Geometry',3000,'ReferenceBones\0\1Geometry','Mesh',children=[
        node('GeometryVersion',124),
        node('Vertices',np.asarray(verts,dtype=np.float64).reshape(-1)),
        node('PolygonVertexIndex',np.asarray(polygons,dtype=np.int32)),
        node('LayerElementNormal',0,children=[node('Version',101),node('Name',''),
             node('MappingInformationType','ByPolygonVertex'),node('ReferenceInformationType','Direct'),
             node('Normals',np.asarray(normals,dtype=np.float64).reshape(-1))]),
        node('Layer',0,children=[node('Version',100),node('LayerElement',children=[
             node('Type','LayerElementNormal'),node('TypedIndex',0)])])]))
    objects.append(node('Model',3001,'ReferenceBones\0\1Model','Mesh',children=[
        node('Version',232),properties(prop('Lcl Translation','Lcl Translation',0.,0.,0.),
        prop('Lcl Rotation','Lcl Rotation',0.,0.,0.),prop('Lcl Scaling','Lcl Scaling',1.,1.,1.)),
        node('Shading',True),node('Culling','CullingOff')]))
    link(3000,3001);link(3001,0)
    objects.append(node('Deformer',4000,'ReferenceSkin\0\1Deformer','Skin',children=[
        node('Version',101),node('Link_DeformAcuracy',50.)]))
    link(4000,3000)
    for i in range(n):
        objects.append(node('Deformer',4100+i,names[i]+'\0\1SubDeformer','Cluster',children=[
            node('Version',100),node('UserData','',''),
            node('Indexes',np.arange(i*6,i*6+6,dtype=np.int32)),
            node('Weights',np.ones(6,dtype=np.float64)),
            node('Transform',np.eye(4).T.reshape(-1)),
            node('TransformLink',world[i].T.reshape(-1))]))
        link(4100+i,4000);link(ids[i],4100+i)
    pose=[node('PoseNode',children=[node('Node',3001),node('Matrix',np.eye(4).T.reshape(-1))])]
    pose.extend(node('PoseNode',children=[node('Node',int(ids[i])),node('Matrix',world[i].T.reshape(-1))])
                for i in range(n))
    objects.append(node('Pose',5000,'BindPose\0\1Pose','BindPose',children=[
        node('Type','BindPose'),node('Version',100),node('NbPoseNodes',n+1),*pose]))
    import struct
    settings=properties(*(prop(k,'int',v) for k,v in (
        ('UpAxis',2),('UpAxisSign',1),('FrontAxis',1),('FrontAxisSign',-1),('CoordAxis',0),('CoordAxisSign',1))),
        prop('UnitScaleFactor','double',1.),prop('OriginalUnitScaleFactor','double',1.))
    from collections import Counter
    counts=Counter(o[0] for o in objects)
    definitions=[node('Version',100),node('Count',len(objects))]
    definitions.extend(node('ObjectType',kind,children=[node('Count',count)]) for kind,count in counts.items())
    with Path(path).open('xb') as out:
        out.write(b'Kaydara FBX Binary  \0\x1a\0'+struct.pack('<I',7400))
        for item in (
            node('FBXHeaderExtension',children=[node('FBXHeaderVersion',1003),node('FBXVersion',7400),node('Creator','RGB MoCap Live Link Reference')]),
            node('GlobalSettings',children=[node('Version',1000),settings]),
            node('Documents',children=[node('Count',1),node('Document',800,'Scene','Scene',children=[properties(),node('RootNode',0)])]),
            node('References'),node('Definitions',children=definitions),
            node('Objects',children=objects),node('Connections',children=connections)):
            _element(out,item)
        out.write(b'\0'*13)
        out.write(bytes.fromhex('fabcab09d0c8d466b176fb831cf7267e'))
        out.write(b'\0'*((-out.tell())%16));out.write(struct.pack('<I',7400));out.write(b'\0'*120)
        out.write(bytes.fromhex('f85a8c6adef5d97eece90ce3758f290b'))


def run(request,emit=lambda *args,**kwargs:None):
    from .blend_support import partial,build,cleanup
    request=Path(request)
    if request.stat().st_size>1024**2:raise ValueError('Reference request is too large.')
    data=json.loads(request.read_text(encoding='utf-8'))
    target=Path(data['target'])
    if target.suffix.lower()!='.fbx' or not target.parent.is_dir():
        raise ValueError('Choose an FBX file in an existing folder.')
    temp=partial(target,request)
    temp.mkdir()
    marker=temp/'.rgbmocap-owner'
    try:
        marker.write_text(request.stem,encoding='utf-8')
    except OSError:
        marker.unlink(missing_ok=True)
        temp.rmdir()
        raise
    try:
        write(temp/'source.fbx',data['schema'])
        (temp/'rig.json').write_bytes(json.dumps(data['schema'],allow_nan=False).encode('utf-8'))
        build(temp,'stream_reference_scene.py',emit,'stream_reference')
        output=temp/'reference.fbx'
        if not output.is_file():raise RuntimeError('Blender did not write the reference FBX.')
        os.replace(output,target)
    finally:
        cleanup(temp,request)
        request.unlink(missing_ok=True)
    return target


def main(emit):
    import argparse,traceback
    from .compute_policy import configure_worker_cpu
    configure_worker_cpu('stream_reference')
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[2:])
    start=time.monotonic()
    try:
        target=run(args.output,emit)
    except Exception as error:
        traceback.print_exc()
        emit('failed',stage='stream_reference',message=str(error))
        return 1
    emit('complete',stage='stream_reference',files=1,elapsed=time.monotonic()-start,output=str(target))
    return 0
