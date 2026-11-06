"""Optional transforms of exported copies, never of committed source motion."""
import numpy as np


def symmetric_rest(names, parents, rest, ends):
    """Mirror averaged paired segments; place the central chain vertically."""
    parents=np.asarray(parents);rest=np.asarray(rest,float);ends=np.asarray(ends,float)
    if len(names)!=52 or len(set(names))!=52 or rest.shape!=(52,3) or ends.shape!=(52,3) or parents.shape!=(52,):
        raise ValueError('Expected the complete anatomical 52-bone skeleton')
    if names[0]!='Hips' or parents[0]!=-1 or any(not 0<=parents[j]<j for j in range(1,52)):
        raise ValueError('Invalid export hierarchy')
    if not np.isfinite(rest).all() or not np.isfinite(ends).all():raise ValueError('Non-finite rest pose')
    offsets=rest.copy();offsets[1:]-=rest[parents[1:]]
    result=np.zeros_like(rest);tips=ends.copy();mirror=np.array([-1.,1.,1.])
    central={'Hips','Spine','Spine1','Spine2','Neck','Head'}
    lookup={name:j for j,name in enumerate(names)}
    for j,name in enumerate(names):
        if name in central:
            offsets[j]=[0.,float(np.linalg.norm(offsets[j])),0.]
            tips[j]=[0.,float(np.linalg.norm(ends[j])),0.]
        elif name.startswith('Left'):
            other='Right'+name[4:]
            if other not in lookup:raise ValueError('Missing paired bone: '+other)
            k=lookup[other]
            expected=names[parents[j]]
            expected='Right'+expected[4:] if expected.startswith('Left') else expected
            if names[parents[k]]!=expected:raise ValueError('Mismatched paired hierarchy')
            for values in (offsets,tips):
                a=values[j].copy();b=values[k]*mirror
                length=(np.linalg.norm(a)+np.linalg.norm(b))*.5
                direction=a+b;norm=np.linalg.norm(direction)
                if length>1e-10 and norm<1e-10:raise ValueError('Opposed paired rest directions')
                values[j]=direction*(length/norm) if norm>1e-10 else 0.
                values[k]=values[j]*mirror
        elif name.startswith('Right'):
            if 'Left'+name[5:] not in lookup:raise ValueError('Missing paired bone: '+name)
        else:raise ValueError('Unsupported anatomical bone: '+name)
    for j,p in enumerate(parents):result[j]=offsets[j] if p<0 else result[p]+offsets[j]
    floor=min(result[lookup['LeftToeBase'],1],result[lookup['RightToeBase'],1])
    result[:,1]-=floor
    return result,tips


def centered_transform(transform, first_root, origins, group):
    """Keep one XZ offset per actor across its ordered visible takes."""
    matrix=np.asarray(transform,float).copy()
    root=matrix[:3,:3]@np.asarray(first_root,float)+matrix[:3,3]
    if not np.isfinite(root).all():raise ValueError('Non-finite initial root')
    if group not in origins:origins[group]=np.array([root[0],0.,root[2]])
    matrix[:3,3]-=origins[group]
    return matrix


def read_header(inp, symmetric=False):
    """Read a bounded, strict source BVH header and optionally replace offsets."""
    lines=[];names=[];parents=[];offsets=[];ends=[];offset_lines={};end_lines={}
    stack=[];pending=None;channels=0;root_order=None;orders=[];header_bytes=0
    for _ in range(4096):
        line=inp.readline(65537)
        if not line or len(line)>65536:raise ValueError('Invalid BVH hierarchy')
        header_bytes+=len(line)
        if header_bytes>2*1024**2:raise ValueError('BVH hierarchy is too large')
        index=len(lines);lines.append(line);tokens=line.split()
        if tokens[:1] in ([b'ROOT'],[b'JOINT']):
            if len(tokens)!=2:raise ValueError('Invalid BVH joint')
            parent=stack[-1] if stack else -1
            if parent is None or (tokens[0]==b'ROOT')!=(not names):raise ValueError('Invalid BVH hierarchy root')
            pending=len(names);names.append(tokens[1].decode('ascii'));parents.append(parent)
            if len(names)>52:raise ValueError('Expected 52 anatomical bones')
            offsets.append(None);ends.append(np.zeros(3))
        elif tokens==[b'End',b'Site']:pending=None
        elif tokens==[b'{']:stack.append(pending)
        elif tokens==[b'}']:
            if not stack:raise ValueError('Invalid BVH braces')
            stack.pop()
        elif tokens[:1]==[b'OFFSET']:
            value=np.asarray(tokens[1:],float)
            if value.shape!=(3,) or not np.isfinite(value).all() or not stack:raise ValueError('Invalid BVH offset')
            if stack[-1] is None:
                joint=stack[-2];ends[joint]=value;end_lines[joint]=index
            else:
                joint=stack[-1];offsets[joint]=value;offset_lines[joint]=index
        elif tokens[:1]==[b'CHANNELS']:
            if not stack or stack[-1] is None:raise ValueError('Invalid BVH channels')
            joint=stack[-1];count=int(tokens[1]);expected=6 if joint==0 else 3
            if count!=expected or len(tokens)!=count+2:raise ValueError('Invalid BVH channel count')
            if joint==0 and tokens[2:5]!=[b'Xposition',b'Yposition',b'Zposition']:raise ValueError('Invalid root position channels')
            rotation=tokens[5:] if joint==0 else tokens[2:]
            if sorted(rotation)!=[b'Xrotation',b'Yrotation',b'Zrotation']:raise ValueError('Invalid BVH rotation channels')
            order=''.join(v[:1].decode('ascii') for v in rotation);orders.append(order);channels+=count
            if joint==0:root_order=order
        if line.strip()==b'MOTION':break
    else:raise ValueError('BVH hierarchy is too large')
    if stack or len(names)!=52 or channels!=159 or len(orders)!=52 or any(v is None for v in offsets):raise ValueError('Expected the complete 52-bone BVH')
    if symmetric:
        rest=np.asarray(offsets,float);parents=np.asarray(parents)
        for j,p in enumerate(parents):
            if p>=0:rest[j]+=rest[p]
        rest,tips=symmetric_rest(names,parents,rest,np.asarray(ends))
        for j,index in offset_lines.items():
            value=rest[j] if parents[j]<0 else rest[j]-rest[parents[j]]
            lines[index]=b'  OFFSET '+(' '.join(f'{v:.9f}' for v in value)).encode('ascii')+b'\n'
        for j,index in end_lines.items():lines[index]=b'    OFFSET '+(' '.join(f'{v:.9f}' for v in tips[j])).encode('ascii')+b'\n'
    return lines,root_order


def first_bvh_root(inp):
    position=inp.tell()
    try:
        read_header(inp)
        if not inp.readline(256).startswith(b'Frames:') or not inp.readline(256).startswith(b'Frame Time:'):raise ValueError('Invalid BVH timing')
        row=np.asarray(inp.readline(65537).split(),float)
        if row.shape!=(159,) or not np.isfinite(row).all():raise ValueError('Invalid initial BVH frame')
        return row[:3]
    finally:inp.seek(position)
