"""Original octahedral display geometry for the committed BVH hierarchy.

Blender's octahedral / joint / outline passes were studied as a visual reference.
No Blender code or assets are embedded. Geometry never edits motion matrices.
"""
import numpy as np


def reference_frame(delta):
    """Zero-roll +Y bone frame in the BVH's own coordinate system.

    Blender creates BVH edit bones before applying the import axis conversion.
    A shortest-arc rotation from +Y reproduces that reference frame; constructing
    a frame from the animated direction would discard the committed bone roll.
    """
    direction=np.asarray(delta,dtype=np.float64)/np.linalg.norm(delta)
    x,y,z=direction;transverse=np.hypot(x,z)
    # Match the stable, symmetric reference used at the -Y singularity.
    if transverse<=2.5e-4 and y<0:
        return np.diag([-1.,-1.,1.])
    if transverse<1e-12:
        return np.diag([-1.,-1.,1.]) if y<0 else np.eye(3)
    axis=np.array([z,0.,-x])/transverse
    angle=np.arctan2(transverse,y);s=np.sin(angle);c=np.cos(angle)
    cross=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    return c*np.eye(3)+(1-c)*np.outer(axis,axis)+s*cross


def preview_current(project,kind,loaded):
    """Small receipts only; reject a replaced generation before displaying it."""
    import json
    from .mixamo_data import current
    if not loaded or not current(project,kind):return False
    try:
        now=json.loads((project/(kind+'_mixamo.json')).read_text(encoding='utf-8'))
        return all(loaded.get(key)==now.get(key) for key in ('frames_file','joints_file','asset_file','outputs'))
    except (OSError,ValueError):return False


def geometry(rest,parents,ends,key,*,count=52,rest_axes=None):
    rest=np.asarray(rest,dtype=np.float64);parents=np.asarray(parents);ends=np.asarray(ends,dtype=np.float64)
    if rest.shape!=(count,3) or parents.shape!=(count,) or ends.shape!=(count,3):
        raise ValueError(f'Export skeleton metadata must contain {count} joints')
    if not np.issubdtype(parents.dtype,np.integer) or not np.isfinite(rest).all() or not np.isfinite(ends).all() or parents[0]!=-1:
        raise ValueError('Invalid export skeleton metadata')
    if any(not 0<=parents[j]<j for j in range(1,count)):raise ValueError('Invalid export hierarchy')
    vertices=[];faces=[];edges=[];segments=[]
    def append(points,triangles,bone,wire=(),smooth_center=None):
        if rest_axes is not None:
            points=points@rest_axes[bone]
            if smooth_center is not None:smooth_center=smooth_center@rest_axes[bone]
        start=len(vertices)
        triangles=np.asarray(triangles,dtype=np.int32);p=points[triangles]
        if smooth_center is None:
            normals=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
            normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-15)
            normals=np.repeat(normals[:,None],3,axis=1)
        else:
            normals=p-smooth_center
            normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-15)
        count=len(triangles)*3
        vertices.extend(np.column_stack((p.reshape(-1,3),normals.reshape(-1,3),np.full(count,bone))).tolist())
        faces.extend((np.arange(count).reshape(-1,3)+start).tolist())
        # Separate edge vertices avoid drawing triangulation diagonals on joints.
        if wire:
            offset=len(vertices);p=points[np.asarray(wire)].reshape(-1,3)
            vertices.extend(np.column_stack((p,np.tile([0.,1.,0.],(len(p),1)),np.full(len(p),bone))).tolist())
            edges.extend((np.arange(len(p)).reshape(-1,2)+offset).tolist())
        return start
    def sphere(center,radius,bone):
        # Small bounded joint caps, eight longitudes and four latitude bands.
        points=[center+radius*np.array([0,1,0]),center+radius*np.array([0,-1,0])]
        for lat in (np.pi/4,np.pi/2,3*np.pi/4):
            for angle in np.arange(8)*np.pi/4:
                points.append(center+radius*np.array([np.sin(lat)*np.cos(angle),np.cos(lat),np.sin(lat)*np.sin(angle)]))
        tris=[]
        for i in range(8):
            j=(i+1)%8;tris.extend([(0,2+i,2+j),(1,18+j,18+i)])
            for ring in (2,10):tris.extend([(ring+i,ring+8+i,ring+8+j),(ring+i,ring+8+j,ring+j)])
        append(np.asarray(points),[tuple(reversed(t)) for t in tris],bone,smooth_center=center)
    for bone in range(count):
        children=np.flatnonzero(parents==bone)
        # One display bone per BVH joint, including palms and branching trunks.
        # Blender's importer places a branch tail at the mean child rest head.
        delta=rest[children].mean(axis=0)-rest[bone] if len(children) else ends[bone].copy()
        if np.linalg.norm(delta)<=.001:delta=delta+np.array([0.,.1,0.])
        child=int(children[0]) if len(children)==1 else -2 if len(children) else -1
        segments.append((bone,child,delta))
    # BVH import decides connection from local offsets (not world coincidence).
    # Reproduce its public-file precision and vector arithmetic for cap sharing.
    offsets=rest.copy();offsets[1:]-=rest[parents[1:]]
    offsets=np.round(offsets,9).astype(np.float32)
    local_tails=[]
    for bone in range(count):
        children=np.flatnonzero(parents==bone)
        outgoing=offsets[children].sum(axis=0)/np.float32(len(children)) if len(children) else np.round(ends[bone],9).astype(np.float32)
        tail=offsets[bone]+outgoing
        if np.linalg.norm(tail-offsets[bone])<=.001:tail=tail+np.array([0.,.1,0.],np.float32)
        local_tails.append(tail)
    endpoint_vertices=[];octahedra=[];joint_caps=[]
    for bone,child,delta in segments:
        length=np.linalg.norm(delta)
        frame=reference_frame(delta)
        # Octahedral shoulders at 10% length, square +/-10% in local X/Z.
        # Keep this constant display reference separate from anatomical axes.
        display_tail=length*frame[:,1]
        ring=[display_tail*.1+length*.1*(sx*frame[:,0]+sz*frame[:,2])
              for sx,sz in ((1,1),(1,-1),(-1,-1),(-1,1))]
        points=np.asarray([np.zeros(3),display_tail,*ring]);tri=[];wire=[]
        for i in range(4):
            a=2+i;b=2+(i+1)%4;tri.extend([(0,b,a),(1,a,b)]);wire.extend([(0,a),(1,a),(a,b)])
        offset=append(points,tri,bone,wire)
        endpoint_vertices.append((bone,child,offset,offset+3))
        octahedra.append(points)
        # Connected heads use the parent's tail cap, as in Blender. Branches
        # remain visibly disconnected instead of inventing extra fan bones.
        parent=int(parents[bone])
        connected=parent>=0 and np.array_equal(local_tails[parent],offsets[bone])
        for center in ([display_tail] if connected else [np.zeros(3),display_tail]):
            sphere(center,length*.05,bone);joint_caps.append((bone,center.copy(),length*.05))
    packed=np.asarray(vertices,np.float32)
    if rest_axes is not None:segments=[(bone,child,delta@rest_axes[bone]) for bone,child,delta in segments]
    return dict(packed=packed,faces=np.asarray(faces,np.uint32),edges=np.asarray(edges,np.uint32),bone_count=count,
        vertex_count=len(packed),mode='rig',key=('bone-overlay-v3',*key),segments=segments,
        endpoint_vertices=endpoint_vertices,octahedra=np.asarray(octahedra),joint_caps=joint_caps)


def open_preview(project,kind,cancelled=lambda:False):
    """Read tiny skeleton metadata only on the existing background loader."""
    from .mixamo_data import open_preview as load
    result=load(project,kind,cancelled)
    if result:
        arrays,asset,info=result
        try:
            with np.load(project/info['asset_file'],allow_pickle=False) as source:
                asset['bones']=geometry(source['rest'],source['parents'],source['ends'],asset['key'])
                asset['parents']=source['parents'].copy()
                from .bone_stream import StreamRig
                try:
                    asset['stream_rig']=StreamRig.mixamo(info['names'],source)
                except (OSError,ValueError,KeyError,TypeError):
                    # Old preview assets may lack the full export bind.
                    pass
        except (OSError,ValueError,KeyError) as error:
            asset['bone_warning']='Export bone overlay unavailable: '+str(error)
    return result
