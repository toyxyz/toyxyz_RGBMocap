"""Small screen-space actor labels; no mesh readback or motion changes."""
import numpy as np
from PySide6.QtCore import QRectF


def character_samples(asset):
    """Prepare bounded skin samples off-thread for animated name placement only."""
    vertices=asset['vertices'];rows=asset['rows'];influences=asset['weights']
    if not len(vertices):return None
    indices=np.unique(np.r_[np.linspace(0,len(vertices)-1,min(512,len(vertices)),dtype=int),
                            vertices.argmin(axis=0),vertices.argmax(axis=0)])
    width=max(1,min(16,int(rows[indices,1].max())))
    bones=np.zeros((len(indices),width),np.intp)
    weights=np.zeros((len(indices),width),np.float32)
    for i,vertex in enumerate(indices):
        start,count=map(int,rows[vertex]);entries=influences[start:start+count]
        if len(entries)>width:entries=entries[np.argpartition(entries[:,1],-width)[-width:]]
        n=len(entries)
        if n:
            bones[i,:n]=entries[:,0].astype(np.intp);weights[i,:n]=entries[:,1]
            weights[i]/=max(float(weights[i].sum()),1e-8)
    return vertices[indices].copy(),bones,weights


def character_points(samples,frames,transform=None):
    """Skin at most 518 samples / 16 influences; never read back the GPU mesh."""
    if samples is None:return np.empty((0,3))
    vertices,bones,weights=samples
    matrices=frames[bones]
    points=(np.einsum('nkij,nj->nki',matrices[:,:,:3,:3],vertices)+matrices[:,:,:3,3])
    points=np.sum(points*weights[:,:,None],axis=1)
    if transform is not None:
        points=points@transform[:3,:3].T+transform[:3,3]
    return points


def body_points(joints, frames=None, asset=None, transform=None):
    """Use at most the committed 52 bone heads/tips, regardless of clip length."""
    points=np.asarray(joints)[:22]
    if frames is not None and asset is not None and 'bones' in asset:
        segments=asset['bones']['segments']
        tips=[frames[bone,:3,:3] @ delta + frames[bone,:3,3]
              for bone,_,delta in segments]
        points=np.concatenate((frames[:len(segments),:3,3],tips))
    if transform is not None:
        points=points @ transform[:3,:3].T + transform[:3,3]
    return points


def label_rect(xy,depth,focal,width,height,text_width,text_height):
    """Anchor above projected body bounds; badge dimensions are logical pixels."""
    valid=np.isfinite(xy).all(axis=1) & np.isfinite(depth) & (depth>.08)
    if not valid.any():return None
    xy=xy[valid];depth=depth[valid]
    # Account for skin outside joint centres without scanning full mesh vertices.
    pad=min(80.,focal*.11/float(np.median(depth)))
    left,top=xy.min(axis=0)-pad
    right,bottom=xy.max(axis=0)+pad
    if right<0 or left>width or bottom<0 or top>height:return None
    badge_width=min(text_width+20,max(0,width-16))
    badge_height=text_height+10
    x=np.clip((left+right-badge_width)/2,8,max(8,width-badge_width-8))
    y=np.clip(top-badge_height-10,8,max(8,height-badge_height-56))
    return QRectF(float(x),float(y),float(badge_width),float(badge_height))


def separate_labels(rect,placed,width):
    """Stack crowded labels upwards while keeping their screen size unchanged."""
    rect=QRectF(rect)
    def clear(candidate):
        return not any(candidate.intersects(other.adjusted(-4,-4,4,4)) for other in placed)
    if clear(rect):return rect
    # Check every placed badge, rather than bouncing between two neighbours.
    # Prefer space above; at the top edge try the free horizontal positions.
    for tops in ([other.top()-rect.height()-6 for other in placed], [rect.top()],
                 [other.bottom()+6 for other in placed]):
        candidates=[]
        for top in tops:
            if top<8:continue
            for left in [rect.left(),*[edge for other in placed for edge in
                                      (other.left()-rect.width()-6,other.right()+6)]]:
                if not 8<=left<=width-rect.width()-8:continue
                candidate=QRectF(left,top,rect.width(),rect.height())
                if clear(candidate):candidates.append(candidate)
        if candidates:
            return min(candidates,key=lambda r:(r.left()-rect.left())**2+(r.top()-rect.top())**2)
    return rect
