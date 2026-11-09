"""Per-bone fixed Euler order selected to avoid native Blender singularities."""
import itertools,warnings
import numpy as np
from scipy.spatial.transform import Rotation
from .bvh_bind import bind_pose,local_frames
from .bone_overlay import reference_frame

def continuous_angles(angles):
 """Choose equivalent Euler triples without changing any keyed rotation."""
 a=angles.copy();previous=np.zeros(3)
 for t in range(len(a)):
  x=a[t].copy();y=x.copy();y[0]+=np.pi;y[1]=np.pi-y[1];y[2]+=np.pi
  x+=2*np.pi*np.round((previous-x)/(2*np.pi));y+=2*np.pi*np.round((previous-y)/(2*np.pi))
  a[t]=x if np.linalg.norm(x-previous)<=np.linalg.norm(y-previous) else y;previous=a[t]
 return a

def wrist_order(pose):
 """Score fixed wrist orders in Blender's bone basis at fractional frames.

 SLERP is an interpolation reference, not anatomical ground truth. This affects
 encoding only; every keyed matrix stays exact and no Blender add-on is needed.
 """
 if len(pose)<2:return None
 relative=Rotation.from_matrix(pose[:-1].swapaxes(-1,-2)@pose[1:]).as_rotvec()
 references=[pose[:-1]@Rotation.from_rotvec(relative*w).as_matrix() for w in (.25,.5,.75)]
 options=[]
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  for seq in itertools.permutations('XYZ'):
   order=''.join(seq);a=continuous_angles(Rotation.from_matrix(pose).as_euler(order.lower()))
   errors=[]
   for w,reference in zip((.25,.5,.75),references):
    interpolated=Rotation.from_euler(order.lower(),(1-w)*a[:-1]+w*a[1:]).as_matrix()
    errors.append(Rotation.from_matrix(interpolated@reference.swapaxes(-1,-2)).magnitude())
   error=np.asarray(errors);options.append((float(abs(np.cos(a[:,1])).min()),float(error.max()+.05*error.mean()),order))
 admissible=[o for o in options if o[0]>.02] or options
 return min(admissible,key=lambda o:o[1])[2]

def write_bvh(path,rig,roots,local,fps):
 if len(roots)!=len(local) or not len(roots) or not np.isfinite(fps) or fps<=0:raise ValueError("Invalid BVH animation dimensions or frame rate")
 if not np.isfinite(roots).all() or not np.isfinite(local).all():raise ValueError("Non-finite BVH animation")
 binding=bind_pose(rig);rot=local_frames(local,binding);rest=binding['rest'];parents=binding['parents'];ends=binding['ends'];angles=[];orders=[];quality=[]
 for j in range(52):
  children=np.flatnonzero(parents==j);delta=rest[children].mean(0)-rest[j] if len(children) else ends[j]
  if np.linalg.norm(delta)<.001:delta=delta+np.array([0.,.1,0.])
  D=reference_frame(delta);pose=D.T@rot[:,j]@D;options=[]
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   for seq in itertools.permutations('XYZ'):
    order=''.join(seq);a=Rotation.from_matrix(rot[:,j]).as_euler(order);b=Rotation.from_matrix(pose).as_euler(order.lower())
    margin=min(float(abs(np.cos(a[:,1])).min()),float(abs(np.cos(b[:,1])).min()))
    options.append((margin,order,a))
  override=wrist_order(pose) if rig.get('source_animation_version',0)>=17 and j in (20,21) else None
  margin,order,a=next(o for o in options if o[1]==override) if override else max(options,key=lambda x:x[0])
  a=continuous_angles(a)
  np.testing.assert_allclose(Rotation.from_euler(order,a).as_matrix(),rot[:,j],atol=1e-5)
  angles.append(np.degrees(a));orders.append(order);quality.append(margin)
 write_channels(path,binding['names'],parents,rest,ends,roots,np.stack(angles,1),orders,fps)
 return dict(orders=orders,minimum_singularity_margin=min(quality))


def write_channels(path,names,parents,rest,ends,roots,angles,orders,fps,*,exclusive=False):
 """Write the hierarchy and precomputed Euler channels in depth-first order."""
 order=[];lines=['HIERARCHY']
 def visit(j,depth):
  pad='  '*depth;order.append(j);parent=parents[j];lines.extend([pad+('ROOT ' if parent<0 else 'JOINT ')+names[j],pad+'{'])
  offset=rest[j] if parent<0 else rest[j]-rest[parent];lines.append(pad+'  OFFSET '+' '.join(f'{v:.9f}' for v in offset))
  lines.append(pad+'  CHANNELS '+('6 Xposition Yposition Zposition ' if parent<0 else '3 ')+' '.join(c+'rotation' for c in orders[j]))
  children=np.flatnonzero(parents==j)
  for c in children:visit(int(c),depth+1)
  if not len(children):lines.extend([pad+'  End Site',pad+'  {',pad+'    OFFSET '+' '.join(f'{v:.9f}' for v in ends[j]),pad+'  }'])
  lines.append(pad+'}')
 visit(0,0);lines.extend(['MOTION',f'Frames: {len(roots)}',f'Frame Time: {1/fps:.10f}'])
 with open(path,'x' if exclusive else 'w',encoding='ascii') as f:
  f.write('\n'.join(lines)+'\n')
  for root,angle in zip(roots,angles):f.write(' '.join(f'{v:.9f}' for v in np.r_[root,angle[order].reshape(-1)])+'\n')
