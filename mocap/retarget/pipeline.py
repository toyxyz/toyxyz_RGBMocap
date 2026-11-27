"""Versioned traditional retarget pipeline, running in its own worker."""
import os,time,json
from pathlib import Path
import numpy as np
from . import solver as m, stabilize as stab, surface, measured as t, foot_calibration
from .calibration import clavicle_rotations,desired,reorient_pose

def run(folder,kind):
 original_safe=m.source_safe_reach;original_two=m.two_bone
 try:return _run(folder,kind)
 finally:
  m.source_safe_reach=original_safe;m.two_bone=original_two

def _run(folder,kind):
 OUT=Path(folder);label='motion'
 os.environ['PERFORMANCE_FAST']='1';os.environ['STOP_STAGNATION']='1'
 m.OUT=m.PREVIOUS=OUT;m.LABEL=label
 foot_calibration.AUDIT=OUT;foot_calibration.SOURCES.clear()
 source=dict(np.load(OUT/'motion-source.npz'));native=np.load(OUT/'motion-source-neutral.npz')['native_world']
 m.FPS=float(source.get('fps',30.))
 neutral=np.load(OUT/'motion-neutral-calibration.npz')['neutral'];active=np.load(OUT/'motion-source-contacts.npz')['active']
 evidence=np.load(OUT/'motion-evidence.npz');F=evidence['frames'];F0=evidence['neutral_frames']
 original_safe=m.source_safe_reach;original_two=m.two_bone
 reference_points=None;safe_cache={}
 def safe_reach(rest,src,side):return original_safe(rest,dict(src,points=reference_points,_safe_reach=safe_cache),side)
 def two_bone(base,root,goals,planned,src):return original_two(base,root,goals,planned,dict(src,world=base['world']))
 m.source_safe_reach=safe_reach;m.two_bone=two_bone
 started=time.perf_counter();geo=np.load(OUT/f'{kind}-contact-geometry.npz');target=np.load(OUT/f'{kind}-canonical.npz');rest=target['rest']
 ratio=np.mean(np.linalg.norm(rest[[4,5]]-rest[[1,2]],axis=1)+np.linalg.norm(rest[[7,8]]-rest[[4,5]],axis=1))/np.mean(np.linalg.norm(source['rest'][[4,5]]-source['rest'][[1,2]],axis=1)+np.linalg.norm(source['rest'][[7,8]]-source['rest'][[4,5]],axis=1))
 local=m.baseline(source['world'],source['rest'],rest);local,cal=m.corrected_world(local,m.PARENTS,label,geo)
 root=source['points'][:,0]*ratio;root[:,1]+=rest[0,1]-source['rest'][0,1]*ratio;p,w=m.fk(local,rest,root)
 raw=dict(local=local,root=root,points=p,world=w,rest=rest,names=target['names'],ratio=ratio,**cal)
 base,_=m.characterize(raw,raw,source,native);want=desired(source['world'],neutral,rest)
 base['world'][:,[13,14]]=clavicle_rotations(source['world'],neutral,base['world'][:,9],source['rest'],rest)
 for j in [1,2,4,5]:base['world'][:,j]=want[:,j]
 base['local']=m.local_from_world(base['world']);base['points'],base['world']=m.fk(base['local'],rest,base['root'])
 # Store a rotation-only control before contact correction, same calibration.
 np.savez(OUT/f'{label}-{kind}-before.npz',**base)
 reference_points=base['points'].copy();safe_cache={}
 fixed,plan,diagnostic=m.refine(base,source,active,geo)
 np.savez(OUT/f'{label}-{kind}-rotation.npz',**base);np.savez(OUT/f'{label}-{kind}-stable-contact.npz',**fixed)
 np.savez(OUT/f'{label}-{kind}-contact-plan.npz',active=active,anchors=plan['anchors'],goals=plan['goals'],support_weights=plan['weights'],palm_ids_left=plan['palms'][0],palm_ids_right=plan['palms'][1])
 stages=m.run(kind);old=dict(np.load(OUT/f'{label}-{kind}-stable-contact.npz'));old=reorient_pose(old,source['world'],neutral,m.align_many,m.fk,m.local_from_world)
 seed=dict(np.load(OUT/f'{label}-{kind}-root-base.npz'));old,stability=stab.solve(old,seed,active,{})
 old=reorient_pose(old,source['world'],neutral,m.align_many,m.fk,m.local_from_world)
 poles,ft=surface.mapped_poles(old,F,F0);old=surface.rebuild(old,poles,ft)
 frames,deltas,angles=t.target_frames(rest);w=old['world'].copy()
 for j in [1,2,4,5]:w[:,j]=w[:,j]@deltas[j]
 local=m.local_from_world(w);pp,ww=m.fk(local,rest,old['root']);free_delta=pp[:,[7,8]]-old['points'][:,[7,8]];free_delta[:,:,1]=0;fixed_delta=free_delta.copy();runs=[]
 for s in [0,1]:
  for a,b in m.trajectory.runs(active[:,s]):
   fixed_delta[a:b,s]=np.median(free_delta[a:min(b,a+5),s],axis=0);runs.append(dict(foot=s,start=a,end=b))
 off,_=m.trajectory.contact_offsets(np.zeros((len(active),4,3)),active[:,:2],fixed_delta,free_delta,np.zeros((len(active),3)),runs,m.FPS,np.ones((len(active),2)),~active[:,:2].any(1))
 delta=free_delta+off;goals=old['points'][:,[7,8]]+delta;mapped=np.stack([F[:,s]@F0[s].T@frames[h] for s,h in enumerate([1,2])],1)
 if max(abs(np.array(angles)))<.05:new=old;report=dict(seconds=0,reason='parallel neutral legs')
 else:
  direct,unreachable=t.rebuild_at_goals(old,goals,mapped)
  solved,report=t.solve(direct,old,active,dict(iterations=10,physical_sigma=.008,contact_sigma=.00005,root_prior=.004,root_limit=min(.015,.025*ratio)),goal_delta=goals-direct['points'][:,[7,8]])
  axis=t.unit(solved['points'][:,[7,8]]-solved['points'][:,[1,2]]);new=surface.rebuild(solved,t.unit(np.cross(mapped[:,:,:,0],axis)),mapped)
 plan=dict(np.load(OUT/f'{label}-{kind}-contact-plan.npz'));plan['anchors'][:,:2]+=fixed_delta;plan['goals'][:,:2]+=delta
 np.savez(OUT/f'{label}-{kind}-contact-plan.npz',**plan);np.savez(OUT/f'{label}-{kind}-stance.npz',**new)
 length=np.linalg.norm(new['points'][:,1:]-new['points'][:,m.PARENTS[1:]],axis=-1);expect=np.linalg.norm(rest[1:]-rest[m.PARENTS[1:]],axis=-1)
 assert np.max(abs(length-expect))<1e-8 and np.isfinite(new['world']).all()

 # Restore the selected source's absolute world anchor after proportion scaling.
 for method in ['stance','rotation']:
  path=OUT/f'{label}-{kind}-{method}.npz';data=dict(np.load(path))
  data['root']+=source['origin'];data['points']+=source['origin']
  np.savez(path,**data)
 m.source_safe_reach=original_safe;m.two_bone=original_two
 return dict(seconds=time.perf_counter()-started,fixed_length_error_mm=float(abs(length-expect).max()*1000),frames=len(active))
