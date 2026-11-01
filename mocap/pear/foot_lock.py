"""Small causal live-foot constraint. No meshes, inference, recording or lookahead."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .bone_roll import JOINTS,TIPS


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def aim(a,b):
    a,b=unit(a),unit(b);v=np.cross(a,b);c=float(np.clip(a@b,-1,1))
    if c < -.99999:
        axis=unit(np.cross(a,np.eye(3)[np.argmin(abs(a))]));return 2*np.outer(axis,axis)-np.eye(3)
    k=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+k+k@k/max(1+c,1e-12)


class Contact:
    def __init__(self):
        self.locked=False;self.candidate=0.;self.age=1.;self.offset=np.zeros(3);self.offset_v=np.zeros(3)
        self.anchor=None;self.position=None;self.velocity=np.zeros(3);self.input=None;self.speed=0.
        self.strength=0.;self.strength_v=0.

    def update(self,p,dt,enabled=True):
        velocity=(p-self.input)/dt if self.input is not None else np.zeros(3)
        self.input=p.copy();self.speed+=(1-np.exp(-dt/.045))*(np.linalg.norm(velocity)-self.speed)
        # Contact evidence always comes from the uncorrected input, never the lock.
        self.candidate=self.candidate+dt if enabled and -.06<p[1]<.045 and self.speed<.18 else 0.
        locked=self.locked
        if not locked and self.candidate>=.075:locked=True
        if locked and (not enabled or p[1]>.065 or self.speed>.30 or (self.anchor is not None and np.linalg.norm(p-self.anchor)>.12)):locked=False
        base=self.anchor if self.locked else p;base_v=np.zeros(3) if self.locked else velocity
        duration=.16;self.age+=dt;t=min(1.,self.age/duration)
        h0=2*t**3-3*t*t+1;h1=t**3-2*t*t+t
        value=base+h0*self.offset+h1*duration*self.offset_v
        speed=base_v+(6*t*t-6*t)/duration*self.offset+(3*t*t-4*t+1)*self.offset_v
        if locked!=self.locked:
            self.locked=locked
            if locked:self.anchor=p.copy();self.anchor[1]=0.
            base=self.anchor if locked else p;base_v=np.zeros(3) if locked else velocity
            self.offset=value-base;self.offset_v=speed-base_v;self.age=0.
        # A separate continuous activation also hands pelvis displacement back
        # to free feet; ending an IK pass must not drop that displacement.
        goal=float(locked);omega=24.;decay=np.exp(-omega*dt);x=self.strength-goal;v=self.strength_v
        self.strength=goal+(x+(v+omega*x)*dt)*decay
        self.strength_v=(v-omega*(v+omega*x)*dt)*decay
        self.position=value;self.velocity=speed
        return value


class LiveFootLock:
    def __init__(self):self.reset()
    def reset(self):
        self.time=None;self.feet=[Contact(),Contact()];self.shift=0.;self.shift_v=0.;self.last=None;self.poles=[None,None]
        self.corrections=np.tile(np.eye(3),(2,2,1,1))

    def smooth_correction(self,side,bone,target,dt):
        previous=self.corrections[side,bone]
        error=Rotation.from_matrix(target@previous.T).as_rotvec()
        value=Rotation.from_rotvec(error*(1-np.exp(-dt/.05))).as_matrix()@previous
        self.corrections[side,bone]=value
        return value

    def apply(self,pose,rig,timestamp,enabled=True,move_root=True):
        started=time.perf_counter();dt=timestamp-self.time if self.time is not None else 0.
        if not enabled or rig.sole_offsets is None:
            self.reset();return pose,dict(enabled=False,ms=(time.perf_counter()-started)*1000)
        if dt==0 and self.last is not None:return self.last
        if not 0<dt<=.25:self.reset();dt=1/30
        self.time=timestamp
        source=np.asarray(pose['joints'],dtype=float);rot=np.asarray(pose['rotations'],dtype=float)
        points=np.array([source[a]+rot[a]@offset for a,offset in zip((7,8),rig.sole_offsets)])
        # High feet never become contacts, including a held pose in mid-air.
        targets=np.array([f.update(p,dt) for f,p in zip(self.feet,points)])
        reach=[];limit=0.
        for side,(hip,knee,ankle) in enumerate(((1,4,7),(2,5,8))):
            a,b,c=source[[hip,knee,ankle]];length=np.linalg.norm(b-a)+np.linalg.norm(c-b)
            maximum=min(length-1e-7,np.linalg.norm(c-a))
            reach.append(maximum)
            if self.feet[side].strength>1e-4:
                target=targets[side]-rot[ankle]@rig.sole_offsets[side]
                horizontal=np.linalg.norm((target-a)[[0,2]])
                if horizontal<maximum:limit=min(limit,(target[1]+np.sqrt(maximum**2-horizontal**2)-a[1])*self.feet[side].strength)
        goal=max(-.04,limit) if move_root else 0.
        # Critically damped, bounded pelvis compensation. It cannot oscillate.
        omega=12.;decay=np.exp(-omega*dt);x=self.shift-goal;v=self.shift_v
        self.shift=goal+(x+(v+omega*x)*dt)*decay
        self.shift_v=(v-omega*(v+omega*x)*dt)*decay
        j=source.copy();j[:,1]+=self.shift if move_root else 0.
        r=rot.copy();facing=np.asarray(pose['bone_facing'],dtype=float).copy();errors=[]
        for side,(hip,knee,ankle,toe) in enumerate(((1,4,7,10),(2,5,8,11))):
            # Fade-out targets remain active until their inertial offset expires.
            contact=self.feet[side]
            inactive=not contact.locked and contact.age>=.16 and contact.strength<1e-4
            a,b,c=j[[hip,knee,ankle]].copy();target=targets[side]-rot[ankle]@rig.sole_offsets[side]
            if move_root:target[1]+=self.shift*(1-contact.strength)
            l1,l2=np.linalg.norm(b-a),np.linalg.norm(c-b)
            if min(l1,l2)<1e-7 or reach[side]<abs(l1-l2)+1e-7 or np.linalg.norm(target-a)<1e-8:
                errors.append(float(np.linalg.norm(c-target)));continue
            axis=unit(target-a)
            distance=np.clip(np.linalg.norm(target-a),abs(l1-l2)+1e-7,reach[side])
            # Near full extension, tiny height errors otherwise become a large
            # square-root knee displacement. Retain the source bend there and
            # accept foot residual instead of manufacturing crouch/extension.
            cosine=np.clip((l1*l1+l2*l2-reach[side]**2)/(2*l1*l2),-1.,1.)
            flex=180-np.degrees(np.arccos(cosine));observable=np.clip((flex-6)/12,0.,1.)
            observable=observable*observable*(3-2*observable)
            distance=reach[side]+observable*(distance-reach[side])
            knee_side=b-a-axis*np.dot(b-a,axis)
            if np.linalg.norm(knee_side)<.001:
                knee_side=self.poles[side] if self.poles[side] is not None else pose['bone_facing'][hip,:,2]
                knee_side=knee_side-axis*np.dot(knee_side,axis)
            if np.linalg.norm(knee_side)<1e-8:knee_side=np.cross(axis,np.eye(3)[np.argmin(abs(axis))])
            pole=unit(knee_side);self.poles[side]=pole
            along=(l1*l1-l2*l2+distance*distance)/(2*distance)
            new_b=a+axis*along+pole*np.sqrt(max(0,l1*l1-along*along));new_c=a+axis*distance
            desired=(aim(b-a,new_b-a),aim(c-b,new_c-new_b)) if not inactive else (np.eye(3),np.eye(3))
            applied=[self.smooth_correction(side,k,value,dt) for k,value in enumerate(desired)]
            new_b=a+applied[0]@(b-a);new_c=new_b+applied[1]@(c-b)
            for bone,correction in zip((hip,knee),applied):
                r[bone]=correction@r[bone]
                facing[int(np.flatnonzero(JOINTS==bone)[0])]=correction@facing[int(np.flatnonzero(JOINTS==bone)[0])]
            delta=new_c-c;j[knee]=new_b;j[ankle]=new_c;j[toe]+=delta;j[TIPS[toe]]+=delta
            errors.append(float(np.linalg.norm(new_c-target)))
        result=dict(pose,joints=j.astype('float32'),rotations=r.astype('float32'),bone_facing=facing.astype('float32'))
        result['bounds']=np.ascontiguousarray(j[np.r_[JOINTS,list(TIPS.values())]],dtype='float32')
        metrics=dict(enabled=True,locked=[f.locked for f in self.feet],residual_mm=[e*1000 for e in errors],pelvis_mm=float(self.shift*1000),ms=(time.perf_counter()-started)*1000)
        self.last=(result,metrics);return self.last
