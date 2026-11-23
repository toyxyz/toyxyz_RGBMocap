"""Map prepared source rigs to VMC without evaluating a model or a body mesh."""
import re
import numpy as np
from .pear_vmc import UNITY, quaternion

ALIASES = {
    'Hips':'Hips', 'Spine':'Spine', 'Spine1':'Chest', 'Spine2':'UpperChest',
    'Neck':'Neck', 'Head':'Head',
}
for side in ('Left', 'Right'):
    for source, target in (
        ('UpLeg','UpperLeg'), ('Leg','LowerLeg'), ('Foot','Foot'),
        ('ToeBase','Toes'), ('Shoulder','Shoulder'), ('Arm','UpperArm'),
        ('ForeArm','LowerArm'), ('Hand','Hand')):
        ALIASES[side+source] = side+target


def humanoid_names(names):
    """SOMA uses three spine bones and extra neck/finger/end joints."""
    clean = [str(name).rsplit(':',1)[-1] for name in names]
    soma = 'LeftShin' in clean and 'Neck2' in clean
    aliases = dict(ALIASES)
    if soma:
        aliases.update(Spine1='Spine', Spine2='Chest', Chest='UpperChest', Neck2='Neck')
        for side in ('Left', 'Right'):
            aliases[side+'Leg'] = side+'UpperLeg'
            aliases[side+'Shin'] = side+'LowerLeg'
    mapped = []
    for name in clean:
        result = aliases.get(name)
        match = re.fullmatch(r'(Left|Right)Hand(Thumb|Index|Middle|Ring|Pinky|Little)([123])',name)
        if match:
            side, finger, segment = match.groups()
            result = side+('Little' if finger=='Pinky' else finger)+('Proximal','Intermediate','Distal')[int(segment)-1]
        mapped.append(result)
    return mapped


class OfflineRig:
    """Keep original bone axes, collapsing only unmapped helper joints.

    The resulting local transforms reconstruct the mapped source pivots exactly.
    They are not an automatic bind-pose retarget to arbitrary avatar models.
    """
    def __init__(self, stream_rig):
        names = [bone[0] for bone in stream_rig.bones]
        self.parents = np.asarray([bone[1] for bone in stream_rig.bones],dtype=int)
        self.count = len(names)
        if not 22 <= self.count <= 128 or self.parents[0] != -1 or np.any(self.parents[1:]<0) or np.any(self.parents[1:]>=np.arange(1,self.count)):
            raise ValueError('Invalid prepared VMC hierarchy')
        self.names = humanoid_names(names)
        selected = [i for i,name in enumerate(self.names) if name is not None]
        known = [self.names[i] for i in selected]
        required = {'Hips','Spine','Head','LeftUpperLeg','RightUpperLeg',
                    'LeftLowerLeg','RightLowerLeg','LeftUpperArm','RightUpperArm',
                    'LeftLowerArm','RightLowerArm','LeftHand','RightHand'}
        if len(known)!=len(set(known)) or not required.issubset(known):
            raise ValueError('Prepared skeleton has no complete supported humanoid mapping')
        self.selected = tuple(selected)
        self.mapped_parents = []
        for index in selected:
            parent = int(self.parents[index])
            while parent >= 0 and self.names[parent] is None:
                parent = int(self.parents[parent])
            self.mapped_parents.append(parent)
        if self.names[0] != 'Hips' or any(parent < 0 for parent in self.mapped_parents[1:]):
            raise ValueError('VMC requires one Hips root')

    def transforms(self, world, transform=None):
        world = np.asarray(world,dtype=float)
        if world.shape != (self.count,4,4) or not np.isfinite(world).all():
            raise ValueError('Invalid prepared VMC frame')
        r = world[:,:3,:3].copy(); j = world[:,:3,3].copy()
        if transform is not None:
            transform = np.asarray(transform,dtype=float)
            if transform.shape != (4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3],[0,0,0,1],atol=1e-5):
                raise ValueError('Invalid source scene transform')
            r = transform[:3,:3] @ r
            j = j @ transform[:3,:3].T + transform[:3,3]
        if np.max(abs(r.swapaxes(-1,-2)@r-np.eye(3))) > .005 or np.min(np.linalg.det(r)) < .99:
            raise ValueError('VMC requires rigid bone transforms without scale')
        result = []
        for index, parent in zip(self.selected,self.mapped_parents):
            if parent < 0:
                position, local = j[index],r[index]
            else:
                position = r[parent].T @ (j[index]-j[parent])
                local = r[parent].T @ r[index]
            result.append((self.names[index],UNITY@position,quaternion(UNITY@local@UNITY)))
        return result
