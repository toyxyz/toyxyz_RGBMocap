"""Model-free prepared-rig mapping and scene-transform verification."""
from pathlib import Path
import importlib.util
import sys
import types
import unittest
import numpy as np

ROOT=Path(__file__).parents[1]
# Load the source package without importing any model or Qt runtime.
if 'mocap' not in sys.modules:
    package=types.ModuleType('mocap');package.__path__=[str(ROOT/'mocap')];sys.modules['mocap']=package
from mocap.vmc_rig import OfflineRig,humanoid_names
from mocap.pear_vmc import UNITY,transform_packets
from test_pear_vmc import rotation,axis_rotation,decode

BODY=('Hips','LeftUpLeg','RightUpLeg','Spine','LeftLeg','RightLeg','Spine1',
      'LeftFoot','RightFoot','Spine2','LeftToeBase','RightToeBase','Neck',
      'LeftShoulder','RightShoulder','Head','LeftArm','RightArm','LeftForeArm',
      'RightForeArm','LeftHand','RightHand')
PARENTS=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]


def rig_fixture(soma=False):
    names=list(BODY);parents=list(PARENTS)
    for side,hand in (('Left',20),('Right',21)):
        for finger in ('Thumb','Index','Middle','Ring','Pinky'):
            parent=hand
            for segment in range(1,4):
                names.append(side+'Hand'+finger+str(segment));parents.append(parent);parent=len(names)-1
    if soma:
        # Same semantic body, SOMA names plus an unmapped physical neck helper.
        names[3]='Spine1';names[6]='Spine2';names[9]='Chest';names[12]='Neck2'
        for side,hip,knee in (('Left',1,4),('Right',2,5)):
            names[hip]=side+'Leg';names[knee]=side+'Shin'
        names.insert(12,'Neck1');parents=[p+1 if p>=12 else p for p in parents]
        parents.insert(12,9);parents[13]=12
    n=len(names);rng=np.random.default_rng(68)
    offsets=rng.normal(0,.08,(n,3));offsets[0]=[.3,1.1,2.]
    matrices=np.tile(np.eye(4),(n,1,1))
    for i,p in enumerate(parents):
        local=axis_rotation(i%3,i*.04)
        matrices[i,:3,:3]=local if p<0 else matrices[p,:3,:3]@local
        matrices[i,:3,3]=offsets[i] if p<0 else matrices[p,:3,3]+matrices[p,:3,:3]@offsets[i]
    rig=types.SimpleNamespace(bones=[[name,p,[],.1] for name,p in zip(names,parents)],key='test-soma' if soma else 'test-smpl')
    return rig,matrices


class OfflineRigTests(unittest.TestCase):
    def check_fk(self,soma=False,scene=None):
        rig,world=rig_fixture(soma);adapter=OfflineRig(rig);before=world.copy()
        bones=adapter.transforms(world,scene);points={};frames={}
        expected=world[:,:3,3] if scene is None else world[:,:3,3]@scene[:3,:3].T+scene[:3,3]
        for i,parent,(_,p,q) in zip(adapter.selected,adapter.mapped_parents,bones):
            points[i]=p if parent<0 else points[parent]+frames[parent]@p
            frames[i]=rotation(q) if parent<0 else frames[parent]@rotation(q)
            np.testing.assert_allclose(points[i],UNITY@expected[i],atol=1e-12)
        np.testing.assert_array_equal(world,before)
        self.assertEqual(len(bones),52)
        messages=[m for packet in transform_packets(bones,1.) for m in decode(packet)]
        self.assertEqual(sum(m[0]=='/VMC/Ext/Bone/Pos' for m in messages),52)

    def test_common_smpl_prepared_rig(self):self.check_fk()
    def test_soma_collapsed_neck(self):self.check_fk(True)
    def test_scene_placement(self):
        scene=np.eye(4);scene[:3,:3]=axis_rotation(1,.7);scene[:3,3]=[2.,.1,-3.]
        self.check_fk(False,scene);self.check_fk(True,scene)
    def test_known_native_soma_semantics(self):
        names=['Hips','Spine1','Spine2','Chest','Neck1','Neck2','Head',
               'LeftLeg','LeftShin','LeftHandPinky3','LeftHandIndex4','HeadEnd']
        self.assertEqual(humanoid_names(names),['Hips','Spine','Chest','UpperChest',None,'Neck','Head',
               'LeftUpperLeg','LeftLowerLeg','LeftLittleDistal',None,None])
    def test_invalid_inputs(self):
        rig,world=rig_fixture();adapter=OfflineRig(rig)
        with self.assertRaises(ValueError):adapter.transforms(world[:22])
        world[2,0,0]=np.nan
        with self.assertRaises(ValueError):adapter.transforms(world)
        rig,world=rig_fixture();scaled=np.diag([2.,2.,2.,1.])
        with self.assertRaises(ValueError):adapter.transforms(world,scaled)
        rig.bones[1][0]='Unknown'
        with self.assertRaises(ValueError):OfflineRig(rig)


if __name__=='__main__':unittest.main(verbosity=2)
