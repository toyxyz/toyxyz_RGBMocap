"""Numerical and protocol checks without Unreal, GPU, projects or model data."""
import json
import struct
import unittest
import numpy as np

from mocap.bone_stream import StreamRig, Y_TO_Z
from mocap.unreal_stream import (SOURCE_TO_UE, YAW, local_transforms, rig_schema,
                                snapshot, subject_name, encode, LIMIT)


def quat_matrix(q):
    x,y,z,w=q
    return np.array(((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
                     (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
                     (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))))


class LiveLinkTests(unittest.TestCase):
    def test_parent_local_roundtrip_and_handedness(self):
        rng=np.random.default_rng(42)
        q=rng.normal(size=(40,4))
        q/=np.linalg.norm(q,axis=1,keepdims=True)
        rotations=np.stack([quat_matrix(v) for v in q])
        points=rng.normal(size=(40,3))
        world=np.concatenate((Y_TO_Z@rotations,(Y_TO_Z@points[...,None])),axis=2)
        parents=[-1]+[(i-1)//2 for i in range(1,40)]
        local=np.array(local_transforms(world,parents))
        expected_r=YAW@SOURCE_TO_UE@rotations@SOURCE_TO_UE.T
        expected_p=(YAW@SOURCE_TO_UE@points[...,None])[...,0]*100
        rebuilt_r,rebuilt_p=[],[]
        for i,parent in enumerate(parents):
            r=quat_matrix(local[i,3:])
            p=local[i,:3]
            if parent>=0:
                p=rebuilt_p[parent]+rebuilt_r[parent]@p
                r=rebuilt_r[parent]@r
            rebuilt_r.append(r)
            rebuilt_p.append(p)
        np.testing.assert_allclose(rebuilt_r,expected_r,atol=1e-12)
        np.testing.assert_allclose(rebuilt_p,expected_p,atol=1e-10)

    def rig(self):
        return StreamRig(['Hips','Child'],[-1,0],np.array(((0,1,0),(0,2,0))),
                         np.tile(np.eye(3),(2,1,1)),np.tile(np.eye(3),(2,1,1)),[1.,.2])

    def test_scene_transform_and_leaf_lengths(self):
        rig=self.rig()
        world=np.tile(np.eye(4),(2,1,1))
        world[:,:3,3]=((0,1,0),(0,2,0))
        transform=np.eye(4)
        transform[:3,3]=(3,0,0)
        a=snapshot((('a'*64,'Person 1',rig,80,world,transform),),{})['actors'][0]
        b=snapshot((('a'*64,'Person 1',rig,80,world,None),),{})['actors'][0]
        np.testing.assert_allclose(np.array(a['pose'])[1],np.array(b['pose'])[1])
        np.testing.assert_allclose(np.array(a['pose'])[0,:3]-np.array(b['pose'])[0,:3],(300,0,0),atol=1e-5)
        np.testing.assert_allclose(a['lengths'],[100,20])
        self.assertEqual(a['frame'],80)
        self.assertEqual(a['rest'],b['rest'])

    def test_invalid_rig_and_nonrigid_inputs(self):
        rig=self.rig()
        rig.bones[1][1]=1
        with self.assertRaises(ValueError):rig_schema(rig)
        rig=self.rig()
        rig.bones[1][0]='hIPS'
        with self.assertRaises(ValueError):rig_schema(rig)
        for invalid in (np.zeros((2,3,4)),np.full((2,3,4),np.nan)):
            with self.assertRaises(ValueError):local_transforms(invalid,[-1,0])

    def test_snapshot_releases_unused_schemas(self):
        rig=self.rig()
        world=np.tile(np.eye(4),(2,1,1))
        cache={}
        snapshot((('a'*64,'Person 1',rig,0,world,None),),cache)
        self.assertEqual(len(cache),1)
        self.assertEqual(snapshot((),cache),dict(version=1,actors=[]))
        self.assertFalse(cache)

    def test_readable_stable_subject_names(self):
        actor='a29d53a7'+'0'*56
        self.assertEqual(subject_name(actor,'Person 1'),'RGBMoCap_Person_1_a29d53a7')
        self.assertEqual(subject_name(actor,' Hero / 1: '),'RGBMoCap_Hero_1_a29d53a7')
        self.assertTrue(subject_name(actor,'인물 1').startswith('RGBMoCap_인물_1_'))
        self.assertLessEqual(len(subject_name(actor,'long'*100)+'_Reference'),90)
        self.assertEqual(subject_name(actor,''),'RGBMoCap_Person_a29d53a7')
        self.assertNotEqual(subject_name(actor,'Person 1'),subject_name('b'*64,'Person 1'))
        rig=self.rig()
        world=np.tile(np.eye(4),(2,1,1))
        cache={}
        for frame in (0,80,180,0):
            item=snapshot(((actor,'Person 1',rig,frame,world,None),),cache)['actors'][0]
            self.assertEqual(item['subject'],'RGBMoCap_Person_1_a29d53a7')
        # Rare short-ID collisions must fail visibly instead of mixing two actors.
        with self.assertRaisesRegex(ValueError,'collide'):
            snapshot(((actor,'Person 1',rig,0,world,None),
                      ('a29d53a7'+'1'*56,'person 1',rig,0,world,None)),{})
        with self.assertRaises(ValueError):subject_name('invalid','Person 1')

    def test_utf16_limits_match_unreal_strings(self):
        rig=self.rig()
        world=np.tile(np.eye(4),(2,1,1))
        label='\U0001f680'*61+' Person'
        actor=snapshot((('a'*64,label,rig,0,world,None),),{})['actors'][0]
        self.assertEqual(actor['label'],'\U0001f680'*60)
        self.assertEqual(len(actor['label'].encode('utf-16-le')),240)
        rig.bones[1][0]='\U00010400'*61
        with self.assertRaises(ValueError):rig_schema(rig)
        rig.bones[1][0]='\U00010400'*60
        self.assertEqual(rig_schema(rig)['names'][1],rig.bones[1][0])

    def test_protocol_bounds_and_finite_json(self):
        raw=encode(dict(version=1,actors=[]))
        self.assertEqual(struct.unpack('<I',raw[:4])[0],len(raw)-4)
        self.assertEqual(json.loads(raw[4:])['version'],1)
        with self.assertRaises(ValueError):encode({'bad':float('nan')})
        with self.assertRaises(ValueError):encode({'big':'a'*LIMIT})


if __name__=='__main__':unittest.main()
