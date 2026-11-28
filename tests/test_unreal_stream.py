"""Numerical and protocol checks without Unreal, GPU, projects or model data."""
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from mocap.bone_stream import StreamRig, Y_TO_Z
from mocap.export_options import align_auto_floor, recorded_auto_floor
from mocap.unreal_stream import (SOURCE_TO_UE, YAW, local_transforms, rig_schema,
                                snapshot, subject_name, encode, LIMIT)


def quat_matrix(q):
    x,y,z,w=q
    return np.array(((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
                     (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
                     (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))))



class UnrealBakeTests(unittest.TestCase):
    def test_take_payload_keeps_sample_count_and_order(self):
        from threading import Event
        from mocap.unreal_bake import BakeTake, HEADER
        from types import SimpleNamespace
        frames = np.repeat(np.eye(4)[None,None], 3, axis=0)
        frames[:,0,0,3] = [0., 1., 4.]
        rig = SimpleNamespace(bones=[('root', -1)],
                              pose=lambda row, _: np.asarray(row[:, :3], dtype='<f4').tobytes())
        take = BakeTake(frames, 3, 30000/1001, rig, None)
        def convert(world, parents):
            return [[world[0,0,3], 0, 0, 0, 0, 0, 1]]
        encoded = take.encode(convert, Event())
        length, head = next(encoded)
        body = b''.join(encoded)
        self.assertEqual(HEADER.unpack(head), (b'RMB1', 30000, 1001, 3, 1))
        self.assertEqual(length, len(head)+len(body))
        np.testing.assert_equal(np.frombuffer(body, dtype='<f4').reshape(3,7)[:,0], [0,1,4])

    def test_bake_rejects_changed_frame_count(self):
        from threading import Event
        from mocap.unreal_bake import BakeTake
        from types import SimpleNamespace
        take = BakeTake(np.zeros((3,1,4,4)), 4, 30., SimpleNamespace(bones=[('root', -1)]), None)
        with self.assertRaisesRegex(ValueError, 'take changed'):
            next(take.encode(None, Event()))

    def test_face_is_opt_in_and_matches_the_take_clock(self):
        from http.client import HTTPConnection
        from threading import Event
        from types import SimpleNamespace
        from mocap.unreal_bake import BakeFace, BakeServer, BakeTake, HEADER

        def resolve(_project, frame, *, person, state):
            self.assertEqual((person, state), (2, {'start': 1.25, 'fps': 30.}))
            return 0, [-1, 0, 0, 1][frame], frame+5

        archive = SimpleNamespace(resolve=resolve, index={2: 0}, values=np.zeros((1, 2, 52)))
        archive.values[0, :, 23] = [.25, .75]
        frames = np.tile(np.eye(4), (4, 1, 1, 1))
        rig = SimpleNamespace(bones=[('root', -1)],
                              pose=lambda row, _: np.asarray(row[:, :3], dtype='<f4').tobytes())
        take = BakeTake(frames, 4, 30., rig, None, BakeFace(archive, 2, 1.25, 30.))
        server = BakeServer(lambda *_: [[0, 0, 0, 0, 0, 0, 1]], Event())
        self.addCleanup(server.stop)
        server.takes = {('Actor', 'Rig', 'Take'): take}
        connection = HTTPConnection('127.0.0.1', server.server_port, timeout=3)
        self.addCleanup(connection.close)
        url = '/take?subject=Actor&rig=Rig&take=Take'

        def get(headers):
            connection.request('GET', url, headers=headers)
            response = connection.getresponse()
            return response.status, response.read()

        self.assertEqual(get({'X-RGBMoCap-Face': '1'})[0], 403)
        auth = {'Authorization': 'Bearer ' + server.token}
        status, body = get(auth)
        self.assertEqual(status, 200)
        self.assertEqual(HEADER.unpack_from(body), (b'RMB1', 30, 1, 4, 1))
        status, face = get(auth | {'X-RGBMoCap-Face': '1'})
        self.assertEqual(status, 200)
        self.assertEqual(HEADER.unpack_from(face), (b'RMB2', 30, 1, 4, 1))
        samples = np.frombuffer(face, offset=HEADER.size, dtype='<f4').reshape(4, 59)
        np.testing.assert_array_equal(samples[:, :7], np.frombuffer(body, offset=HEADER.size, dtype='<f4').reshape(4, 7))
        np.testing.assert_array_equal(samples[:, 7+23], [0., .25, .25, .75])

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

    def test_auto_floor_moves_stream_motion_once_with_the_same_reference(self):
        rig = self.rig()
        world = np.tile(np.eye(4), (2, 1, 1))
        world[:, 1, 3] = [1., 2.]
        transform = np.eye(4)
        transform[:3, 3] = [3., .28, 4.]
        actor = 'a' * 64
        raw = snapshot(((actor, 'Person', rig, 0, world, transform),), {})['actors'][0]
        for floor in (-.018, .009):
            with self.subTest(floor=floor):
                rig.floor = (0., floor)
                shifted = snapshot(((actor, 'Person', rig, 0, world, transform),), {})['actors'][0]
                expected = np.zeros((2, 7))
                expected[0, 2] = -floor * 100.
                np.testing.assert_allclose(np.array(shifted['pose']) - raw['pose'], expected, atol=2e-5)
                self.assertEqual((shifted['rig'], shifted['rest']), (raw['rig'], raw['rest']))
                self.assertAlmostEqual(transform[1, 3], .28)

    def test_auto_floor_reading_preserves_explicit_height_and_uses_shared_world(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'corrected_initial_correction.json'
            report = dict(floor=-.018, segment_count=3, options=dict(mode='flat', auto_floor=True))

            def floor_for(kind='corrected_initial'):
                p.write_text(json.dumps(report), encoding='utf-8')
                return recorded_auto_floor(tmp, kind)

            self.assertEqual(floor_for(), (-.018, -.018))
            self.assertIsNone(floor_for('initial'))
            report['options']['auto_floor'] = False
            self.assertIsNone(floor_for())
            report['options'].update(auto_floor=True, mode='multi_level')
            self.assertIsNone(floor_for())
            report['options']['mode'] = 'flat'
            report['segment_count'] = 0
            self.assertIsNone(floor_for())
            report['shared_floor'] = dict(local_plane_height=-.11, local_plane_normal=[0, 1, 0])
            transform = np.eye(4)
            transform[1, 3] = .15
            self.assertAlmostEqual(align_auto_floor(transform, floor_for())[1, 3], .168)
            self.assertAlmostEqual(align_auto_floor(None, floor_for())[1, 3], .11)
            report['shared_floor']['local_plane_normal'] = [0, .8, .6]
            self.assertIsNone(align_auto_floor(None, floor_for()))

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
