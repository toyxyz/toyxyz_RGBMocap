"""Numerical/OSC/UDP verification without camera, licensed assets or GPU."""
import importlib.util
from pathlib import Path
import socket
import struct
import time
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('pear_vmc', Path(__file__).parents[1]/'mocap/pear_vmc.py')
vmc = importlib.util.module_from_spec(spec); spec.loader.exec_module(vmc)
PARENTS = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,15,15,15,
                    20,25,26,20,28,29,20,31,32,20,34,35,20,37,38,
                    21,40,41,21,43,44,21,46,47,21,49,50,21,52,53], dtype=np.int32)


def axis_rotation(axis, angle):
    a = np.zeros(3); a[axis] = 1
    x,y,z = a
    k = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3) + np.sin(angle)*k + (1-np.cos(angle))*(k@k)


def fixture():
    rng = np.random.default_rng(32)
    offsets = rng.normal(size=(55,3))*.08
    offsets[0] = [.25,1.1,2.]
    offsets[1] = [.12,-.1,0]; offsets[2] = [-.12,-.1,0]
    local = np.stack([axis_rotation(i%3, .03*i) for i in range(55)])
    positions = offsets.copy(); world = local.copy()
    for i in range(1,55):
        p = PARENTS[i]
        positions[i] = positions[p] + world[p]@offsets[i]
        world[i] = world[p]@local[i]
    return dict(joints=positions, rotations=world), offsets, local


def rotation(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def decode(packet):
    """Independent OSC reader: verifies types, padding, lengths and big endian."""
    if packet[:8]==b'#bundle\0':
        assert struct.unpack('>Q', packet[8:16])[0]==1
        offset=16; result=[]
        while offset<len(packet):
            length=struct.unpack_from('>I',packet,offset)[0]; offset+=4
            assert length>0 and offset+length<=len(packet)
            result.extend(decode(packet[offset:offset+length])); offset+=length
        return result
    offset=0
    def string():
        nonlocal offset
        end=packet.index(0,offset); value=packet[offset:end].decode('utf8')
        offset=(end+4)//4*4
        return value
    address=string(); tags=string(); values=[]
    assert tags[0]==','
    for tag in tags[1:]:
        if tag=='s':values.append(string())
        else:
            assert tag in 'if'
            values.append(struct.unpack_from('>'+tag,packet,offset)[0]); offset+=4
    assert offset==len(packet)
    return [(address,tags,values)]


class VmcTests(unittest.TestCase):
    def test_parent_local_and_fk_round_trip(self):
        pose, offsets, local=fixture(); untouched={k:v.copy() for k,v in pose.items()}
        bones=vmc.transforms(pose,PARENTS)
        self.assertEqual(len(bones),52)
        output={}; world={}
        for index,(name,p,q) in zip(vmc.NAMES,bones):
            np.testing.assert_allclose(p,vmc.UNITY@offsets[index],atol=1e-12)
            np.testing.assert_allclose(rotation(q),vmc.UNITY@local[index]@vmc.UNITY,atol=1e-12)
            parent=PARENTS[index]
            output[index]=p if parent<0 else output[parent]+world[parent]@p
            world[index]=rotation(q) if parent<0 else world[parent]@rotation(q)
            np.testing.assert_allclose(output[index],vmc.UNITY@pose['joints'][index],atol=1e-12)
        self.assertEqual(bones[1][0],'LeftUpperLeg'); self.assertLess(bones[1][1][0],0)
        for key in pose:np.testing.assert_array_equal(pose[key],untouched[key])

    def test_aligned_basis_and_posed_hierarchy_round_trip(self):
        pose,_,_=fixture()
        # Treat any physically valid FK pose as a post-correction packet.
        aligned=dict(joints=pose['joints']@vmc.DISPLAY.T,rotations=vmc.DISPLAY@pose['rotations'])
        for expected,actual in zip(vmc.transforms(pose,PARENTS),vmc.transforms(aligned,PARENTS,True)):
            self.assertEqual(expected[0],actual[0])
            np.testing.assert_allclose(expected[1],actual[1],atol=1e-12)
            np.testing.assert_allclose(rotation(expected[2]),rotation(actual[2]),atol=1e-12)

    def test_quaternions_at_pi(self):
        for axis in range(3):
            for angle in (0, .3, -1.2, np.pi, -np.pi+1e-9):
                r=axis_rotation(axis,angle); q=vmc.quaternion(r)
                self.assertAlmostEqual(np.linalg.norm(q),1)
                np.testing.assert_allclose(rotation(q),r,atol=1e-12)

    def test_protocol_and_finger_names(self):
        pose,_,_=fixture(); packets=vmc.frame_packets(pose,PARENTS,False,12.5)
        self.assertGreater(len(packets),1); self.assertTrue(all(len(p)<=1400 for p in packets))
        messages=[m for p in packets for m in decode(p)]
        bones=[m for m in messages if m[0]=='/VMC/Ext/Bone/Pos']
        self.assertEqual(len(bones),52); self.assertTrue(all(m[1]==',sfffffff' for m in bones))
        names={m[2][0] for m in bones}
        self.assertIn('RightLittleDistal',names); self.assertIn('LeftThumbProximal',names)
        self.assertNotIn('Jaw',names); self.assertEqual(messages[-1][0],'/VMC/Ext/T')
        self.assertEqual(messages[-1][2],[12.5]); self.assertEqual(messages[-2][2],[1])
        self.assertEqual(messages[0][2],['root',0.,0.,0.,0.,0.,0.,1.])

    def test_invalid_pose_and_destination(self):
        pose,_,_=fixture()
        for bad in (np.full((55,3,3),np.nan),np.zeros((55,3,3))):
            with self.assertRaises(ValueError):vmc.transforms(dict(pose,rotations=bad),PARENTS)
        for host in ('example.com','0.0.0.0','239.1.2.3'):
            with self.assertRaises(ValueError):vmc.Sender(host)
        parents=PARENTS.copy(); parents[5]=55
        with self.assertRaises(ValueError):vmc.transforms(pose,parents)

    def test_udp_missing_timeout_and_stop(self):
        receiver=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        receiver.bind(('127.0.0.1',0)); receiver.settimeout(2)
        sender=vmc.Sender('127.0.0.1',receiver.getsockname()[1])
        def until_ok(value):
            received=[]; deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                received.extend(decode(receiver.recvfrom(2048)[0]))
                if any(a=='/VMC/Ext/OK' and v==[value] for a,t,v in received):return received
            self.fail('Missing tracking status')
        try:
            until_ok(0); pose,_,_=fixture(); sender.submit(pose,PARENTS)
            messages=until_ok(1)
            self.assertEqual(sum(a=='/VMC/Ext/Bone/Pos' for a,t,v in messages),52)
            until_ok(0)  # Real worker timeout, no replay of a stale pose.
            sender.submit(pose,PARENTS); until_ok(1)
            sender.submit(None); until_ok(0)
            sender.close(); sender.thread.join(2); self.assertFalse(sender.thread.is_alive())
            until_ok(0); self.assertEqual(sender.error,'')
        finally:
            sender.close(); sender.thread.join(2); receiver.close()


if __name__=='__main__':unittest.main(verbosity=2)
