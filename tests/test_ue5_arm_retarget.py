"""Anatomical regression for the mannequin arm, palm and digit bake."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from mocap.ue5_retarget import ARM, FINGERS, _arm_targets, _apply, _palm, _unit


class MannequinArmTests(unittest.TestCase):
    def check_arm(self, side, sign):
        prefix = 'Left' if side == 'l' else 'Right'
        source_names = [prefix + part for part in ('Shoulder', 'Arm', 'ForeArm', 'Hand')]
        target_names = [part + '_' + side for part in ARM]
        source_names += [f'{prefix}Hand{digit}{segment}' for digit in FINGERS for segment in range(1, 4)]
        target_names += [f'{digit.lower()}_{segment:02}_{side}' for digit in FINGERS for segment in range(1, 4)]
        source = {name: index for index, name in enumerate(source_names)}
        target = {name: index for index, name in enumerate(target_names)}
        source_p = np.zeros((len(source), 3))
        target_p = np.zeros((len(target), 3))
        source_p[:4] = [(0, 0, 0), (10, 0, 0), (36, 0, 0), (61, 0, 0)]
        target_p[:4] = [(0, 0, 0), (12, 0, 0), (30, 0, -24), (45, 18, -44)]
        for digit in range(5):
            first = 4 + digit * 3
            for segment in range(3):
                spread = 4. - digit * 1.6
                source_p[first + segment] = source_p[3] + [8 + 3*segment, spread, 0]
                target_p[first + segment] = target_p[3] + [6 + 2*segment, 0, spread]
        source_p[:, 0] *= sign
        target_p[:, 0] *= sign
        base_r = np.broadcast_to(np.eye(3), (len(target), 3, 3))
        motion = np.broadcast_to(np.eye(3), (2, len(source), 3, 3)).copy()
        posed = np.stack((source_p, source_p))
        motion[1, 2] = Rotation.from_rotvec([0., 0., sign * 1.2]).as_matrix()
        motion[1, 3] = Rotation.from_euler('xyz', [0.9, 0.35, sign * 1.2]).as_matrix()
        posed[:, 3] = posed[:, 2] + _apply(motion[:, 2], source_p[3]-source_p[2])
        tips = np.broadcast_to((sign * 2., 0.5, -0.5), source_p.shape)
        for digit in range(5):
            start = 4 + digit * 3
            posed[:, start] = posed[:, 3] + _apply(motion[:, 3], source_p[start] - source_p[3])
            for segment in range(3):
                bone = start + segment
                motion[:, bone] = motion[:, 3]
                bend = Rotation.from_rotvec([0., sign * (0.3 + digit * .12) * (segment+1), 0.]).as_matrix()
                motion[1, bone] = motion[1, 3] @ bend
                if segment < 2:
                    posed[:, bone + 1] = posed[:, bone] + _apply(motion[:, bone], source_p[bone+1]-source_p[bone])
        result = _arm_targets(source, target, source_p, posed, target_p, base_r, motion, tips, prefix, side)
        np.testing.assert_allclose(_unit(_apply(result[2], target_p[3]-target_p[2])),
                                   _unit(posed[:, 3]-posed[:, 2]), atol=1e-7)
        anchors = (7, 10, 16)
        np.testing.assert_allclose(result[3] @ _palm(target_p, 3, anchors), _palm(posed, 3, anchors), atol=1e-7)
        for bone in (4, 5, 13, 14):
            np.testing.assert_allclose(_unit(_apply(result[bone], target_p[bone+1]-target_p[bone])),
                                       _unit(posed[:, bone+1]-posed[:, bone]), atol=1e-7)
        for last in (6, 9, 18):
            np.testing.assert_allclose(_unit(_apply(result[last], (sign, 0., 0.))),
                                       _unit(_apply(motion[:, last], tips[last])), atol=1e-7)
        self.assertTrue(all(np.isfinite(matrix).all() for matrix in result.values()))

    def test_left_bent_and_extended_arm_and_fingers(self):
        self.check_arm('l', 1.)

    def test_right_bent_and_extended_arm_and_fingers(self):
        self.check_arm('r', -1.)


if __name__ == '__main__':
    unittest.main()
