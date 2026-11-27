import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from mocap.retarget.calibration import clavicle_rest_lift, clavicle_rotations


def rest_pose(upper_deg, clavicle_deg):
    arm, collar = np.radians([upper_deg, clavicle_deg])
    out = np.zeros((22, 3))
    out[[13, 14]] = [[.02, 0, 0], [-.02, 0, 0]]
    for side, sign in enumerate([1, -1]):
        out[16 + side] = out[13 + side] + [.15 * sign * np.cos(collar), .15 * np.sin(collar), 0]
        out[18 + side] = out[16 + side] + [.30 * sign * np.cos(arm), .30 * np.sin(arm), 0]
    return out


class ClavicleRetargetTests(unittest.TestCase):
    def test_t_pose_preserves_target_rest_and_chest_follow(self):
        neutral = np.repeat(np.eye(3)[None], 22, axis=0)
        neutral[9] = Rotation.from_euler('x', -12, degrees=True).as_matrix()
        neutral[13:15] = Rotation.from_euler('z', [24, -20], degrees=True).as_matrix()
        source_chest = Rotation.from_euler('y', 30, degrees=True).as_matrix()
        target_chest = Rotation.from_euler('yx', [-25, 10], degrees=True).as_matrix()
        source = source_chest @ neutral[None]
        result = clavicle_rotations(source, neutral, target_chest[None],
                                    rest_pose(0, 18), rest_pose(-8, -24))
        np.testing.assert_allclose(result[0], np.repeat(target_chest[None], 2, axis=0), atol=1e-12)

    def test_a_pose_lift_keeps_one_sided_shrug_about_corrected_reference(self):
        neutral = np.repeat(np.eye(3)[None], 22, axis=0)
        neutral[13:15] = Rotation.from_euler('z', [20, -23], degrees=True).as_matrix()
        chest = Rotation.from_euler('yx', [22, -15], degrees=True).as_matrix()
        shrug = Rotation.from_euler('z', 15, degrees=True).as_matrix()
        source = chest @ neutral[None]
        source[0, 13] = chest @ shrug @ neutral[13]
        reference, target = rest_pose(0, 18), rest_pose(-55, -9)
        lift = clavicle_rest_lift(reference, target)
        part = 27 * (1 - np.cos(np.radians(45)))
        np.testing.assert_allclose(lift, Rotation.from_euler('z', [part, -part], degrees=True).as_matrix(), atol=1e-12)
        result = clavicle_rotations(source, neutral, chest[None], reference, target)
        np.testing.assert_allclose(result[0], np.stack([chest @ shrug @ lift[0], chest @ lift[1]]), atol=1e-12)

    def test_matching_a_poses_do_not_add_a_lift(self):
        result = clavicle_rest_lift(rest_pose(-55, 3), rest_pose(-55, -9))
        np.testing.assert_allclose(result, np.broadcast_to(np.eye(3), result.shape), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
