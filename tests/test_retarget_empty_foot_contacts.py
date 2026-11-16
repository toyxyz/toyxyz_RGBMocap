"""Model-free reproducer and regression tests for the traditional leg solver."""

import unittest

import numpy as np

from mocap.retarget import solver


def leg_inputs():
    frame_count = 3
    rest = np.zeros((22, 3))
    for side, x in enumerate((-0.1, 0.1)):
        hip, knee, ankle = solver.CHAINS[side]
        rest[[hip, knee, ankle]] = [[x, 0, 0], [x, -0.4, 0.1], [x, -0.8, 0]]
    local = np.tile(np.eye(3), (frame_count, len(rest), 1, 1))
    root = np.zeros((frame_count, 3))
    points, world = solver.fk(local, rest, root)
    pose = dict(rest=rest, root=root, local=local, points=points, world=world, ratio=1.0)
    goals = points[:, [7, 8]].copy()
    saved = {"active": np.zeros((frame_count, 4), dtype=bool)}
    return pose, goals, saved


class EmptyFootContactsTest(unittest.TestCase):
    def assert_no_foot_contacts(self, pose, goals, saved):
        output, report = solver.legs(
            dict(pose, _contact_ankle_goals=goals), pose, pose, {}, saved
        )
        self.assertTrue(report["success"])
        self.assertEqual(report["leg_ankle_target_max_mm"], 0.0)
        for key in ("root", "local", "points", "world"):
            np.testing.assert_array_equal(output[key], pose[key])

    def test_all_contacts_disabled(self):
        self.assert_no_foot_contacts(*leg_inputs())

    def test_hand_contact_without_foot_contact(self):
        pose, goals, saved = leg_inputs()
        saved["active"][1, 2] = True
        self.assert_no_foot_contacts(pose, goals, saved)

    def test_active_feet_keep_the_measured_maximum(self):
        pose, goals, saved = leg_inputs()
        saved["active"][0, 0] = True
        saved["active"][2, 1] = True
        goals[0, 0, 1] -= 0.12
        goals[2, 1, 1] -= 0.25
        goals[1, 1, 1] -= 1.0  # An inactive target must not affect the maximum.
        output, report = solver.legs(
            dict(pose, _contact_ankle_goals=goals), pose, pose, {}, saved
        )
        expected = max(
            np.linalg.norm(output["points"][0, 7] - goals[0, 0]),
            np.linalg.norm(output["points"][2, 8] - goals[2, 1]),
        ) * 1000
        self.assertGreater(expected, 100.0)
        self.assertAlmostEqual(report["leg_ankle_target_max_mm"], expected)


if __name__ == "__main__":
    unittest.main()
