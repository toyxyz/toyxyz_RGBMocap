import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from mocap.first_person.arm_ik import estimate_lengths
from mocap.first_person.runtime import load_settings


class FirstPersonTests(unittest.TestCase):
    def test_cli_lists_first_person_models(self):
        from mocap.cli import parser

        args = parser().parse_args(["models", "list", "--workflow", "vidihand2"])
        self.assertEqual(args.workflow, "vidihand2")

    def test_runtime_reports_a_non_text_wsl_distribution(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "first-person-arms.json"
            settings = {
                "mode": "wsl",
                "distribution": "Ubuntu",
                "python": "/home/user/ViDiHand/.venv/bin/python",
                "upstream": "/home/user/ViDiHand",
            }
            path.write_text(json.dumps(settings), encoding="utf-8")
            self.assertEqual(load_settings(path), settings)
            for invalid in (None, {}, ["Ubuntu"], "Ubuntu other"):
                with self.subTest(distribution=invalid):
                    path.write_text(json.dumps({**settings, "distribution": invalid}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Invalid WSL distribution name"):
                        load_settings(path)

    def test_shared_arm_length_uses_sustained_reach_at_end_of_take(self):
        distances = np.full((300, 2), 0.58)
        tracked = np.ones_like(distances, dtype=bool)
        original = estimate_lengths(distances, tracked)
        distances[100, 0] = 2.0
        self.assertEqual(estimate_lengths(distances, tracked), original)
        distances[-12:, 1] = 0.73
        self.assertGreaterEqual(sum(estimate_lengths(distances, tracked)), 0.73)
