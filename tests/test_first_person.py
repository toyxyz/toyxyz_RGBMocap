import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from mocap.first_person.arm_ik import estimate_lengths
from mocap.first_person.runtime import command, load_chumpy, load_settings, source_path


class FirstPersonTests(unittest.TestCase):
    def test_managed_settings_use_the_flow_profile_and_pinned_source(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            settings = root / ".setup" / "first-person-arms.json"
            with self.assertRaisesRegex(FileNotFoundError, "install.cmd"):
                load_settings(settings, root=root)
            python = root / "runtime" / "python.exe"
            python.parent.mkdir()
            python.touch()
            (root / "runtime_flow").mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "Setup_ViDiHand2.cmd"):
                load_settings(settings, root=root)

            source = source_path(root)
            for name in ("vidihand_v2/inference.py", "vidihand_v2/config.yaml", "utils/io.py", "utils/mano.py"):
                entry = source / name
                entry.parent.mkdir(parents=True, exist_ok=True)
                entry.touch()
            managed = load_settings(settings, root=root)
            self.assertEqual(managed, {"mode": "managed", "python": str(python), "upstream": str(source)})
            options = command(managed, Path("clip.mp4"), None, Path("out.npz"), Path("cancel"),
                              refit_from=Path("old.npz"))
            self.assertEqual(options[0], str(python))
            self.assertEqual(options[options.index("--upstream") - 1], "managed")

            settings.write_text(json.dumps({"mode":"wsl", "distribution":"Ubuntu",
                                            "python":"/venv/bin/python", "upstream":"/src/ViDiHand"}),
                                encoding="utf-8")
            self.assertEqual(load_settings(settings, root=root)["mode"], "wsl")
            settings.write_text('{"mode":"managed"}', encoding="utf-8")
            self.assertEqual(load_settings(settings, root=root), managed)

    def test_chumpy_import_preserves_existing_numpy_aliases(self):
        keys = ("bool", "int", "float", "complex", "object", "str", "unicode")
        before = {key: vars(np)[key] for key in keys if key in vars(np)}

        def import_chumpy(name):
            self.assertEqual(name, "chumpy")
            for alias in keys:
                self.assertIn(alias, vars(np))
            raise ImportError("broken dependency")

        with patch("mocap.first_person.runtime.importlib.import_module", side_effect=import_chumpy):
            with self.assertRaisesRegex(ImportError, "broken dependency"):
                load_chumpy()
        self.assertEqual({key: vars(np)[key] for key in keys if key in vars(np)}, before)

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
