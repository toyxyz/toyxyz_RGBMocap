import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class LinuxSetupTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'requires POSIX shell')
    def test_windows_install_is_rejected_before_setup_state_is_created(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'tools').mkdir()
            for name in ('git_bootstrap.sh', 'posix_install_guard.sh'):
                shutil.copyfile(ROOT / 'tools' / name, root / 'tools' / name)
            (root / 'runtime').mkdir()
            (root / 'runtime/python.exe').touch()
            for source in ('install.sh', 'update.sh'):
                with self.subTest(source=source):
                    target = root / source
                    shutil.copyfile(ROOT / source, target)
                    result = subprocess.run(['sh', str(target)], capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('contains a Windows RGB MoCap installation', result.stderr)
                    self.assertFalse((root / '.setup').exists())

    @unittest.skipUnless(os.name == 'posix', 'requires POSIX Blender paths')
    def test_linux_blender_override_precedes_managed_blender(self):
        from mocap.retarget import runtime

        with TemporaryDirectory() as temp:
            root = Path(temp)
            managed = root / 'runtime_character/blender-linux/blender'
            external = root / 'external/blender'
            for executable in (managed, external):
                executable.parent.mkdir(parents=True)
                executable.touch()
            with patch.object(runtime, 'ROOT', root), patch.object(runtime.shutil, 'which', return_value=None):
                for override, expected in ((external, external), (external.parent, external),
                                           (root / 'missing', managed), ('', managed)):
                    with self.subTest(override=override), patch.dict(os.environ, RGBMOCAP_BLENDER=str(override)):
                        self.assertEqual(runtime.blender(), expected.resolve())

    @unittest.skipUnless(os.name == 'posix', 'requires POSIX Blender paths')
    def test_linux_blender_ignores_working_directory_without_override(self):
        from mocap.retarget import runtime

        with TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'blender').touch()
            path_blender = folder / 'path-blender'
            path_blender.touch()
            old_dir = Path.cwd()
            try:
                os.chdir(folder)
                with patch.object(runtime, 'ROOT', folder), patch.dict(os.environ, RGBMOCAP_BLENDER=''), \
                        patch.object(runtime.shutil, 'which', return_value=str(path_blender)):
                    self.assertEqual(runtime.blender(), path_blender.resolve())
            finally:
                os.chdir(old_dir)

    @unittest.skipUnless(os.name == 'posix', 'requires POSIX private-memory query')
    def test_private_memory_permission_error_marks_metric_unavailable(self):
        import psutil
        from mocap.process_memory import private_bytes

        process = Mock()
        process.memory_full_info.return_value.uss = 4096
        self.assertEqual(private_bytes(process), 4096)
        process.memory_full_info.side_effect = psutil.AccessDenied()
        self.assertIsNone(private_bytes(process))

    def test_download_preserves_registered_in_place_file(self):
        from mocap import models

        with TemporaryDirectory() as temp:
            source = Path(temp) / 'model.ckpt'
            data = b'external model owned by the user'
            source.write_bytes(data)
            with patch.dict(models.MODEL_FILES, {'gvhmr': source}):
                with self.assertRaisesRegex(RuntimeError, 'registered in place'):
                    models.download('gvhmr')
            self.assertEqual(source.read_bytes(), data)
            self.assertEqual(list(source.parent.iterdir()), [source])


if __name__ == '__main__':
    unittest.main()
