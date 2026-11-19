import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

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
