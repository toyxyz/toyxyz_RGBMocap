"""FlowHMR uses scene intrinsics for moving cameras while Static Camera stays independent."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mocap import flow_backend


class FlowIntrinsicsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = {
            'frames': 4, 'fps': 30, 'width': 1920, 'height': 1080,
            'flow_options': {'camera_interval': 0},
        }
        (self.root / 'people.json').write_text('{}', encoding='utf-8')
        self.manifest(self.root, self.state)

    def manifest(self, path, state):
        path.mkdir(exist_ok=True)
        (path / 'project.json').write_text(json.dumps(state), encoding='utf-8')

    def save_camera(self, path, focal, **metadata):
        k = np.repeat(np.eye(3)[None], len(focal), axis=0)
        k[:, 0, 0] = focal
        k[:, 1, 1] = np.asarray(focal) + 20
        k[:, 0, 2] = 962
        k[:, 1, 2] = 538
        np.savez(path / 'flow_camera.npz', K=k, **metadata)

    def test_full_scene_median_and_feature_cache_are_shared_across_takes(self):
        self.save_camera(self.root, [900., 1200., -1., 1500.],
                         k_space='original_frame', image_size_hw=[1080, 1920])
        expected = [[1200., 0., 962.], [0., 1220., 538.], [0., 0., 1.]]
        takes = []
        for index, interval in enumerate(((0, 2), (2, 4))):
            take = self.root / f'take-{index}'
            self.manifest(take, {**self.state, 'frames': 2, 'people_root': '..',
                                 'person_frames': interval, 'flow_options': {'camera_interval': -1}})
            self.save_camera(take, [800 + index, 800 + index])
            takes.append(take)
        signatures = [flow_backend.identity(take, 'flow_features') for take in takes]
        self.assertEqual([sig['camera_intrinsics'] for sig in signatures], [expected, expected])
        self.save_camera(self.root, [950., 1200., -1., 1450.])
        self.assertEqual([flow_backend.identity(take, 'flow_features') for take in takes], signatures)
        self.save_camera(self.root, [1000., 1300., -1., 1500.])
        changed = [flow_backend.identity(take, 'flow_features') for take in takes]
        self.assertEqual([sig['camera_intrinsics'][0][0] for sig in changed], [1300., 1300.])
        self.assertNotEqual(changed, signatures)
        self.assertEqual([sig['inputs'] for sig in changed], [sig['inputs'] for sig in signatures])

    def test_direct_inference_rejects_features_with_old_intrinsics(self):
        self.save_camera(self.root, [1200.] * 4)
        expected = flow_backend._moving_camera_intrinsics(self.root, self.state)
        placement = self.root / 'flow_placement.json'
        with self.assertRaisesRegex(ValueError, 'Run Reconstruct 3D'):
            flow_backend._require_feature_intrinsics(self.root, expected)
        placement.write_text(json.dumps({'intrinsics': np.eye(3).tolist()}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Run Reconstruct 3D'):
            flow_backend._require_feature_intrinsics(self.root, expected)
        placement.write_text(json.dumps({'intrinsics': expected.tolist()}), encoding='utf-8')
        flow_backend._require_feature_intrinsics(self.root, expected)

    def test_static_needs_no_vggt_archive(self):
        self.state['flow_options']['camera_interval'] = -1
        self.manifest(self.root, self.state)
        signature = flow_backend.identity(self.root, 'flow_features')
        self.assertNotIn('camera_intrinsics', signature)
        self.assertFalse((self.root / 'flow_camera.npz').exists())

    def test_camera_opt_separates_moving_cache_and_leaves_static_alone(self):
        (self.root / 'input.mp4').touch()
        with patch('mocap.vggt_camera_opt.available', return_value=True):
            default = flow_backend.identity(self.root, 'flow_camera')
            self.assertTrue(default['vggt_opt'])
            self.state['flow_options']['vggt_opt'] = False
            self.manifest(self.root, self.state)
            standard = flow_backend.identity(self.root, 'flow_camera')
            self.assertFalse(standard['vggt_opt'])
            self.assertNotEqual(default, standard)
        with patch('mocap.vggt_camera_opt.available', return_value=False):
            self.state['flow_options']['vggt_opt'] = True
            self.manifest(self.root, self.state)
            self.assertEqual(standard, flow_backend.identity(self.root, 'flow_camera'))

        self.state['flow_options']['camera_interval'] = -1
        self.manifest(self.root, self.state)
        static = flow_backend.identity(self.root, 'flow_camera')
        self.state['flow_options']['vggt_opt'] = False
        self.manifest(self.root, self.state)
        self.assertEqual(static, flow_backend.identity(self.root, 'flow_camera'))

    def test_missing_or_incomplete_moving_camera(self):
        with self.assertRaisesRegex(ValueError, 'Run camera estimation again'):
            flow_backend.identity(self.root, 'flow_features')
        self.save_camera(self.root, [1200.] * 4)
        pending = self.root / 'flow_camera_pending.json'
        pending.touch()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            flow_backend.identity(self.root, 'flow_features')

    def test_invalid_moving_camera_metadata(self):
        for metadata in ({'k_space': 'network_input'}, {'image_size_hw': [512, 512]}):
            with self.subTest(metadata=metadata):
                self.save_camera(self.root, [1200.] * 4, **metadata)
                with self.assertRaisesRegex(ValueError, 'valid VGGT camera intrinsics'):
                    flow_backend.identity(self.root, 'flow_features')
        self.save_camera(self.root, [1200.] * 3)
        with self.assertRaisesRegex(ValueError, 'scene frame count'):
            flow_backend.identity(self.root, 'flow_features')
        self.save_camera(self.root, [0., -1., np.nan, np.inf])
        with self.assertRaisesRegex(ValueError, 'no valid original-frame intrinsics'):
            flow_backend.identity(self.root, 'flow_features')


if __name__ == '__main__':
    unittest.main()
