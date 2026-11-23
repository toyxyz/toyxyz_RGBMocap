"""Headless command line: the same worker queue, arguments and settings as the GUI, without starting workers."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mocap import cli, paths


class Recorder:
    def __init__(self, project):
        self.project = project
        self.calls = []

    def __call__(self, stage, arguments, env, console):
        self.calls.append((stage, list(arguments)))
        if stage == 'prepare':
            return [dict(type='prepared', project=str(self.project))]
        return [dict(type='complete', stage=stage)]


class CommandLineTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        for target, value in ((paths, 'HOME'),):
            patcher = patch.object(target, value, self.home)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.project = self.home / 'projects/clip_0123_project/exmr'
        self.video = self.home / 'clip.mp4'
        for name, value in (('video_range', lambda video, start, end: (0.0, 10.033333333333333)),
                            ('missing_models', lambda stages: [])):
            patcher = patch.object(cli, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_main(self, *argv):
        recorder = Recorder(self.project)
        with patch.object(cli, 'run_stage', recorder):
            self.assertEqual(cli.main(['--quiet', *argv]), 0)
        return recorder.calls

    def test_reconstruct_queue_matches_gui(self):
        calls = self.run_main('reconstruct', str(self.video))
        project = str(self.project)
        self.assertEqual([stage for stage, _ in calls], ['prepare', 'track', 'exmr_batch', 'correct', 'export'])
        self.assertEqual(calls[0][1], ['--video', str(self.video.resolve()), '--start', '0.0', '--end', '10.033333333333333',
                                       '--motion-tool', 'exmr', '--result', 'corrected_initial',
                                       '--correction-preset', 'normal', '--correction-mode', 'flat',
                                       '--foot-lock', '1', '--hand-lock', '0', '--pelvis-offset', '0.0'])
        self.assertEqual(calls[1][1], ['--project', project])
        batch = calls[2][1]
        self.assertEqual(batch[batch.index('--batch-phase') + 1], 'all')
        self.assertEqual(json.loads(batch[batch.index('--batch-items') + 1]),
                         [dict(project=project, result='auto', force=False)])
        for stage, arguments in calls[3:]:
            self.assertEqual(arguments, ['--project', project, '--result', 'corrected_initial'])

    def test_without_correction_exports_initial_motion(self):
        calls = self.run_main('reconstruct', str(self.video), '--no-correction', '--moving-camera')
        self.assertEqual([stage for stage, _ in calls], ['prepare', 'track', 'exmr_batch', 'export'])
        self.assertNotIn('--correction-preset', calls[0][1])
        self.assertEqual(calls[-1][1], ['--project', str(self.project), '--result', 'initial'])

    def test_manual_floor_is_passed_like_the_gui(self):
        calls = self.run_main('reconstruct', str(self.video), '--floor', '0.25', '--preset', 'strong')
        prepare = calls[0][1]
        self.assertEqual(prepare[prepare.index('--floor') + 1], '0.25')
        self.assertEqual(prepare[prepare.index('--correction-preset') + 1], 'strong')

    def test_worker_environment_matches_gui_defaults(self):
        env = cli.worker_environment()
        self.assertEqual(json.loads(env['RGB_MOCAP_EXMR_OPTIONS']), dict(static_camera=True, smoothing='global'))
        self.assertEqual(env['RGB_MOCAP_COMPUTE_MODE'], 'auto')
        self.assertEqual(env['RGB_MOCAP_RESOURCE_SAFETY'], '1')

    def test_saved_starting_settings_apply_and_flags_win(self):
        settings = self.home / 'config/local.json'
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps(dict(resource_safety=False, default_settings=dict(
            exmr_options=dict(static_camera=False, smoothing='none'),
            correction=dict(mode='multi_level', preset='weak', foot_lock=False, hand_lock=True),
            processing_options=dict(resource_mode='low')))), encoding='utf-8')
        defaults = cli.starting_settings()
        self.assertEqual(defaults, dict(moving_camera=True, no_smoothing=True, mode='multi_level', preset='weak',
                                        foot_lock=0, hand_lock=1, compute='low'))
        calls = self.run_main('reconstruct', str(self.video), '--preset', 'normal')
        prepare = calls[0][1]
        self.assertEqual(prepare[prepare.index('--correction-mode') + 1], 'multi_level')
        self.assertEqual(prepare[prepare.index('--correction-preset') + 1], 'normal')
        self.assertEqual(prepare[prepare.index('--foot-lock') + 1], '0')
        self.assertEqual(cli.worker_environment()['RGB_MOCAP_RESOURCE_SAFETY'], '0')

    def test_missing_models_stop_before_any_worker(self):
        with patch.object(cli, 'missing_models', lambda stages: ['SMPL-X Neutral']):
            recorder = Recorder(self.project)
            with patch.object(cli, 'run_stage', recorder):
                self.assertEqual(cli.main(['--quiet', 'reconstruct', str(self.video)]), 1)
        self.assertEqual(recorder.calls, [])


if __name__ == '__main__':
    unittest.main()
