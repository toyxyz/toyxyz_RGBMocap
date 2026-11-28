from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import json
import os
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mocap import cli_workflows as workflows, project_workspace as workspace
from mocap.cli import main, open_app, parser
from mocap.cli_runner import WorkerError, run_process
from mocap.correction_settings import correction_options
from mocap.storage import atomic_json, sha256
from mocap.workflow_jobs import grouped_take_jobs, next_job, person_jobs, take_jobs, video_jobs


class SharedWorkflowTests(unittest.TestCase):
    def test_gui_single_stage_parameters_are_kept(self):
        jobs = video_jobs('video.mp4', 0.0, 3.0, 'exmr',
                         ['track', 'exmr_preprocess', 'exmr_features', 'exmr', 'correct', 'export'],
                         result='corrected_initial',
                         correction=correction_options({'auto_floor': False, 'floor': 0.2}))
        self.assertEqual(jobs[0], ('prepare', ['--video', 'video.mp4', '--start', '0.0', '--end', '3.0',
                         '--motion-tool', 'exmr', '--result', 'corrected_initial', '--floor', '0.2',
                         '--correction-preset', 'normal', '--correction-mode', 'flat', '--foot-lock', '1',
                         '--hand-lock', '0', '--pelvis-offset', '0.0']))
        stage, _, jobs = next_job(jobs[1:], 'take')
        self.assertEqual(stage, 'track')
        stage, args, jobs = next_job(jobs, 'take')
        self.assertEqual(stage, 'exmr_batch')
        self.assertEqual(args[args.index('--batch-phase') + 1], 'all')
        self.assertEqual(next_job(jobs, 'take')[1], ['--project', 'take', '--result', 'corrected_initial'])

    def test_people_keep_exclusive_end_and_all_inference_precedes_correction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            people = person_jobs(root, [1, 2], 3.0, 5.0, {'start': 2.0, 'fps': 30}, 'corrected_initial',
                                 correction_options({'auto_floor': False, 'floor': 0.2}))
            self.assertEqual(people[0][1][4:8], ['--first-frame', '30', '--end-frame', '90'])
            self.assertEqual(people[1][1][-2:], ['--floor', '0.2'])
            inference, post = take_jobs([root/'a', root/'b'],
                                        ['exmr_preprocess', 'exmr_features', 'exmr', 'correct', 'export'],
                                        'corrected_initial')
            queue = grouped_take_jobs(root, inference, [], post)
            self.assertEqual([stage for stage, _ in queue], ['exmr_batch', 'correct', 'export', 'correct', 'export'])


class BvhFloorExportTests(unittest.TestCase):
    def test_reference_and_applied_floor_in_both_profiles(self):
        import numpy as np

        from mocap.biped_bvh import bone_names
        from mocap.export_copy import run
        from mocap.export_options import read_header
        from mocap.source_bvh import write_channels

        kind = 'corrected_initial'
        with TemporaryDirectory() as directory:
            project = Path(directory)
            source = project / 'source.bvh'
            names = list(bone_names())
            rest = np.zeros((52, 3))
            rest[0, 1] = 1.
            positions = np.array([[2., .97, 4.], [2., .99, 4.1]])
            angles = np.zeros((2, 52, 3))
            angles[:, 0, 0] = [365., 366.]
            write_channels(source, names, np.array([-1] + [0] * 51), rest, np.zeros_like(rest),
                           positions, angles, ['XYZ'] * 52, 30.)
            original = source.read_bytes()
            fbx_rest = rest.copy()
            fbx_rest[[10, 11], 1] = [.017, .018]
            np.savez(project / 'rig.npz', fbx_rest=fbx_rest)
            atomic_json(project / f'{kind}_mixamo.json', {'asset_file': 'rig.npz'})
            for profile in ('generic', 'biped'):
                for auto_floor in (False, True):
                    with self.subTest(profile=profile, auto_floor=auto_floor):
                        atomic_json(project / f'{kind}_correction.json', {
                            'floor': -.023, 'segment_count': 1,
                            'options': {'mode': 'flat', 'auto_floor': auto_floor, 'foot_lock': True},
                        })
                        request = project / 'request.json'
                        target = project / f'{profile}-{auto_floor}.bvh'
                        atomic_json(request, [{
                            'project': str(project), 'source': str(source), 'target': str(target),
                            'kind': kind, 'bvh_profile': profile,
                        }])
                        with patch('mocap.export_copy.selected_export', return_value=source):
                            self.assertEqual(run(request), 1)
                        with target.open('rb') as exported:
                            header, _ = read_header(exported)
                            self.assertAlmostEqual(float(header[3].split()[2]), 1.017)
                            self.assertEqual(exported.readline().strip(), b'Frames: 2')
                            self.assertEqual(exported.readline().strip(), b'Frame Time: 0.0333333333')
                            motion = np.loadtxt(exported)
                        shift = .023 if auto_floor else 0.
                        np.testing.assert_allclose(motion[:, :3], positions + [0., shift, 0.], atol=1e-9)
                        if profile == 'generic':
                            np.testing.assert_array_equal(motion[:, 3], angles[:, 0, 0])
                        self.assertEqual(source.read_bytes(), original)


class ExmrCliTests(unittest.TestCase):
    def test_reject_different_source_and_changed_file_at_same_path(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            original = folder/'original.mp4'
            other = folder/'other.mp4'
            original.write_bytes(b'original')
            other.write_bytes(b'different')
            root = folder/'project'
            workspace.create(root, original, {'path': str(original)}, 'exmr')
            take = root/'exmr'
            take.mkdir()
            atomic_json(take/'project.json', {'pipeline': {'model': 'exmr'}, 'source_sha256': sha256(original)})
            self.assertEqual(workflows.requested_workspace(root, original, 'exmr')[0], root)
            with self.assertRaisesRegex(ValueError, 'differs'):
                workflows.requested_workspace(root, other, 'exmr')
            original.write_bytes(b'new version')
            with self.assertRaisesRegex(ValueError, 'differs'):
                workflows.requested_workspace(root, original, 'exmr')

    def test_correct_partial_option_keeps_saved_floor_and_locks(self):
        args = parser().parse_args(['correct', 'take', '--preset', 'strong'])
        existing = {'correction': dict(auto_floor=False, floor=.17, pelvis_offset=.05, foot_lock=False, hand_lock=True)}
        with patch.object(workflows, 'saved', return_value=None):
            _, result, _ = workflows.chosen_settings(args, existing)
        self.assertEqual(result, correction_options({'preset': 'strong', **existing['correction']}))

    def test_model_mismatch_is_refused_without_writes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            atomic_json(path/'project.json', {'pipeline': {'model': 'flowhmr'}})
            before = (path/'project.json').read_bytes()
            args = parser().parse_args(['correct', str(path)])
            with self.assertRaisesRegex(ValueError, 'not a saved exmr'):
                workflows.correct(args, lambda line: None)
            self.assertEqual((path/'project.json').read_bytes(), before)

    def test_export_same_display_name_is_disambiguated_by_person_id(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [('a/takes/0000_0010', root/'a', {'name': 'Actor', 'id': 1}),
                    ('b/takes/0000_0010', root/'b', {'name': 'Actor', 'id': 2})]
            args = SimpleNamespace(output=None, output_dir=root, overwrite=False)
            targets = workflows.output_paths(args, rows)
            self.assertNotEqual(targets[0], targets[1])
            targets[0].write_bytes(b'prior export')
            with self.assertRaisesRegex(ValueError, 'Destination exists'):
                workflows.output_paths(args, rows)
            self.assertEqual(targets[0].read_bytes(), b'prior export')

    @unittest.skipUnless(os.name == 'nt', 'Windows launcher')
    def test_open_passes_unicode_and_spaces_as_direct_arguments(self):
        with TemporaryDirectory() as directory:
            clip = Path(directory)/'한글 test video.mp4'
            clip.touch()
            with patch('tools.session_lock.app_lock') as lock, patch('mocap.cli.subprocess.Popen') as start:
                result = open_app(clip)
            args = start.call_args.args[0]
            self.assertEqual(Path(args[0]).name, 'python.exe')
            self.assertEqual(args[-2:], ['--video', str(clip.resolve())])
            self.assertTrue(result['launch_requested'])
            lock.assert_called_once()

    def test_runner_keeps_event_channel_separate_from_worker_logs(self):
        payload = (
            "import sys; "
            "print('@@MOCAP@@{\"type\": \"failed\", \"message\": \"not an event\"}', file=sys.stderr); "
            "print('@@MOCAP@@{\"type\": \"prepared\", \"project\": \"test\"}')"
        )
        logs = []
        events = run_process([sys.executable, '-I', '-c', payload], 'test', os.environ, logs.append, False)
        self.assertEqual(events, [{'type': 'prepared', 'project': 'test'}])
        self.assertTrue(any('not an event' in line for line in logs))


    def test_mutating_command_fails_before_runtime_setup_when_locked(self):
        output = StringIO()
        with patch('tools.session_lock.app_lock', side_effect=RuntimeError('Installation is busy')), \
                patch('tools.source_launch.verify') as verify, redirect_stdout(output):
            self.assertEqual(main(['correct', 'take', '--json']), 1)
        verify.assert_not_called()
        self.assertIn('Installation is busy', json.loads(output.getvalue())['error'])

    def test_runner_stops_a_worker_on_resource_pressure(self):
        guard = SimpleNamespace(check=lambda: 'low memory', output=lambda: None)
        with patch('mocap.cli_runner.ProcessingGuard', return_value=guard):
            with self.assertRaisesRegex(WorkerError, 'low memory'):
                run_process([sys.executable, '-I', '-c', 'import time; time.sleep(60)'],
                            'test', os.environ, lambda line: None, True)
