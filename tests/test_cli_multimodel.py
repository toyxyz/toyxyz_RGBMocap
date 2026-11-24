"""CLI coverage for dedicated requests and multi-model operations."""
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mocap.cli import parser
from mocap.cli_tools import batch
from mocap.cli_workflows import camera_refinement_enabled, correct, export, output_paths, reconstruction_result, stage_settings
from mocap.offline_jobs import write_jobs
from mocap.project_workspace import create, read, set_active
from mocap.storage import atomic_json
from mocap.workflow_jobs import sam31_arguments


class MultiModelCLI(unittest.TestCase):
    def test_camera_progress_emits_when_the_progress_file_is_locked(self):
        from contextlib import redirect_stdout
        from mocap.camera_background_worker import publish_progress

        with TemporaryDirectory() as folder, redirect_stdout(StringIO()) as output:
            work = Path(folder)
            previous = {'current': 1, 'total': 3}
            progress = work/'progress.json'
            progress.write_text(json.dumps(previous), encoding='utf8')
            with patch('mocap.camera_background_worker.os.replace', side_effect=PermissionError):
                publish_progress(work, 2, 3)
            self.assertEqual(json.loads(progress.read_text('utf8')), previous)
            self.assertEqual(json.loads(output.getvalue().removeprefix('@@BACKGROUND@@')), {'current': 2, 'total': 3})
            publish_progress(work, 3, 3)
            self.assertEqual(json.loads(progress.read_text('utf8')), {'current': 3, 'total': 3})

    def test_camera_refinement_is_saved_only_in_the_requested_model_view(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            create(root, root/'clip.mp4', {}, 'flowhmr')
            project = root/'flowhmr'
            project.mkdir()
            state = dict(workspace_root='..', pipeline={'model': 'flowhmr'})
            atomic_json(project/'project.json', state)
            set_active(root, 'exmr', view={'source_time': 2, 'camera_background': {'exmr': True}})
            original_view = {'source_time': 1, 'camera_background': {'flowhmr': False, 'exmr': True}}
            set_active(root, 'flowhmr', project, view=original_view)
            tool = parser()
            for switch, enabled in (('--refine-camera-rotation',True),('--no-refine-camera-rotation',False)):
                args = tool.parse_args(['reconstruct','clip.mp4','--model','flowhmr',switch])
                self.assertEqual(reconstruction_result(args, project)['refine_camera_rotation'], enabled)
                self.assertEqual(camera_refinement_enabled(project, 'flowhmr', None), enabled)
                self.assertEqual(camera_refinement_enabled(project, 'flowhmr', not enabled), not enabled)
                saved = read(root)
                self.assertEqual(saved['view']['source_time'], 1)
                self.assertTrue(saved['view']['camera_background']['exmr'])
                self.assertEqual(saved['models']['exmr']['view']['source_time'], 2)
                self.assertEqual(saved['models']['flowhmr']['active'], 'flowhmr')
                self.assertEqual(json.loads((project/'project.json').read_text('utf8')), state)

    def test_camera_refinement_requires_an_animated_blender_camera(self):
        tool = parser()
        for options, message in (
            ([], 'requires --format blend'),
            (['--format','blend','--no-camera-animation'], 'camera animation enabled'),
            (['--format','blend','--model','gvhmr'], 'available for FlowHMR and ExMR'),
        ):
            with self.subTest(options=options):
                args = tool.parse_args(['export','project','--output','motion',
                                        '--refine-camera-rotation',*options])
                with self.assertRaisesRegex(ValueError, message):
                    export(args, lambda message: None)

    def test_flow_camera_intervals_match_the_inference_worker(self):
        from mocap.flow_backend import camera_interval

        tool = parser()
        for interval in range(-1, 31):
            args = ['reconstruct', 'x.mp4', '--model', 'flowhmr', '--camera-interval', str(interval)]
            with self.subTest(interval=interval):
                try:
                    camera_interval(72, interval)
                except ValueError:
                    with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                        tool.parse_args(args)
                    self.assertEqual(error.exception.code, 2)
                else:
                    self.assertEqual(tool.parse_args(args).camera_interval, interval)

    def test_flow_static_and_gvhmr_refinements_use_existing_worker_names(self):
        tool = parser()
        with patch('mocap.cli_workflows.saved', return_value=None):
            flow = tool.parse_args(['reconstruct','x.mp4','--model','flowhmr','--camera','static','--hamer'])
            stages, result, options = stage_settings(flow, {}, True)
            self.assertEqual(stages[-2:], ['correct','flow_export'])
            self.assertEqual(result, 'corrected_initial')
            self.assertEqual(options['camera_interval'], -1)
            self.assertTrue(options['hamer'])
            gv = tool.parse_args(['reconstruct','x.mp4','--model','gvhmr','--camera','static','--no-footmr'])
            stages, result, _ = stage_settings(gv, {}, False)
            self.assertEqual(result, 'handflow_refined')
            self.assertIn('handflow', stages)
            self.assertNotIn('footmr', stages)
            gv.camera = 'moving'
            with self.assertRaisesRegex(ValueError, 'fixed camera'):
                stage_settings(gv, {}, False)

    def test_gui_and_cli_share_exact_dedicated_request(self):
        with TemporaryDirectory() as folder, patch('mocap.offline_jobs.HOME',Path(folder)):
            path, jobs = write_jobs('mamma', {'root':'workspace','inputs':[]})
            self.assertEqual(json.loads(path.read_text(encoding='utf8')), dict(version=1,root='workspace',inputs=[]))
            self.assertEqual([stage for stage, _ in jobs],
                             ['mamma_prepare','mamma_masks','mamma_landmarks','mamma_camera','mamma_fit','mamma_publish'])
            self.assertTrue(all(flags==['--request',str(path)] for _, flags in jobs))

    def test_sam31_arguments_keep_prompt_anchor_and_points(self):
        flags=sam31_arguments('person',1.5,[(.25,.5)])
        self.assertEqual(flags[0],'--sam31-options')
        self.assertEqual(json.loads(flags[1]), dict(prompt='person',time=1.5,points=[[.25,.5]]))

    def test_successful_correction_returns_result_after_queue_is_consumed(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'hmr4d_results.pt').touch()
            options=parser().parse_args(['correct',folder,'--model','gvhmr'])
            fake=([('single',root,{'id':1,'name':'Person 1'})])
            with patch('mocap.cli_workflows.select_takes',return_value=(root,None,fake)), \
                    patch('mocap.cli_workflows.manifest',return_value={'pipeline':{'model':'gvhmr','result':'initial'}}), \
                    patch('mocap.cli_workflows.result_is_current',return_value=True), \
                    patch('mocap.cli_workflows.run_jobs',side_effect=lambda jobs,*args:jobs.clear()):
                self.assertEqual(correct(options,lambda message:None)['result'],'corrected_initial')

    def test_batch_plan_defers_only_matching_dependent_export(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            project=root/'project'
            queue=root/'queue.json'
            queue.write_text(json.dumps({'version':1,'jobs':[
                ['correct',str(project)],
                ['export',str(project),'--output',str(root/'future.fbx')],
                ['export',str(root/'other'),'--output',str(root/'other.fbx')],
            ]}),encoding='utf8')
            options=parser().parse_args(['batch',str(queue),'--dry-run','--keep-going'])
            def executor(job,report):
                if job.command=='correct':
                    return {'project':str(project),'dry_run':True}
                raise ValueError('not ready')
            with patch('mocap.cli_workflows.execute',side_effect=executor):
                result=batch(options,lambda message:None)
            self.assertEqual(result['jobs'][1]['depends_on_job'],1)
            self.assertFalse(result['jobs'][2]['ok'])
            self.assertFalse((root/'future.fbx').exists())

    def test_combined_blender_exports_one_new_bundle(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            results=[('t1',root,{'id':1,'name':'a'}),('t2',root,{'id':2,'name':'b'})]
            args=SimpleNamespace(output=root/'motion.blend',output_dir=None,format='blend',combined=True,
                                 whole_scene=False,overwrite=False)
            self.assertEqual(output_paths(args,results),[root/'motion/motion.blend']*2)
            (root/'motion').mkdir()
            with self.assertRaisesRegex(ValueError,'new Blender bundle'):
                output_paths(args,results)
