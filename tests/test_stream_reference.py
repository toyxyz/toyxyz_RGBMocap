"""Reference export must preserve existing files on invalid input/failure."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from mocap.stream_reference import run,validate
from mocap.blend_support import partial


class ReferenceExportTest(unittest.TestCase):
    def schema(self):
        return dict(names=['Hips','Child'],parents=[-1,0],
                    rest=[[0,90,0,0,0,0,1],[0,10,0,0,0,0,1]],lengths=[10,5])

    def request(self,folder,schema=None):
        target=folder/'reference.fbx'
        request=folder/'reference-request.json'
        target.write_bytes(b'previous export')
        request.write_text(json.dumps(dict(target=str(target),schema=schema or self.schema())),encoding='utf-8')
        return request,target

    def test_hierarchy_and_finite_validation(self):
        schema=self.schema();schema['parents']=[-1,1]
        with self.assertRaises(ValueError):validate(schema)
        schema=self.schema();schema['rest'][0][0]=float('nan')
        with self.assertRaises(ValueError):validate(schema)
        schema=self.schema();schema['names'][1]='hips'
        with self.assertRaises(ValueError):validate(schema)
        schema=self.schema();schema['rest'][0][6]=2
        with self.assertRaises(ValueError):validate(schema)

    def test_failed_writer_retains_existing_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            request,target=self.request(Path(tmp))
            with patch('mocap.blend_support.build',side_effect=RuntimeError('writer failed')):
                with self.assertRaises(RuntimeError):run(request)
            self.assertEqual(target.read_bytes(),b'previous export')
            self.assertFalse(partial(target,request).exists())
            self.assertFalse(request.exists())

    def test_invalid_rig_retains_existing_export(self):
        schema=self.schema();schema['lengths'][1]=0
        with tempfile.TemporaryDirectory() as tmp:
            request,target=self.request(Path(tmp),schema)
            with self.assertRaises(ValueError):run(request)
            self.assertEqual(target.read_bytes(),b'previous export')
            self.assertFalse(partial(target,request).exists())

    def test_publish_only_completed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            request,target=self.request(Path(tmp))
            def build(work,*args):
                self.assertEqual(target.read_bytes(),b'previous export')
                self.assertTrue((work/'source.fbx').is_file())
                (work/'reference.fbx').write_bytes(b'completed reference')
            with patch('mocap.blend_support.build',side_effect=build):run(request)
            self.assertEqual(target.read_bytes(),b'completed reference')
            self.assertFalse(partial(target,request).exists())
            self.assertFalse(request.exists())


if __name__=='__main__':unittest.main()
