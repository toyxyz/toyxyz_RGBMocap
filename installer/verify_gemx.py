"""Verify the optional environment before setup publishes its registration."""

import importlib.metadata
import json
import os
from pathlib import Path
import sys

folder, lock, report = map(Path, sys.argv[1:])
os.environ['WARP_CACHE_PATH'] = str(report.parent.parent / 'tmp/gemx-warp')
if sys.version_info[:3] != (3, 12, 10) or sys.prefix == sys.base_prefix:
    raise RuntimeError('Expected isolated Python 3.12.10.')
for row in json.loads(lock.read_text(encoding='utf-8'))['packages']:
    if importlib.metadata.version(row['name']) != row['version']:
        raise RuntimeError('Incorrect installed package: ' + row['name'])
sys.path.insert(0, str(folder / 'engine'))
import torch
import onnxruntime
import warp
from gem.inference.body_video import load_body_video_config
from gem.motion.geometry import MotionGeometry

if not torch.cuda.is_available() or 'CPUExecutionProvider' not in onnxruntime.get_available_providers():
    raise RuntimeError('Expected CUDA and the ONNX CPU execution provider.')
sample = torch.arange(16, dtype=torch.float32, device='cuda').reshape(4, 4)
expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
torch.testing.assert_close((sample @ sample).cpu(), expected @ expected)
warp.init()
if not warp.is_cuda_available():
    raise RuntimeError('SOMA-X Warp could not initialize CUDA. Check for an NVIDIA R580-series or newer driver.')


@warp.kernel
def increment(values: warp.array(dtype=warp.float32)):
    values[warp.tid()] += 1.0


values = warp.zeros(4, dtype=warp.float32, device='cuda:0')
warp.launch(increment, dim=4, inputs=[values], device='cuda:0')
if values.numpy().tolist() != [1.] * 4:
    raise RuntimeError('SOMA-X Warp CUDA kernel verification failed.')
config = load_body_video_config(folder / 'engine/docs/configs/body_video_native77.yaml')
if config['motion']['network']['obs_num_joints'] != 77 or MotionGeometry is None:
    raise RuntimeError('GEM-X Native77 is unavailable.')
report.write_text(json.dumps(dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                                  cuda=list(torch.cuda.get_device_capability()))), encoding='utf-8')
