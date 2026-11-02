"""Windows build adaptations for separately downloaded DroidCalib/LieTorch.

The providers' sources and licenses stay in the private support installation.
These changes adapt compiler/PyTorch interfaces, not the SLAM equations.
"""
from pathlib import Path
import shutil,re


def prepare(source,engine):
    source=Path(source);engine=Path(engine)
    # Python modules use absolute imports inside the upstream droid_slam folder.
    shutil.copytree(engine/'pipeline/droidcalib/droid_slam',source/'droid_slam')
    shutil.copytree(engine/'pipeline/droidcalib/src',source/'src')
    for name in ('LICENSE','3rd-party-licenses.txt'):
        shutil.copy2(engine/'pipeline/droidcalib'/name,source/name)
    for path in source.rglob('*'):
        if path.suffix in ('.cpp','.cu'):
            text=path.read_text(encoding='utf8').replace('.type(),','.scalar_type(),')
            if path.parent==source/'src':text=re.sub(r'\blong\b','int64_t',text)
            if path.name=='lietorch_gpu.cu':
                # MSVC otherwise gives CPU templates and __global__ launch
                # stubs identical decorated names; the linker can select the
                # CPU body and dereference a CUDA pointer on the host.
                for name in re.findall(r'__global__ void (\w+)\(',text):
                    text=re.sub(r'\b'+name+r'\b','lietorch_cuda_'+name,text)
            path.write_text(text,encoding='utf8')
    masked=(engine/'pipeline/camera/masked_droid_slam.py').read_text(encoding='utf8')
    # Windows worker spawning would duplicate the parent/video/model memory.
    masked=masked.replace('num_workers=8 if os.cpu_count() > 8 else os.cpu_count()', 'num_workers=0')
    # Bound Metric3D activation storage without changing its resolution/precision.
    masked=masked.replace('batch_size = 8','batch_size = 1').replace('batch_size = 32','batch_size = 1')
    (source/'masked_droid_slam.py').write_text(masked,encoding='utf8')
    (source/'setup_windows.py').write_text('''from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
from pathlib import Path
import os
S=Path(__file__).resolve().parent
os.chdir(S)
args={'cxx':['/O2','/MD','/GL-','/wd4819'], 'nvcc':['-O2','-Xcompiler=/MD','-Xcompiler=/GL-']}
extensions=[CUDAExtension('lietorch_backends',include_dirs=[str(S/'thirdparty/lietorch/lietorch/include'),str(S/'thirdparty/eigen')],sources=['thirdparty/lietorch/lietorch/src/'+n for n in ('lietorch.cpp','lietorch_gpu.cu','lietorch_cpu.cpp')],extra_compile_args=args),
CUDAExtension('droid_backends_intr',include_dirs=[str(S/'thirdparty/eigen')],sources=['src/'+n for n in ('droid.cpp','droid_kernels.cu','correlation_kernels.cu','altcorr_kernel.cu')],extra_compile_args=args)]
for e in extensions:e.library_dirs.append(str(S/'torch-libs'))
setup(name='rgbmocap-prompthmr-camera',version='0.1',ext_modules=extensions,cmdclass={'build_ext':BuildExtension})
''',encoding='utf8')
