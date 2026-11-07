"""Build the pinned Kimodo CPU backend; Windows users receive the built module, Linux builds it at install."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = os.name == 'nt'
SUFFIX = '.pyd' if WINDOWS else '.so'

def build(build_dir, output_dir, python=sys.executable, deps=ROOT/'.build-inputs/native-deps'):
    """Configure and build _motion_correction for `python`; return the copied module paths."""
    cmake=shutil.which('cmake')
    if not cmake and WINDOWS:
        candidates=list(Path('C:/Program Files (x86)/Microsoft Visual Studio').glob('*/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'))
        if candidates:cmake=str(candidates[-1])
    if not cmake:raise RuntimeError('Install CMake (and a C++17 compiler) to build motion correction')
    source=ROOT/'third_party/KimodoMotionCorrection'
    build=Path(build_dir).resolve()
    build.mkdir(parents=True,exist_ok=True)
    # Another project's registered Eigen/pybind11 must never stand in for the pinned sources.
    command=[cmake,'-S',str(source),'-B',str(build),'-DPython3_EXECUTABLE='+str(python),'-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON',
             '-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF','-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF']
    if WINDOWS:command+=['-G','Visual Studio 17 2022','-A','x64','-DCMAKE_CXX_FLAGS=/utf-8 /EHsc /wd4127 /wd4267']
    # A reused cache makes pybind11's subproject return early; reconfigure fresh over the fetched sources.
    else:command+=['--fresh','-DCMAKE_BUILD_TYPE=Release']
    if (deps/'pybind11/share/cmake/pybind11').exists():
        command+=['-Dpybind11_DIR='+str((deps/'pybind11/share/cmake/pybind11').resolve())]
    elif not WINDOWS:command+=['-DCMAKE_DISABLE_FIND_PACKAGE_pybind11=ON']
    if (deps/'eigen-3.4.0').exists():
        command+=['-DFETCHCONTENT_SOURCE_DIR_EIGEN='+str((deps/'eigen-3.4.0').resolve())]
    elif not WINDOWS:command+=['-DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=ON']
    log_path=build/'build.log'
    with log_path.open('w',encoding='utf-8') as log:
        try:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            subprocess.run([cmake,'--build',str(build),'--config','Release','--target','_motion_correction','--parallel','2'],check=True,stdout=log,stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            raise RuntimeError('Motion correction build failed; see '+str(log_path)) from None
    built=list((build/'Release' if WINDOWS else build).glob('_motion_correction*'+SUFFIX))
    if len(built)!=1:raise RuntimeError('Expected one built motion correction module in '+str(build))
    target=Path(output_dir).resolve();target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(built[0],target/built[0].name)
    return [target/built[0].name]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--deps', type=Path, default=ROOT/'.build-inputs/native-deps')
    parser.add_argument('--build-dir',type=Path,default=ROOT/'build/motion-correction')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'third_party/KimodoMotionCorrection/bin')
    args=parser.parse_args()
    for path in build(args.build_dir,args.output_dir,deps=args.deps):print(path)

if __name__=='__main__':main()
