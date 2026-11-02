@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
echo PromptHMR optional Windows runtime setup. Close RGB MoCap first.
echo Separate Python 3.11, Torch 2.7.1 CUDA 12.8. Models are obtained separately.
echo Allow about 18 GiB free for setup, excluding models and compiler tools.
echo Native SAM2 build needs Visual Studio 2022 C++ Build Tools and CUDA Toolkit 12.8.
echo Review https://github.com/yufu-wang/PromptHMR/blob/main/LICENSE
echo Non-commercial scientific research use only; other dependencies have separate licenses.
choice /c YN /n /m "I reviewed these licenses and agree to the permitted use. Continue? [Y/N] "
if errorlevel 2 exit /b 1
echo Optional moving-camera support uses DroidCalib [AGPL-3.0], LieTorch and Metric3D.
echo Review https://github.com/boschresearch/DroidCalib/blob/main/LICENSE
echo and https://github.com/YvanYin/Metric3D/blob/main/LICENSE before choosing Y.
echo Camera support reuses Python/Torch; allow an additional 6 GiB for building,
echo plus 1.55 GiB for the two camera models obtained separately in Model Manager.
set "phmr_camera="
choice /c YN /n /m "Prepare moving-camera support as well? [Y/N] "
if not errorlevel 2 set "phmr_camera=--camera"
if not exist "runtime\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
"runtime\python.exe" -I -X utf8 -B tools\setup_prompthmr.py --accept-licenses %phmr_camera%
set "phmr_result=%errorlevel%"
pause
exit /b %phmr_result%
