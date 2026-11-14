@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
echo MAMMA optional runtime setup. Close RGB MoCap before continuing.
echo This installs a separate Python 3.11 environment. Models are obtained separately.
echo Allow about 18 GiB free for installation, excluding models and compiler tools.
echo Native compilation needs Visual Studio 2022 C++ Build Tools and CUDA Toolkit 12.8.
echo Review https://github.com/cuevhv/mamma/blob/main/LICENSE and
echo https://github.com/cuevhv/pytorch_sdf/blob/main/LICENSE before proceeding.
echo Restricted components are downloaded from upstream for your permitted local use.
choice /c YN /n /m "I reviewed these licenses and agree to the permitted use. Continue? [Y/N] "
if not "%errorlevel%"=="1" exit /b 1
if not exist "runtime\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
"runtime\python.exe" -I -X utf8 -B tools\setup_mamma.py --accept-licenses
set "mamma_result=%errorlevel%"
pause
exit /b %mamma_result%
