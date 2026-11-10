@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist "runtime_exmr\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
echo Optional SAM 3.1 setup. Close RGB MoCap before continuing.
echo The setup reuses the existing ExMR Python and CUDA runtime.
echo Review the SAM terms at https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/LICENSE
choice /c YN /n /m "I reviewed the upstream terms. Continue? [Y/N] "
if not "%errorlevel%"=="1" exit /b 1
"runtime_exmr\python.exe" -I -X utf8 -B tools\setup_sam31.py
set "sam_result=%errorlevel%"
pause
exit /b %sam_result%
