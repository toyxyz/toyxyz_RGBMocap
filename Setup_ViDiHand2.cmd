@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
echo ViDiHand2 source setup. Close RGB MoCap before continuing.
echo Reuses the installed FlowHMR runtime. Prepare the model files in Model Manager.
"runtime\python.exe" -I -X utf8 -B tools\setup_vidihand2.py
set "vidihand2_result=%errorlevel%"
pause
exit /b %vidihand2_result%
