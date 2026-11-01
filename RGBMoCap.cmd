@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo RGB MoCap is not installed. Run install.cmd first.
  pause
  exit /b 1
)
"%~dp0runtime\python.exe" -I -X utf8 -u -B "%~dp0launch_source.py" %*
set "app_exit=%errorlevel%"
if not "%app_exit%"=="0" (
  echo.
  echo RGB MoCap exited with code %app_exit%. See the messages above.
  pause
)
exit /b %app_exit%
