@echo off
setlocal DisableDelayedExpansion
if not exist "%~dp0runtime\python.exe" (
  echo RGB MoCap is not installed. Run install.cmd first.
  exit /b 1
)
"%~dp0runtime\python.exe" -I -X utf8 -u -B "%~dp0cli_boot.py" %*
exit /b %errorlevel%
