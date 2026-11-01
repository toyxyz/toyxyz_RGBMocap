@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
"%~dp0runtime\python.exe" -I -X utf8 -B "%~dp0tools\git_update.py" %*
set "result=%ERRORLEVEL%"
if not "%result%"=="0" echo Update stopped; review the message above.
if "%~1"=="" pause
exit /b %result%
