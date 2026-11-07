@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
rem Parse the final block before Git can replace this running batch file.
rem Clear a possible environment shadow; CALL expands only the numeric status.
set "ERRORLEVEL="
(
  "%~dp0runtime\python.exe" -I -X utf8 -B "%~dp0tools\git_update.py" %*
  call exit /b %%errorlevel%%
)
