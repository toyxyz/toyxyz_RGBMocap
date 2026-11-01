@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\git_bootstrap.ps1" %*
set "result=%ERRORLEVEL%"
if not "%result%"=="0" echo Installation failed. See the message above; rerun install.cmd to retry.
if "%~1"=="" pause
exit /b %result%
