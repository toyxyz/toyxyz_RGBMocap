@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
echo Optional Unreal Live Link setup. Close RGB MoCap before setup.
echo Uses your installed Unreal Engine 5.6+.
echo Use --prebuilt with a matching sender ZIP, or use Visual Studio C++ tools to build locally.
echo No Unreal binaries or model files are downloaded.
"runtime\python.exe" -I -X utf8 -B tools\setup_unreal_stream.py %*
set "rgb_unreal_result=%errorlevel%"
pause
exit /b %rgb_unreal_result%
