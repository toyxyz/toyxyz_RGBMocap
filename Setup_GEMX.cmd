@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo Run install.cmd first.
  pause
  exit /b 1
)
echo GEM-X optional Windows setup. Close RGB MoCap before continuing.
echo Allow 12 GiB of temporary free space for runtime setup. Models are separate.
echo The pinned NVIDIA Warp wheel needs a Turing or newer GPU and R580-series or newer driver.
echo Review GEM-X source and weight terms, SOMA-X terms, and the detector notice:
echo https://github.com/NVlabs/GEM-X/blob/4aef1452e5df171deff4f293c08e643eb450981e/LICENSE
echo https://huggingface.co/nvidia/GEM-X
echo https://huggingface.co/nvidia/SOMA-X
echo https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
choice /c YN /n /m "I reviewed the upstream terms. Continue? [Y/N] "
if not "%errorlevel%"=="1" exit /b 1
"runtime\python.exe" -I -X utf8 -B tools\setup_gemx.py
set "gemx_result=%errorlevel%"
pause
exit /b %gemx_result%
