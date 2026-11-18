# Linux support (experimental, first pass)

This is an additive Linux layer. Windows behaviour is unchanged: every new helper
returns the value that used to be hard-coded when `os.name == 'nt'`.

## Quick start (Arch / CachyOS)

```
sudo pacman -S --needed uv ffmpeg xcb-util-cursor   # plus the NVIDIA driver
python3 tools/setup_linux.py --profiles shared exmr # minimal; omit --profiles for all five
./RGBMoCap.sh
```

`setup_linux.py --plan` prints what would be installed without changing anything.
Model files are still prepared in **Model Manager**, as on Windows. The setup
downloads its own Python 3.10 / 3.12 through `uv`, so the system Python is irrelevant.

## What changed

- `mocap/platform_compat.py`: `NO_WINDOW`, `exe_name()`, `venv_python()`, `site_packages()`,
  `dev_worker_command()`.
- `mocap/paths.py`: on non-Windows, workers start as `<env>/bin/python -m mocap.worker_progress`
  using the `.venv`, `.venv-flow`, `.venv-hand`, `.venv-htd`, `.venv-exmr` layout.
  `mocap/runtime_registry.py` is deliberately **not** modified: the Windows installer loads
  it standalone and pins its SHA-256 in `installer/runtime-lock.json`.
- `creationflags=0x08000000` / `subprocess.CREATE_NO_WINDOW` replaced by `NO_WINDOW` (0 on
  POSIX, where any non-zero value raises `ValueError`). Before this, opening any video failed.
- `video.executable()` uses system `ffmpeg`/`ffprobe` when `tools/bin/*.exe` is absent.
- `retarget/runtime.py`: finds `blender` (set `RGBMOCAP_BLENDER` to the binary or its folder).
- `PYOPENGL_PLATFORM` is `egl` instead of `win32` in the ExMR and HandFlow workers.
- New: `tools/setup_linux.py`, `launch_linux.py`, `RGBMoCap.sh`.

## Not done yet

- **Motion correction**: only a Windows `.pyd` is shipped. Build `third_party/KimodoMotionCorrection`
  (its CMake already has a GCC branch) and update the `*.pyd` globs in `mocap/correction.py`
  and `mocap/worker.py`. Until then *Apply Correction* fails.
- **Blender 4.5 LTS** is not installed by the script (use the official tarball).
- **Windows-only optional models**: PEAR live, GEM-X, MAMMA, PromptHMR, SAM 3.1, Track Face and the
  FlowHMR HaMeR finger pass assume the managed `runtime*/python.exe` layout.
- Cancelling a job kills the worker but may leave its ffmpeg children running.
- Webcam enumeration (`dshow`/`MSMF`) is Windows-only.

## What was and was not tested

Tested offline in a Linux sandbox: syntax of every changed file, the helper functions,
`video.probe()` on a real clip (fails before the patch, works after), worker command
resolution, `setup_linux.py --plan` against the Windows lock's package counts, and that the
locally rebuilt bundled wheels install.

**Not tested**: the PyTorch/CUDA installs (no network), the Qt GUI, any model inference,
Model Manager downloads, or export. Treat the first real run as the real test.
