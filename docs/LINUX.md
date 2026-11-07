# Linux (experimental)

The application is Python 3.10 + PySide6 (Qt 6) + OpenGL 4.3 with separate pinned
worker runtimes. On Linux the same runtime groups, package versions, hash-checked
downloads and staged promotion are used as on Windows; only the wheel files differ.

Scope of this port: the GUI and the single-video **ExMR / FlowHMR / GVHMR** paths,
including FlowHMR's optional HaMeR finger refinement (which runs in the ExMR runtime),
contact correction and FBX/BVH source export. Not yet ported: PEAR live preview
(Windows shared-memory camera streams), MAMMA and PromptHMR optional setups, and a
pinned Blender download for character retargeting and `.blend` bundles.

Tested on Kali Linux (glibc 2.43, NVIDIA RTX A6000, NTFS checkout): installation, launch,
Model Manager, and a FlowHMR + HaMeR reconstruction (Static Camera) through contact
correction and BVH preparation. ExMR and GVHMR reconstruction, Find person, File → Export,
Save Project As and preview-video export are ported but were not run on Linux.

Platform plumbing is aliased in `mocap/resource_cache.py` (`PLATFORM_ONLY`) and
`mocap/cache_compat.py`, so existing Windows inference and correction receipts stay valid.

## Requirements

- x86_64 Linux with glibc 2.28 or newer, an NVIDIA driver supporting CUDA 12.8 and OpenGL 4.3.
- `curl`, `tar`, `cmake` (3.24+), a C++17 compiler and `git` (CMake fetches the pinned
  pybind11 v2.11.1 and Eigen 3.4.0 to build native motion correction).
- `ffmpeg` and `ffprobe` on `PATH` (the system FFmpeg is used instead of the Windows 7.1.1 build).
- Optional, for character import/retarget and Blender bundles: a Blender 4.5 LTS
  executable, selected with `RGBMOCAP_BLENDER=/path/to/blender`.

## Install, launch, update

```bash
./install.sh
./RGBMoCap.sh
./update.sh
```

`install.sh` runs `tools/git_bootstrap.sh`: it downloads the pinned uv binary
(SHA-256 checked), installs private CPython 3.10.22/3.12.10 under `.setup/python/`,
then runs the same `tools/git_managed_install.py` as Windows with
`installer/runtime-lock-linux.json` and `installer/git-support-lock-linux.json`.
Runtimes use the POSIX prefix layout (`runtime/bin/python3.10`,
`runtime/lib/python3.10/site-packages`); `mocap/runtime_registry.py` is the single
place that maps profiles to interpreters on both platforms.

Wheels are extracted to `.setup/tmp` before being copied into a runtime, so a group
briefly needs twice its file count on the install volume (about 55k files for ExMR).
On a nearly full NTFS volume that can exhaust the Master File Table (`Failed to extend
mft data allocation` in the kernel log, reported as "No space left on device" despite
free space). Point extraction elsewhere with an existing absolute directory:

```bash
mkdir -p /tmp/rgbmocap-setup && RGB_MOCAP_SETUP_TEMP=/tmp/rgbmocap-setup ./install.sh
```

The Linux lock is derived from the Windows lock by `tools/linux_lock.py` (same names
and versions; Linux-only requirements such as torch's `nvidia-*` and `triton` are added
at their exact pins). Regenerate it after changing the Windows lock or the registry.
