# Linux (experimental)

The Linux installer uses the same pinned runtime profiles as Windows with
separate hash-checked Linux wheels. Both platforms keep their own Python
runtimes, models, projects and caches. Try the AppImage below or install from the `linux_support` branch
in a dedicated checkout on the Linux filesystem. The Linux paths cover single-video ExMR,
FlowHMR and GVHMR, native motion correction, source export and optional Track
Face. The optional PEAR, MAMMA, PromptHMR, GEM-X, SAM 3.1 and Unreal Live Link
sender setups use Windows tooling.

## Requirements

Use x86-64 Linux with a compatible NVIDIA driver for CUDA 12.8, OpenGL 4.3 or
newer, FFmpeg and FFprobe, and a C++17 compiler with CMake 3.24 or newer. On
Ubuntu 24.04, including WSL 2 with WSLg, install the system libraries first:

```sh
sudo apt update
sudo apt install git curl ca-certificates build-essential cmake ffmpeg mesa-utils \
  libgl1 libegl1 libglu1-mesa libxkbcommon-x11-0 libxcb-cursor0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
  libxcb-randr0 libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0
```

Track Face additionally uses PortAudio (`libportaudio2` on Ubuntu). For
Blender bundles, compatibility FBX and character retargeting, install the
pinned Blender 4.5.0 build with `./install.sh --with-blender`. You can also use
Blender 4.5 LTS from `PATH`. To override the managed Blender, set
`RGBMOCAP_BLENDER` to the Blender executable or to the directory containing it.
Install `libsm6` and `libice6` on Ubuntu 24.04 for the standalone Blender build.
The optional installer verifies the official archive and stages the build
before publishing it under `runtime_character/blender-linux`. If a later release
changes the pin, keep the current directory as a backup before requesting the
new managed Blender.

On WSL 2, use a checkout on the Linux filesystem, for example in your home
directory, and use the NVIDIA Windows host driver. WSL exposes CUDA to Ubuntu.
The launcher defaults to Qt's XCB backend when X11 or XWayland provides
`DISPLAY`. An explicit `QT_QPA_PLATFORM` takes precedence. If the Qt viewport
uses software rendering on an NVIDIA WSLg system, check
`glxinfo -B` and try the following per-launch settings:

```sh
QT_QPA_PLATFORM=xcb GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA ./RGBMoCap.sh
```

## Test the AppImage on Linux Mint

Download the [x86-64 AppImage and its SHA-256 checksum][appimage-release].
This experimental build was tested with CUDA on Ubuntu 24.04 under WSL 2.
Native Linux Mint testing is welcome. Install the host dependencies listed
above and run the AppImage from a terminal to see setup progress.

```sh
sha256sum -c RGBMoCap-Linux-0dc69476-x86_64.AppImage.sha256
chmod +x RGBMoCap-Linux-0dc69476-x86_64.AppImage
./RGBMoCap-Linux-0dc69476-x86_64.AppImage
```

The AppImage is about 21 MiB. On first launch it installs the pinned Python
runtimes and builds native motion correction. Allow 32.3 GiB of free space for
setup and about 8.1 GiB of runtime downloads. The WSL test installation occupied
24.3 GiB including the installer cache. Use Model Manager to obtain the models
for your pipeline, which need additional storage.

By default the AppImage keeps its installation, projects, and setup log under
`~/.local/share/rgbmocap/appimage/<source-revision>/`. It uses
`$XDG_DATA_HOME/rgbmocap/appimage/<source-revision>/` when an absolute
`XDG_DATA_HOME` is set. Later launches of the same build reuse its installed
runtimes. The setup log is `appimage-setup.log` inside that folder.
Each new source revision uses a separate folder. To open an existing project,
use File > Open Project and browse to the previous build's projects folder.

If FUSE cannot mount the AppImage, run it with `--appimage-extract-and-run`.
For runtime repair without opening the app, use `--setup-only --verify-all`.
For the optional Blender installation, use `--setup-only --with-blender`.

[appimage-release]: https://github.com/toyxyz/toyxyz_RGBMocap/releases/tag/linux-appimage-0dc69476

### AppImage setup cannot find CMake

The published AppImage builds the native motion correction module on first
launch. This requires host CMake 3.24 or newer and a C++17 compiler. It does
not install system build tools for you.

If setup stops at **Building native motion correction from source** with an
error asking for CMake, check whether the launching shell can find it. On
Ubuntu 24.04-based Linux Mint, the documented setup is:

```sh
sudo apt update
sudo apt install cmake build-essential
cmake --version
g++ --version
./RGBMoCap-Linux-0dc69476-x86_64.AppImage
```

Run the last command from the folder containing the downloaded AppImage. If
the error suggests running `./install.sh`, that command is for a source
checkout. Retry the same AppImage and consult `appimage-setup.log` if it still
fails. See [the Linux Mint report](https://github.com/toyxyz/toyxyz_RGBMocap/issues/56#issuecomment-7033302555).

The AppImage linked above contains source revision `0dc69476`. Follow the source
installation steps below to use changes made to `linux_support` after that build.

## Install, launch and update

```sh
git clone --branch linux_support --single-branch https://github.com/toyxyz/toyxyz_RGBMocap.git
cd toyxyz_RGBMocap
./install.sh
./RGBMoCap.sh
./update.sh
```

Use Model Manager to check or download models for the selected pipeline.
`update.sh` fast-forwards a clean local `linux_support` branch tracking its
remote counterpart and verifies the installed runtimes. To upgrade from an older
Linux build whose updater still asks for `main`, run `git pull --ff-only` on
`linux_support` and then `./install.sh` once. Use `./install.sh --verify-all`
for a full runtime check. Installation and runtime repair preserve models and
projects.

The installer verifies the pinned uv executable, installs private Python 3.10
and 3.12 under `.setup/python/`, and compiles the native motion correction
module. Linux runtime paths use the POSIX prefix layout, defined alongside
Windows paths in `mocap/runtime_registry.py`.

On a filesystem with limited metadata space, wheel extraction can fail even
when byte capacity remains. To place the installer's scratch files elsewhere,
create an absolute directory and set `RGB_MOCAP_SETUP_TEMP`:

```sh
mkdir -p /tmp/rgbmocap-setup
RGB_MOCAP_SETUP_TEMP=/tmp/rgbmocap-setup ./install.sh
```

Review both platform locks after changing the runtime registry or package
pins. `tools/linux_lock.py` derives the Linux runtime lock from the Windows
pins and adds exact versions for Linux-only requirements.
