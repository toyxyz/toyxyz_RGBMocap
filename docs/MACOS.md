# macOS on Apple Silicon (experimental)

The macOS installer uses the same pinned runtime profiles as Windows and
Linux, with separate hash-checked wheels for Apple Silicon and no CUDA
packages. Models run on the GPU through PyTorch MPS. This port lives on the
`macos_support` branch, which extends `linux_support`, and has been tested on
one machine only. Treat it as a preview and compare important results with a
Windows or Linux run.

## Status

Tested on an Apple M5 Max (36 GiB) with macOS 26.6. Each stage below completed
without errors through the app's worker on a 300-frame, 1080x1920 clip. The
results were not compared numerically with a CUDA run.

| Feature | Status | Time for 300 frames |
| --- | --- | --- |
| Installation and `--verify-all` | Works | About 6 minutes |
| App, timeline, video preview with audio | Works | |
| 3D viewport: SMPL-X mesh and bone overlay | Works | Real time |
| GVHMR reconstruction | Works | 17 s tracking, 44 s reconstruction |
| Find person | Works | 22 s |
| HTD Refine | Works | 35 s and 136 s |
| FootMR | Works | 24 s |
| HandFlow | Works | 33 s |
| RIFE frame interpolation before reconstruction | Works | 6 s for a 4-second range |
| Contact correction (native module) | Runs | 1 s |
| ExMR keypoint preprocessing | Works | 54 s |
| ExMR features and reconstruction | Not tested | |
| FlowHMR | Not tested | |
| FBX and BVH export | Works | Under 1 s |
| Compatibility FBX and Blender project export, one or several people | Works with Blender 5.1.2 | 1 to 3 s |
| Character import and retargeting | Not tested | |
| Imported character preview | Not available | |
| Track Face | Not tested | |
| PEAR, MAMMA, PromptHMR, GEM-X, SAM 3.1, Unreal Live Link sender | Not available (Windows tooling) | |

Reconstruction was also run from the app's own controls: one person with
GVHMR, FootMR and HandFlow, and several people after **Find person** on a
four-dancer clip.

## Requirements

- An Apple Silicon Mac. Intel Macs are refused by the installer.
- macOS 14 or newer. This is the minimum of the pinned wheels; only macOS 26
  was tested.
- Xcode command line tools, CMake 3.24 or newer, FFmpeg and FFprobe, and Git:

  ```sh
  xcode-select --install
  brew install cmake ffmpeg git
  ```

- About 7.3 GiB of free space for the runtimes (about 1 GiB of downloads),
  plus the models for your pipeline. The GVHMR set with all add-ons is about
  15 GiB.
- Blender for compatibility FBX, Blender project export and retargeting. No
  managed Blender is installed on macOS. `Blender.app` in `/Applications` or
  `~/Applications` is found automatically; otherwise set `RGBMOCAP_BLENDER`
  to the application or its executable. The project is validated with
  Blender 4.5 LTS. On macOS only 5.1.2 was tried: the exported FBX files and
  Blender project opened there with the armature upright, 300 animated frames,
  the camera and the linked video. Plain FBX and BVH export do not use
  Blender.

## Install and launch

Until the branch is merged upstream it is available from this fork:

```sh
git clone --branch macos_support --single-branch https://github.com/lvalics/toyxyz_RGBMocap.git
cd toyxyz_RGBMocap
./install.sh
./RGBMoCap.sh
```

The installer verifies the pinned uv executable, installs private Python 3.10
and 3.12 under `.setup/python/`, installs the six runtime groups from
`installer/runtime-lock-macos.json` with required hashes, and compiles the
native motion correction module for arm64. Use `./install.sh --verify-all` for
a full runtime check. Installation and repair preserve models and projects.

`update.sh` is not adapted to this branch yet. To update, run
`git pull --ff-only` and then `./install.sh`.

Use **Model Manager** to check or download models for the selected pipeline.
Automatic Google Drive downloads can be refused with a public download limit;
use the row's **Manual Download...** and **Import...** actions in that case.
Registered files (SMPL-X, MANO, SMPL-H) and gated Hugging Face files need your
own account, as on other platforms.

## How it works

- **GPU.** `mocap/mps_compat.py` is installed when a worker starts. If CUDA is
  available it does nothing. Otherwise, on a Mac with MPS, it redirects
  `.cuda()` and `.to('cuda')` to MPS and answers the `torch.cuda` memory
  queries used for batch plans and Resource Safety from `torch.mps`. Windows
  and Linux behavior is unchanged.
- **Viewport.** macOS provides OpenGL 4.1 without compute shaders. Below
  OpenGL 4.3 the mesh and bone renderers evaluate the same skinning and
  normals with NumPy. With 4.3 the compute-shader path is used as before.
- **Motion correction.** The C++ math layer is written with SSE intrinsics.
  On arm64 the build fetches the pinned
  [sse2neon](https://github.com/DLTcollab/sse2neon) v1.8.0 header (MIT) to
  translate them to NEON.
- **Locks.** `tools/macos_lock.py` derives the macOS runtime lock from the
  Windows pins by reusing `tools/linux_lock.py`. Regenerate it after changing
  pins or `mocap/runtime_registry.py`.

## Known limitations

- Results can differ slightly from CUDA. In one test, GVHMR on MPS and on the
  CPU of the same Mac agreed to within 0.001 radians and 0.05 mm.
- MPS reports running out of memory as a plain error, so the automatic
  recovery that follows a CUDA out-of-memory error does not run. Free memory
  is shared with other applications; close heavy ones before long clips.
- Motion correction output was not compared with the x86 build. A few
  approximate reciprocal and square-root instructions round differently on
  arm64.
- MPS has no 64-bit floats. ExMR's Sapiens2 heatmap decoder therefore runs on
  the CPU. Other models may need the same treatment where they are untested.
- The Dock shows the interpreter's name, because the app is not packaged as
  an application bundle. The menu bar shows "RGB MoCap".
- Some texts still refer to Windows setup commands, for example the SAM 3.1
  note in Model Manager.
- The registry change for macOS updated a hash in the Windows and Linux
  runtime locks. Existing installations on those platforms are asked to run
  their install command once; their packages are unchanged. Neither platform
  was retested with these changes.
