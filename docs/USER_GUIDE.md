# RGB MoCap user guide

[Overview](../README.md) · [Build from source](BUILD.md)

A native Windows app for turning video into human motion, reviewing the body mesh
and animation bones, correcting hand/foot contacts, retargeting motion to humanoid
characters, and exporting animation for Blender or MMD. Processing runs locally.

For optional single-video capture of people together, see the
[PromptHMR setup and usage guide](models/PROMPTHMR.md). It supports fixed-camera
video through a separate `Setup_PromptHMR.cmd` runtime, with a People count and
Interaction Prompt option. It performs its own tracking and does not generate
animated fingers. Standard correction, retargeting and FBX controls apply.

## Character retargeting

Transfer a reconstructed person's motion to a character without running AI
inference again. Supported targets are **Mixamo FBX**, **VRM** and **PMX** humanoids.

1. Prepare the source person's motion and apply any contact edits.
2. Choose **File > Import Character...** and select the character file. It appears
   at the origin in its rest pose, below the source people in the **People** list.
3. Right-click the character and choose **Retarget > [source person]**. The status
   bar shows processing progress. Play or scrub the timeline to review the result.
4. Right-click the retargeted character and choose **Export Character...**, or
   select its row and use that command in **File**.

Up to **8 characters** can be imported per project. Each character can use a
different source person. Check or uncheck its row to show or hide it; use
**Remove Character** in its right-click menu to remove it from the project.
Character data and assigned motion are saved inside the project.

Characters are drawn in front of the SMPL source meshes, with **Bone Overlay**
above both. **Name** shows the list name above each visible person or character,
including characters still in their rest pose. Names stay upright and the same
screen size; crowded labels are separated. These controls change the preview only.

Retargeting uses the selected source motion and its applied contact timings.
**Apply Correction** also refreshes characters linked to the corrected source,
including hidden characters. After other source changes, run **Retarget** again.
A character marked **Needs retarget** must be updated before export.
Unreachable hand/foot targets
are not solved by stretching bones. Large differences in body proportions may
require further editing in animation software. Travel distance can differ from
the source because motion is scaled to the target character's leg length;
shorter-legged characters may cover less ground, and longer-legged characters
may cover more. Source contact timings are retained.

Only mapped body/finger bones and necessary body helpers receive animation;
the armature object stays unkeyed.
Hair, clothing and secondary physics are not simulated.

| Export option | Result |
| --- | --- |
| **FBX + Include character mesh** | Character mesh, skeleton and animation. |
| **FBX, mesh unchecked** | Skeleton and animation only. |
| **VMD (PMX characters only)** | Body/finger motion sampled at 30 fps for the same PMX model in MMD. Facial animation and physics are not generated. |

Separate takes produce separate files. Both FBX options preserve the imported
character's original bone hierarchy and rest pose separately from its animation;
the first animation frame does not replace the rest pose. Existing retargeted
characters can be exported again to use the corrected FBX rest-pose handling.

Known humanoid mappings are automatic; custom bone mapping is not yet available.
Files missing required body bones are rejected with a message. The viewport shows
base-color textures at their original resolution, with simplified lighting rather
than a full material/physics renderer. Existing characters prepare a texture cache
in the background on first display; reimporting or retargeting is not required.
Textures are reused during playback. Oversized textures produce a clear error
instead of being silently reduced in resolution.

### Included Blender runtime

`install.cmd` installs a separate **Blender 4.5 LTS** runtime in
`runtime_character`. It runs in the background to import characters, evaluate rig
constraints, bake bone animation and export character FBX/VMD files. You do not
need to install or open Blender separately after running `install.cmd`. Keep this
folder with the app.

RGB MoCap performs the retargeting/contact calculations and viewport playback.
The Blender backend also provides a foundation for future format support and
animation-processing features; these are possibilities, not additional features
currently provided by the app. Hair/clothing physics remains unsupported.

The Git checkout does not include Blender binaries. `install.cmd` downloads the
pinned Blender runtime automatically. Development runs may set
`RGBMOCAP_BLENDER` to a compatible `blender.exe`; see [Build from source](BUILD.md).
Blender retains its GPL license and bundled notices; see [Third-party notices](../THIRD_PARTY_NOTICES.md).

## Install

**Tested configuration:** NVIDIA GeForce RTX 5090 (32 GB VRAM), AMD Ryzen 9
9950X, 96 GB RAM, Windows 11 Pro (64-bit). This is the tested configuration,
not a minimum specification. Other GPUs, drivers and clean Windows installations
have not yet been verified; compatibility reports are welcome.

1. Install [Git for Windows](https://git-scm.com/download/win). Open **Command
   Prompt** in a writable folder, then run:

   ```cmd
   git clone https://github.com/toyxyz/toyxyz_RGBMocap.git
   cd toyxyz_RGBMocap
   ```

   Avoid protected folders such as Program Files. The commands below are for
   Command Prompt; in PowerShell, prefix each `.cmd` filename with `.\`.
2. Run `install.cmd` with an internet connection. It downloads and verifies
   pinned Python environments, Blender 4.5.0, FFmpeg 7.1.1 and small upstream
   support files, and installs the included prebuilt motion-correction module.
   It prepares ExMR, FlowHMR and GVHMR runtimes, their supported add-ons, and
   PEAR's small shared-runtime support together, not only the selected model.
   The AI runtimes remain
   separate; model weights/body data are not included. Downloads and free disk
   space are needed, and the install can take time.
3. Run `RGBMoCap.cmd`. Keep its console open while the GUI runs; launch errors
   and exit codes are displayed there. Processing messages and app **Console** output
   are also sent there, but very long or high-volume output can be truncated or
   dropped. Use **Help > Check System** if needed.
4. In **Model Manager**, download or import the selected AI model files. Some
   models require registration and agreement to their provider's terms and must
   be downloaded manually.

**For MAMMA multi-video capture**, complete those steps, install Visual Studio
2022 C++ Build Tools with a Windows SDK and NVIDIA CUDA Toolkit 12.8, then close
the app and run:

```cmd
Setup_MAMMA.cmd
```

Review the restricted-use license prompt before accepting. Reopen the app,
select **MAMMA**, and prepare its files in **Model Manager**. Its **Runtime
Setup...** button shows readiness and setup instructions. The optional runtime
is not installed by `install.cmd` or `update.cmd`; ExMR, FlowHMR and GVHMR do not
need it. Follow the [MAMMA installation and model checklist](MAMMA.md) for exact
prerequisites, the locked-head body models and multi-video requirements.

**For PromptHMR single-video interaction capture**, complete the base installation
and the same C++/Windows SDK/CUDA Toolkit 12.8 prerequisites, close the app, then run:

```cmd
Setup_PromptHMR.cmd
```

Review its license prompt, reopen the app, select **PromptHMR** and prepare all
files in **Model Manager**. This setup is independent of MAMMA and is also excluded
from `install.cmd`/`update.cmd`. The [PromptHMR guide](models/PROMPTHMR.md) lists
the exact files, fixed-camera limits, native-build verification and repair steps.
The Model Manager's **Runtime Setup...** button shows instructions; it does not
run installation inside the app.

To update, close RGB MoCap and run `update.cmd` in the repository folder. It
refuses to overwrite tracked local edits, requires the `main` branch, pulls only
fast-forward changes and checks the runtimes. Back up or commit local changes
before updating. A direct `git pull` updates source only; close the app first,
and run `install.cmd` if dependencies change. The launcher will ask for setup if
it detects an incomplete update or a dependency mismatch.

If a download or installation fails, read the console message, check the internet
connection and free disk space, and rerun `install.cmd`. A completed download may
be reused. Do not delete `models`, `projects` or the remaining `.setup` state to
retry. If Git reports a conflict or local changes, resolve or back them up before
running `update.cmd` again.

On the tested PC, a separate-folder first install took about 10 min 32 sec with
previously verified package downloads reused; Blender and FFmpeg were downloaded.
A subsequent code update and runtime verification took about 1 min 55 sec. These
are single local tests, not estimates for a download with an empty cache or for
other PCs; connection speed and storage performance affect the time.

### Installed size

The measured model/runtime inventory, plus GeoCalib for automatic MAMMA gravity, totals approximately
**67.9 GB (63.2 GiB)** with the five offline models and supported add-ons installed.
Blender and FFmpeg are included. The inventory excludes projects, imported
characters, exports, caches (including Python bytecode), retained download
archives, obsolete runtime generations and separately installed compiler tools.
Shared model files and hardlinks are counted once.

| Configuration | Runtime | Models | Total |
| --- | ---: | ---: | ---: |
| ExMR + FlowHMR + GVHMR, including supported add-ons | 12.35 GiB | 28.89 GiB | 41.23 GiB |
| MAMMA additional files | 6.11 GiB | 3.53 GiB | 9.65 GiB |
| PromptHMR additional files | 6.22 GiB | 6.13 GiB | 12.34 GiB |
| **Five offline models** | **24.68 GiB** | **38.55 GiB** | **63.22 GiB** |
| PEAR live preview, additional | Small support overlay; shared runtime reused | ~2.84 GiB | ~2.85 GiB |

Adding PEAR to all five offline sets gives an estimated combined total of about
**66.07 GiB (70.9 GB)**. This includes about **0.11 GiB** of required GeoCalib weights; existing VGGT weights are reused. The approximately
11.3 MiB support overlay is included in base setup. Its original
geometry ZIP needs additional download space.

Totals use unrounded bytes; rounded cells may not sum exactly. These are local
observations, not universal requirements. App source files add space beyond this
model/runtime inventory. MAMMA's added model size assumes the base installation's
VGGT-Omega is present; without it, its model set totals about 7.80 GiB. GeoCalib adds about 0.108 GiB for automatic gravity, counted once between MAMMA and PEAR. These totals include that addition to the earlier measured inventory. PromptHMR's
added model size includes SMPL Neutral, while its standard SMPL-X Neutral is
already counted in the base set. Its specific weights/support files alone total
about 6.09 GiB. Installed Python packages remain separate; verified identical
download archives can be reused without merging runtimes.

**MAMMA and PromptHMR are separate optional runtime installations.** After
`install.cmd`, use `Setup_MAMMA.cmd` or `Setup_PromptHMR.cmd` for the model you
need, then prepare its weights in Model Manager. Neither requires the other.
`install.cmd` and `update.cmd` do not install these optional runtimes.

Final installed size is different from free space needed during setup. The base
installer estimates **32.4 GiB free before models**, including temporary work;
each optional setup asks for about **18 GiB free**, excluding models and compiler
tools. Download archives, C++ Build Tools, CUDA Toolkit and generated projects
need additional space. Earlier optional setup reports of 6.19 GiB (MAMMA) and
6.32 GiB (PromptHMR) included generated files excluded from this cache-free
inventory. See [MAMMA](MAMMA.md) and [PromptHMR](models/PROMPTHMR.md) for setup
scope. No clean-Windows or other-GPU installation test is claimed.

## Storage and cleanup

In the installed app, `<install folder>` means the folder containing
`RGBMoCap.cmd`. Close RGB MoCap and any installation or update before deleting files. Use **File →
Open Project Folder** to locate the current project; a **Save Project As** copy
may be outside the installation folder.

| Location | What you can remove and what happens |
| --- | --- |
| `<install folder>\cache\` | Its contents can be deleted. This includes `huggingface`, `torch` and ExMR support caches. They are recreated or downloaded again when needed; the next run may take longer and require internet access. Saved projects and the separate `models` library remain. |
| `<install folder>\projects\<project folder>\` | Delete an entire unwanted project folder. This removes its model results, prepared videos, previews, contact edits and imported/retargeted characters. Reprocessing requires the source video and AI inference; manual edits are lost. Keep projects you still need. |
| Your **Save Project As** destination | The same rule applies to each saved project folder at your chosen location. It contains a copy of the video and project data. Deleting it removes that copy; cleaning the default `projects` folder does not clean external copies. |
| `<install folder>\character_scenes\<session folder>\` | Contains characters imported before a project was opened. Old session folders can be deleted once those imports are no longer needed or have been copied into a saved project. Unsaved session imports are lost. |
| `<install folder>\.setup\downloads\`, `.setup\uv-cache\` and `.setup\tmp\` | Downloaded wheels and temporary cache can be removed only after installation or update completes. They are reused for repair and updates; deleting them may require downloading files again. Keep the rest of `.setup`, including the private Python and installation state. |
| `<install folder>\.setup\mamma-downloads\` | Optional MAMMA package/source download cache. Its contents can be removed after MAMMA setup finishes; repair may need to download them again. Keep `runtime_mamma` and `.setup/mamma-runtime.json`. |
| `<install folder>\.setup\prompthmr-downloads\` | Optional PromptHMR package/source download cache. Remove contents only after setup finishes; repair may need to download them again. Keep `runtime_prompthmr` and `.setup/prompthmr-runtime.json`. |
| `<install folder>\.setup\pear-downloads\` | PEAR source/support and camera-alignment archives. Contents can be removed after installation/update finishes; repair may download them again. Keep `runtime_pear`, `.setup/pear-runtime.json` and `.setup/pear-alignment.json`. |
| `<install folder>\temp\` | Local PEAR Torch/kernel temporary cache. Contents can be removed after closing the app and its workers; the next run can recreate them. |

Repeated work mainly grows project data: each model has its own result folder.
Inside `exmr`, `flowhmr` or `gvhmr`, different processing ranges can add `clips`
subfolders. MAMMA and PromptHMR keep immutable `inference/<generation>` and
`results/<generation>` folders instead. Character imports/retargeting add
character data. These are saved work, not just
disposable caches. Do not delete individual model folders, `characters`, motion
tensors, preview files or receipt files from a project you want to keep; its
metadata can still reference them. Export or back up needed work, then remove an
unwanted project as a whole. Exported FBX/VMD files in other folders must be
managed separately.

Keep `models`, `runtime*`, `mocap`, `third_party`, bundled assets and
the remaining `.setup` repair files. Deleting these can remove required model
files or break the app. Downloaded model archives stored elsewhere can be removed after a successful installation/import if you do not
need an offline copy; keep a backup of manually acquired restricted models.

## Start here

For ExMR, FlowHMR and GVHMR:

1. Choose an **AI Motion Model**, then **File → Open Video**, or drop a video
   into the workspace.
2. Optionally trim the timeline's **Video** track by dragging its ends.
3. Use **Model Manager** to prepare the selected model's files.
4. For ExMR/FlowHMR/GVHMR with several people, use **Find person** and check
   who to reconstruct. MAMMA/PromptHMR track people during reconstruction.
5. Click **Reconstruct 3D**. Review the video, mesh and **Bone Overlay** together.
6. Adjust correction settings or contact timings, then click **Apply Correction**.
7. **File → Export...** saves the displayed, applied animation.

For synchronized multi-camera capture, select MAMMA before opening videos and
follow the [MAMMA workflow](MAMMA.md#reconstruct-and-export). It detects people
during reconstruction, so the **Find person** step is not used. PromptHMR also
tracks people during reconstruction; open one fixed-camera video and start with
**People = 2** and **Interaction Prompt** on. See [its workflow](models/PROMPTHMR.md#controls-and-supported-workflow).

To animate a character, follow [Character retargeting](#character-retargeting),
then use **Export Character...** for FBX or, for PMX, VMD.

**Open Project** restores saved results and settings. **Open Project Folder** opens
their location. **Save Project As** (`Ctrl+Shift+S`) creates a new named project
folder at your chosen location and switches to that copy. It includes the video,
motion, people and contact edits for every model, including all MAMMA input videos
when present. Open its `project.json` to return to it. Each project manages
`exmr`, `gvhmr`, `flowhmr`, `mamma` and `prompthmr` result folders as needed. Selecting a model
loads its saved motion; if it has no result, **Reconstruct 3D** creates its data in
that folder. Result selections, contact edits and settings stay separate by model.
New ExMR/FlowHMR/GVHMR processing ranges stay under that model's `clips` folder;
MAMMA and PromptHMR use their own inference/result generation folders. The previous project
folder format is not migrated; open the original video to create a new project.
Model weights remain in the app's model
library. Trimming does not modify the original video. Processing uses a
30 fps working clip; camera sampling does not reduce its motion frame count.

When reconstructing several people or visible takes with ExMR, FlowHMR or
GVHMR, RGB MoCap processes the same stage for each actor in turn using one loaded
model, then releases it before the next stage. ExMR connects keypoint, body/hand
features and motion phases in one worker. FlowHMR connects features and motion
for one person; its multi-person jobs retain separate phase grouping. GVHMR keeps
its observation and motion stages in one worker. Each actor keeps separate
motion, contacts and exports. This reduces repeated model loading; it does not
combine people into a larger GPU batch. Automatic and Low Memory still apply.
Optional GVHMR refinement/add-ons use their existing separate workers.

Automatic can overlap bounded CPU image preparation with GPU work when CPU/RAM
headroom permits; Low Memory disables that prefetch. For measured worker-startup
and feature-stage gains, see [resource modes and optimizations](PERFORMANCE.md).

## Choose a model

Each model has a [detailed guide](models/README.md): [ExMR](models/EXMR.md),
[FlowHMR](models/FLOWHMR.md), [GVHMR](models/GVHMR.md), [MAMMA](MAMMA.md)
and [PromptHMR](models/PROMPTHMR.md). For live bone preview, use [PEAR](models/PEAR.md).
These cover required files, settings, defaults and workflow limitations.

These quality comparisons are **subjective observations from our test videos**,
not accuracy scores or a universal ranking. See the in-app **Model Guide** for
recorded speed and memory measurements. New sessions start with **ExMR** selected;
opening a project restores its saved model. **Prepare Preview**, below
**Reconstruct 3D**, prepares existing motion without running inference again.

| Model | Observed strengths | Finger animation |
| --- | --- | --- |
| **ExMR** | Generally fast, with good overall motion quality. | Included. |
| **FlowHMR** | Slow, but the best overall motion quality in our tests. | Included. |
| **GVHMR** | Lowest overall quality in our tests, but sometimes better for unusual poses such as backflips. | Optional **HandFlow**; not inferred by GVHMR itself. |
| **MAMMA** | Multi-view capture of human motion and interactions. Needs separate setup and overlapping synchronized views; outside the quality ranking above. | Included. |
| **PromptHMR** | Can improve relative placement and interactions between multiple people in one fixed-camera video. Overall motion quality was lower than ExMR in our subjective tests; contacts and depth can still be wrong. Separate setup. | No finger tracking; fixed neutral pose. |
| **PEAR** | Live single-person bone preview from a webcam or video. Lightweight live foot lock; no offline correction, recording, retargeting or export. | Included. |

The five offline models support contact correction, a 52-bone overlay, FBX/BVH export and portable Blender bundles.
ExMR/FlowHMR/GVHMR use **Find person** for explicit people selection; MAMMA
and PromptHMR track people during reconstruction. The viewport shows the selected
source mesh: SMPL-X for ExMR/GVHMR/MAMMA/PromptHMR,
SMPL-H for FlowHMR. The export skeleton follows that motion; source estimation
errors can remain.

### MAMMA multi-view reconstruction

Select **MAMMA** before opening two or more synchronized fixed-camera videos.
There is no fixed camera-count maximum; available RAM and VRAM determine
practical capacity. Existing memory monitoring and OOM handling stay active.
See [input requirements and remaining app limits](MAMMA.md).
Camera thumbnails show four per row and switch the main preview without changing
the common 30 fps timeline or trim. VGGT estimates cameras by default;
**Import Camera Data...** accepts compatible JSON/MAMMA YAML calibration.
Motion Correction defaults on, preserves original motion and supports the same
contact editing and FBX workflow. **Save Project As** includes all camera videos.

Run **Setup_MAMMA.cmd** separately after the base installation; Model Manager >
**Runtime Setup...** explains its compiler prerequisites, disk use and restricted
licenses. Prepare the nine model assets through Model Manager. MAMMA detects and
matches people during reconstruction, so **Find person** is hidden. **People** can
be **0** for automatic detection or a fixed count. Enter only the number in **Reference Height (m)**; this
sets approximate VGGT scale and is unused with metric camera calibration.
For example, enter `1.70` for 170 cm; it does not force all actors to the same
height. Input videos must already start in sync and show the same performance.
Read [MAMMA setup, input assumptions and limits](MAMMA.md) before reconstruction.
The app uses SAM 2.1 and FP32 fitting; SAM 3.1 research tests remain separate.

## Recorded processing time and VRAM

Historical measurements on the tested PC above (30 fps inputs, October 2026).
Each single-video duration uses the same input across ExMR, FlowHMR and GVHMR.
MAMMA and PromptHMR use different clips and include more processing stages;
their rows are not direct speed comparisons. The older four-model records predate
the latest Low Memory changes.

For mode comparisons and the latest applied acceleration, see
[measured memory savings and speedups](PERFORMANCE.md). Those tests use different
clips and timing scopes from this historical table.

| Video length | Model | Input videos | Processing time | Peak VRAM (GiB) | Peak RAM (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 7 seconds | ExMR | 1 | 1m 08s | 11.32 | 19.31 |
| 7 seconds | FlowHMR | 1 | 2m 49s | 14.89 | 22.19 |
| 7 seconds | GVHMR | 1 | 2m 01s | 5.08 | 12.47 |
| 15 seconds | ExMR | 1 | 1m 47s | 11.32 | 19.31 |
| 15 seconds | FlowHMR | 1 | 5m 16s | 10.79 | 16.97 |
| 15 seconds | GVHMR | 1 | 2m 45s | 5.08 | 11.06 |
| 25.7 seconds | ExMR | 1 | 1m 44s | 11.32 | 19.31 |
| 25.7 seconds | FlowHMR | 1 | 7m 32s | 10.67 | 16.94 |
| 25.7 seconds | GVHMR | 1 | 3m 00s | 5.08 | 12.42 |
| 13.8 seconds | MAMMA | 3 | 11m 52s | 6.47 | 9.11 |
| 14.1 seconds | PromptHMR | 1 | 3m 56s | 4.61 | 8.74 |

For PEAR's separate live-frame timings, current fixed-bone behavior and measured
VRAM, see the [PEAR performance notes](models/PEAR.md#placement-and-performance).
These are not full-clip reconstruction measurements.

For ExMR, FlowHMR and GVHMR, times cover model-processing stages, excluding shared person
tracking, app contact correction and preview/export. GVHMR includes HTD, FootMR
and HandFlow. FlowHMR includes camera estimation and the original SAM path;
these timings predate Fast SAM. MAMMA used three synchronized videos resized
within 1920×1080, two people, SAM 2.1 and VGGT, with 413 frames at 30 fps
(13.77 seconds). Its 712.08-second total includes preparation, correction and
preview; memory peaks cover the four inference stages. See the
[MAMMA stage breakdown](MAMMA.md#recorded-processing-time-and-memory).

PromptHMR used 422 frames at 30 fps, two people and Interaction Prompt on. Its
time includes tracking, native inference/post-optimization and app correction/preview,
excluding public FBX writing. Peak CUDA reserved memory was 7.83 GiB. See its
[stage breakdown and validation](models/PROMPTHMR.md#recorded-validation).

VRAM is peak PyTorch **allocated** memory across workers, not total GPU usage
or reserved memory. RAM for ExMR/FlowHMR/GVHMR is peak private memory;
MAMMA/PromptHMR RAM is peak worker working set. Neither is total system RAM usage.
These are individual test runs,
not minimum hardware requirements or predictions for every video. Resolution,
camera motion, settings and available memory also affect results; a longer
video does not always use more VRAM or take proportionally longer.

## Prepare models

**Model Manager** lists files for the selected tool and its add-ons.

| Action | Meaning |
| --- | --- |
| **Auto Download / Download All** | Download supported missing public files; completed files are kept if you cancel or retry. |
| **Manual Download...** | Show the exact file, provider link and download instructions. Use for files requiring registration or access approval. |
| **Import...** | Validate and register a file you already downloaded. |

Automatic downloads can fail, especially for files hosted on **Google Drive**.
If a download fails, use **Manual Download...** on that model's row, open the
provided link in your browser and download the exact file listed in the
instructions. Use **Import...** directly in the manual download window, or on
the same model row, to select the downloaded file. The instructions can stay
open while you download and import. If it comes in an archive, extract the
required file first. Import checks
the file and places it in the correct model folder; check that the row is ready
before reconstructing.

Some models **do not support automatic download** because their providers
require account registration, access approval and/or acceptance of license terms
before downloading. This is separate from a temporary download failure. Use
**Manual Download...** to visit the provider's site, complete its requirements
in your browser, download the required file, then use **Import...** on the
matching row. The app cannot complete these steps on your behalf.

Registered body/hand models must be obtained under their provider's terms.
Follow the manager's exact file instructions; model families are not interchangeable.
The app does not need your provider password or access token.

| Used by | Registered or access-controlled files |
| --- | --- |
| ExMR / GVHMR | [SMPL-X v1.1 neutral](https://smpl-x.is.tue.mpg.de/download.php): import `models/smplx/SMPLX_NEUTRAL.npz` from the downloaded archive. The app stores it under `models/body_models/smplx/`. |
| FlowHMR | [AMASS SMPL-H neutral](https://mano.is.tue.mpg.de/download.php): Extended SMPL+H model, `smplh/neutral/model.npz` (16 shape components). |
| ExMR / GVHMR HandFlow | [MANO](https://mano.is.tue.mpg.de/download.php): `MANO_RIGHT.pkl` from `mano_v1_2.zip`. |
| ExMR / FlowHMR | [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3): supported `model.ckpt`, `model_config.yaml` and `assets/mhr_model.pt`, each in its matching row. |
| FlowHMR moving camera / MAMMA automatic cameras | [VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega): supported `vggt_omega_1b_512.pt`, shared under `models/flowhmr/`. Not needed for FlowHMR Static Camera or MAMMA Manual Camera Data. |
| MAMMA automatic cameras / PEAR Camera Align | [GeoCalib](https://github.com/cvg/GeoCalib): background vertical direction, shared `models/pear/geocalib-pinhole.tar`. MAMMA manual calibration bypasses it. |
| MAMMA | [Registered MAMMA downloads](https://mamma.is.tue.mpg.de/): `mamma_mask_full_cvpr.ckpt` and `verts_512.pkl`. Follow the matching Model Manager rows. |
| MAMMA | [SMPL-X with removed head bun (NPZ, 392 MB)](https://smpl-x.is.tue.mpg.de/download.php): import the neutral, male and female locked-head bodies into `models/mamma/smplx/`. Standard SMPL-X v1.1 is not interchangeable. |

MAMMA's [complete ten-asset checklist](MAMMA.md#prepare-model-files) also covers
SAM 2.1, YOLO12x and OpenCLIP. Its runtime setup and model imports are separate
steps; a ready runtime does not mean the models are ready.

### Reuse models in a new installation

After installing the app in a new location, close both copies of the app and
copy the **entire `models` folder** from the old installation into the new
installation folder, beside the new **RGBMoCap.cmd**:

```text
Old installation/models/  →  New installation/models/
```

The destination must be `<new installation>/models/`, not
`<new installation>/models/models/`. Keep all subfolders,
configuration files, `.receipt.json` files and license files together. Copy
first and keep the old files until the new installation works.

Open the new app and check **Model Manager** for each model you use. If a copied
file is not marked ready, use **Import...** on its row and select that file
from the copied folder to validate it again. Copying can change file timestamps,
and newer releases may require a different model version; follow the row's
**Manual Download...** instructions if validation fails.

Let `install.cmd` create the new runtime folders; do not copy the old `runtime*`
folders over them. The `models` folder contains model files only. Saved work
is in separate project folders, which can be copied separately and opened
with **File → Open Project**.

## Model and processing settings

Changes to camera, smoothing or feature settings take effect on **Reconstruct 3D**.
GVHMR refinement checkboxes also select saved preview/export results. If the
selected result is missing, reconstruct it; the app reports it as unavailable.

| Setting | Function and when to use it |
| --- | --- |
| **FlowHMR / ExMR → Refine Camera Rotation** | Optional background refinement of the saved moving-camera display track. Default off. Validates small rotation corrections; unreliable samples retain the original. No motion re-inference. |
| **ExMR → Camera** | Static Camera for a camera that stays still without zooming; Moving Camera / SimpleVO when it moves. |
| **ExMR → Smooth Motion** | Reduce small changes between frames. Off keeps the unsmoothed result. |
| **FlowHMR → Fast SAM** | Speed up body/hand image processing with the same model files. Small numerical differences can affect motion and automatic contacts. |
| **FlowHMR → Fast SAM Acceleration** | Two independent, default-on switches: **Adaptive Batching** and **GPU Hand Crops**. Automatic selects 1, 2, 4, 8 or 16 frames from memory headroom and image size; Low Memory or disabling batching uses one frame. Used only with Fast SAM, on Reconstruct 3D. Tooltips and the [FlowHMR guide](models/FLOWHMR.md#controls) explain speed, VRAM and output differences. Batch selection and OOM retry messages appear in Console and CMD. |
| **FlowHMR → Camera / Sampling** | Start with Moving / Unknown - Auto for moving or uncertain cameras. Static Camera skips camera estimation for locked footage. Explicit frame intervals control camera analysis only; denser sampling uses more memory. |
| **GVHMR → HTD Refine** | Improve body-motion consistency over time. Runs before FootMR when both are enabled. |
| **GVHMR → FootMR** | Refine ankle angles from foot observations. Separate from foot contact locking. |
| **GVHMR → HandFlow** | Add finger tracking while preserving the selected body and wrist motion. Requires additional weights and MANO. |
| **Analyze 2D** | Prepare GVHMR refinement observations separately. Reconstruct 3D includes this step when needed. |
| **MAMMA → Camera Calibration** | VGGT-Omega + GeoCalib estimates fixed-camera placement and background vertical direction before 3D fitting. Manual Camera Data uses your compatible JSON/MAMMA YAML calibration without automatic gravity changes. |
| **MAMMA → Import Camera Data...** | Import calibration with camera names matching video filenames without extensions, original pixel intrinsics and metric translations. See the exact [camera format](MAMMA.md#manual-cameras). |
| **MAMMA → People** | Default 0 automatically detects people during reconstruction; set 1–8 if someone is missed or background people are included. Count each person once across all cameras. Find person is hidden. |
| **MAMMA → Reference Height (m)** | Numeric approximate standing height, default 1.70 m. Sets VGGT scene scale, not every actor's height; unused with metric manual calibration. |
| **PromptHMR → People / Interaction Prompt** | People defaults to 2; 0 detects automatically up to eight. Interaction Prompt defaults on; off follows the official default video path. Fixed cameras only; no animated fingers. Both settings apply on Reconstruct 3D. See the [PromptHMR guide](models/PROMPTHMR.md). |
| **Resource Mode** | Automatic uses the fast path when GPU memory is sufficient and selects supported Low Memory paths when needed. Low Memory uses CPU loading/offloading and existing bounded image processing; it can be slower and use more RAM. The selection and reason appear in Console and the CMD window. |
| **Prepare Preview** | Rebuild mesh/skeleton previews from existing motion without running the AI model again. Useful for older projects. |
| **Cancel Job** | Stop queued processing; completed results are kept. |

Processing shows the current step, progress and elapsed time. **Step left** estimates
the current step; **Total left** estimates the queue when timing data is available.
Use **Console** for details and errors. Loading or unmeasured work shows activity
instead of an invented percentage. FlowHMR is limited to 5,000 frames per take
(about 166.7 seconds); available memory may require a shorter clip. MAMMA's
adapter accepts 4–1,800 common frames and at most eight people. It requires
two or more synchronized fixed-camera videos, with no fixed camera-count
maximum. PromptHMR accepts one fixed-camera video, 4–1,800 working frames at
30 fps and up to eight people. Four thumbnails per row is only a layout choice. These limits do not
guarantee that a workload fits RAM or VRAM.

Processing is allowed to start without a predicted RAM/VRAM minimum. While it
runs, the app monitors actual available RAM and Windows allocation headroom.
If either becomes low, it stops the processing job and shows the available memory
in the error message. The app also reports actual CPU/GPU allocation failures.
Completed stages are retained for retry. ExMR Low Memory prepares Sapiens' existing
FP16 layers on CPU and loads body/hand weights on CPU before GPU transfer; its
body/hand feature batches remain 8. FlowHMR Low Memory offloads SAM and VGGT
transformer blocks. MAMMA Low Memory offloads VGGT camera blocks; its masks,
landmarks and fitting retain their existing bounded settings. GVHMR and add-ons
retain their adaptive image batches and full-context attention handling.
Automatic chooses these paths per operation from the current GPU budget and
can retry a supported operation once after CUDA OOM. It does not change the
saved Automatic selection, precision, video resolution, frame count or fitting
iterations. CPU offloading needs RAM: actual RAM shortage can still stop the job.
PromptHMR keeps the validated one-image batch and full temporal context in both
modes; extra block offloading is not implemented. Trim or close other GPU apps
if it cannot fit. No mode guarantees that every model or video fits. A sudden large
allocation can still fail before the next monitor check.

Low Memory primarily reduces **VRAM**, not system RAM. In earlier short paired
checks, ExMR fell **5.68 → 3.87 GiB** allocated VRAM while RAM working set changed
**11.88 → 11.94 GiB**; FlowHMR fell **6.10 → 2.77 GiB** while RAM rose
**8.42 → 8.75 GiB**. GVHMR's short test had no meaningful whole-job saving.
MAMMA's paired full-pipeline saving was not recorded; only the shared VGGT
component has a controlled offload comparison. See [conditions, times and component measurements](PERFORMANCE.md).
These are observations, not minimum requirements. The newest CPU prefetch is
disabled in Low Memory; FlowHMR SAM also uses one-frame batches in that mode.

GPU allocator budgets use ComfyUI's external-use margin: 600 MiB on Windows,
or 700 MiB on GPUs larger than 15 GiB, subtracted from currently free VRAM.
The five offline model paths use this rule; FlowHMR/ExMR no longer subtract fixed
3/2 GiB or impose an additional 95/90% ceiling. Inference working memory is
part of the allowed budget. This is not a guarantee of free VRAM: other apps
and allocations outside PyTorch can still consume it.

## Motion Correction

Contact correction reduces sliding when feet or palms rest on a support. It uses
existing motion without rerunning AI inference. Original and corrected results
are kept separately. Enable Correction and Foot default on for new work; Hand
defaults off. Existing projects retain saved selections.

| Setting | Function and meaning |
| --- | --- |
| **Enable Correction** | Show the corrected result; off shows the uncorrected result. |
| **Correction Mode → Flat Ground** | Use one correction floor for level ground. |
| **Correction Mode → Multi Level** | Keep contacts at their inferred heights for stairs, slopes and changing levels. The single floor controls are inactive. Terrain geometry is not reconstructed. |
| **Foot / Hand** | Correct only checked contact types. Disabling one retains its saved timing edits. Hand correction adjusts the arm for support contacts while retaining hand orientation. |
| **Lock Preset** | Weak allows more original movement; Normal balances holding and smooth transitions; Strong holds more uncertain contacts and may resist intentional sliding. |
| **Auto Floor Height** | Estimate the contact floor. Turn off to enter a manual height. |
| **Floor Height (World Y)** | Floor height in metres along the vertical axis. Higher values raise the correction floor; the gray reference grid stays at zero. |
| **Pelvis Height Offset (from Auto)** | Zero keeps automatic hip adjustment. Positive values undo some automatic lowering; negative values lower the hips further during contact. Raising can weaken foot locks. |
| **Apply Correction** | Commit current settings and timeline timings, update the source preview/export, and recalculate linked characters. |

Edit **L foot / R foot / L hand / R hand** tracks below the video track. Drag empty
space to add an interval, drag it to move, or drag its edges to resize. Delete
removes a selected interval; Ctrl+Z / Ctrl+Y undo/redo. **Reset Locks** restores
automatic timings. The timeline's **?** has full editing instructions.

Apply pending changes before export; stale corrected export is blocked. Automatic
hand detection is intended for stable support such as a floor, or raised supports
in Multi Level. Inspect its timing. Hanging poses, object interaction and larger
animation edits are best handled in a 3D animation editor. Contact detection is
an estimate and can leave residual sliding or penetration.

## People, preview and navigation

| Control | Function |
| --- | --- |
| **Find person** | For ExMR/FlowHMR/GVHMR, discover people for explicit selection. Without it, tracking selects one person and may switch identity during overlap. MAMMA/PromptHMR track people during reconstruction; this button is hidden. |
| **People checkboxes / selected row** | Checked rows control visibility and ExMR/FlowHMR/GVHMR reconstruction targets. MAMMA/PromptHMR reconstruct their detected/requested people together. The selected row controls timeline and contact editing. |
| **Character checkboxes / selected row** | Show or hide imported characters independently of source people. Select a character to export it; right-click it for Retarget, Export Character or Remove Character. |
| **People → ?** | Open instructions for people selection, names, colors and takes. The list footer shows only current status or warnings. |
| **Rename... / color square** | Right-click a row to rename it; click its square to change display color. The name is also used for FBX filenames and rig names. |
| **Take selector** | Choose a visible section when a person leaves and reappears. Long unseen gaps are separate takes. |
| **Overlays / Bone Overlay** | Show video observations / source export bones. Bone Overlay is drawn above source meshes and imported characters. Display only. |
| **View direction / Frame All** | Choose a camera direction / fit visible people in the 3D view. Left drag orbits, middle/right drag pans, wheel zooms. |
| **Floor Grid / Correction Floor** | Show the gray zero-height reference grid / orange correction floor preview. Neither applies correction. |
| **Motion Path / Follow Actor** | Show the person's path / move the viewing camera with the person. |
| **Lock** | Show active hand/foot contacts as translucent horizontal discs in timeline colors. Hand discs are at palm height, including raised supports. Display only. |
| **Name** | Show list names above visible people and imported characters. Text stays upright and the same screen size as you move the camera. Rename a person from the list's right-click menu. |
| **Timeline ruler / blue frame box** | Click or drag to seek. Track editing does not seek. Wheel zooms; middle drag or Shift+wheel pans. |
| **Playback / frame number / speed** | Space plays/pauses; arrows step one frame, Shift+arrows one second. Enter a frame number to jump. Playback speed affects preview only. |
| **Timeline Frame All / Reset Trim** | Fit the full timeline / restore the full clip range. |

ExMR/FlowHMR/GVHMR process checked people independently; MAMMA and PromptHMR
reconstruct people together from their respective inputs. Results are shown in a shared scene.
People share correction mode/floor; timings and pelvis offsets remain per person/take.
Apply Correction updates all available takes, including hidden ones. Relative
depth and contacts between people are estimates; collision avoidance is not provided.

## FBX and Blender

**File → Export...** opens source-motion options: choose **FBX (.fbx)**,
**BVH (.bvh)** or **Blender project + video (.blend)**. The optional
**Add T-pose before motion** defaults off.
For BVH, choose **Generic / Blender** (default) or **3ds Max Biped** in the
**BVH profile** list.
The selected, applied result is exported. Each checked
person/take gets a separate named file and rig; duplicate filenames get suffixes.
People exports use the viewport's shared placement. Takes retain their own frame
counts; align their start times when combining different source intervals.

Source-person exports use a 52-bone hierarchy with **Hips** as root and plain
anatomical names by default. The Biped profile renames exported joints as described
below. It retains
actor lengths, SMPL-neutral curved limbs/feet, calibrated rest axes and straight
open fingers. FBX stores rest transforms separately. With the T-pose option off,
no extra frame is inserted. When enabled, one reference animation frame is placed
before the original motion, at its initial root position; original frames shift
by one and retain their poses, frame rate and intervals. The option does not
replace the bind pose or edit project motion/contact timings.

BVH uses the same 52-bone hierarchy, metre units and Y up, including finger bones
and terminal tips. Import into Blender with **Scale 1 / Forward -Z / Up Y**.
The **3ds Max Biped** profile follows the PR #18 conversion: it uses Character
Studio names (for example, Spine becomes Chest and LeftArm becomes LeftShoulder)
and uniform **ZXY** rotation channels. All 52 joints, fingers, offsets, terminal
tips and keyed poses are retained, with no unit conversion or source-cache edits.
Changing Euler order can change interpolation between keys. Use Generic/Blender
when preserving the existing channel encoding is important. The optional leading
T-pose works with either profile and retains the first motion frame's root position.

In 3ds Max, create a Biped and load the file through **Motion > Motion Capture**.
See [Autodesk's BVH import steps](https://download.autodesk.com/global/docs/motionbuilder2014/en_us/files/Importing_Importing_.bvh_files_into_3ds_Max.htm).
This compatibility profile was checked numerically and imported in Blender;
3ds Max itself was not available for validation. Biped's own retargeting and limb
constraints may affect the result.

BVH cannot store FBX Pre/PostRotation or arbitrary rest-bone roll; the rest axes
can consequently look different. Use FBX when explicit rest transforms are needed.

**Export Character...** preserves the imported target character's own hierarchy
and original rest pose; it does not replace them with the 52-bone source rig.
Character FBX animation is stored on bones, with no armature-object keyframes.

In Blender use **Scale 1**, **Automatic Bone Orientation off**, **Use Pre/Post
Rotation on**, **Ignore Leaf Bones on** and **Animation Offset 1**. The file supplies
frame rate and axes. **Armature Data → Rest Position** shows the rest pose.
The first frame in **Pose Position** shows animation, which can differ from rest.
Use the character workflow above for supported humanoids, or retarget the source
FBX in your animation software.

### Portable Blender projects

Choose **Blender project + video (.blend)** in **File > Export...**, then choose
an output filename. The exporter creates a **new subfolder** containing the
`.blend`, `video.mp4`, `camera_report.json` and instructions. Existing folders are
retained; a numbered folder is chosen when needed. For several checked people
or takes, each gets its own portable folder. The selected applied motion and
shared placement are preserved. Original inference, project files, contacts and
camera receipts are not rewritten; stale results remain blocked.

Open the `.blend` in Blender 4.5 or newer. **Numpad 0** enters the source-camera
view and **Space** plays. The skeleton is editable; its object has no animated
transform. The synchronized source video is a viewport camera background, not
a render background. Move the **whole folder**, keeping the video beside the
project. The app uses its existing Blender runtime to build the bundle; no
additional runtime or AI inference is needed.

**Export saved camera tracking** defaults on. FlowHMR/ExMR can use the saved
rotation and approximate movement estimate. **Camera movement scale** defaults
to the current display scale; 0 fixes the center. **Refine Camera Rotation**
starts with the corresponding sidebar setting and exports only validated
background corrections. Disabling tracking exports the first saved camera as a
fixed camera. These choices affect only the exported camera, never body motion.
Movement scale and reconstruction error can prevent exact video alignment; this
is not measured camera calibration. The export report records the choices.
An optional T-pose adds one animation frame and delays the video/camera timeline
by one to keep the original motion and video paired.

### MAMMA whole-scene Blender projects

When MAMMA is selected, the Blender format also shows **MAMMA whole scene: body
meshes and all cameras**. Check it to export checked people and their visible
takes together, all saved calibrated cameras, synchronized video panels, and a
separate camera-view scene for each camera. The original common timeline is
retained; the T-pose option is unavailable for this export.

Body surfaces are baked **PC2 mesh caches**, including the selected applied
correction. Reference armatures are separate editable skeletons; they are **not
deformation rigs** for the meshes. Keep the `.blend`, `videos/` and `meshes/`
together. Changing a reference skeleton does not deform its cached body.
The presentation floor/lights are aids, not reconstructed scene geometry.
Camera backgrounds are viewport references; the overview's video panels are
image-textured geometry. Each camera scene excludes those panels.

This mode needs the saved MAMMA inference/calibration, matching undistorted
working videos and current body-mesh/skeleton previews. Use Apply Correction or
Prepare Preview if requested; export does not rerun MAMMA. Nonzero distortion
or unsupported camera skew is rejected rather than silently exported incorrectly.
PC2 disk usage is about **120 KiB per frame per person** (10,475 vertices),
in addition to videos, skeleton animation and the Blender file. Copies and
mesh baking run outside the GUI with visible activity/progress and cancellation.

## Saved cameras and Camera Preview

For reconstructed offline motion, **Overlays** above the left video preview also
shows the reconstructed skeleton projected through the selected saved camera.
It defaults on and is independent of the right viewport's **Bone Overlay** and
**Camera** switches. MAMMA projects all visible people into the selected camera
thumbnail. No new reconstruction or Camera Preview selection is needed to enable it.

For a separate alignment view, choose **Camera Preview** in the 3D viewport's
view selector. It shows the current source video with the displayed source
skeleton projected through the saved camera. **Bone Overlay** controls the bones.
Playback and seeking share the existing decoded video frame; no second video
is decoded. For MAMMA, select a camera thumbnail to inspect that view. Middle-drag
and wheel pan/zoom the preview; **Frame All** restores its image framing. Choosing
a normal view returns to the 3D scene.

Enable **Camera** below the 3D viewport to show saved cameras as wire frustums
with an image-top triangle. This switch defaults off and affects display only.
It does not recalibrate cameras, alter motion/correction or reframe the 3D view.
Camera motion follows the source timeline; available displacement paths are
bounded display samples. Saved cameras can be outside the current 3D framing;
zoom out or orbit to inspect them.

- MAMMA displays all saved fixed camera calibrations in the fitted shared world.
- FlowHMR displays saved rotation and the estimated VGGT displacement path.
  Its displacement has no measured metric scale, so the path is approximate.
- ExMR in Moving Camera mode can estimate camera displacement from the original
  paired world/camera root positions. This includes inference error and is not a
  measured camera trajectory. Static Camera mode retains a fixed position.
- When **Camera** is enabled in the ordinary 3D view, **Motion x** controls the
  estimated movement display: 1 uses the estimate, 0 keeps the camera center fixed.
  The default is 1. This affects camera gizmos and their path only; source video
  overlays keep the calibrated projection and exports/motion remain unchanged.
  ExMR movement requires a reference take covering the complete video; otherwise
  the app shows rotation only with a notice.
- Other offline results display their committed reference camera when available;
  this does not imply a complete moving-camera reconstruction.

FlowHMR and ExMR offer **Refine Camera Rotation** below their camera setting.
It defaults off. When enabled, the app optionally post-processes saved moving-camera
rotation using background features, following PR #18. It excludes known actor boxes,
validates against held-out matches, accepts corrections up to 2 degrees, and
interpolates corrections between sampled frames. Rejected samples retain the
original rotation. This can help small camera errors but may do nothing when
background texture/matching is poor or camera motion is large. Static-camera
results are skipped.

The operation runs on the CPU when saved cameras are needed for an enabled video
overlay, Camera Preview or camera gizmo. Progress and elapsed time are visible;
the accepted count and processing time appear in Console/CMD. Disabling the option
restores the original saved camera immediately through the background loader.
Validated results are reused within the session while input stamps remain current.
The checkbox is kept in workspace view settings when the workspace is saved.
No extra models, packages or runtime installation are required.

Refinement affects the video bone overlay, Camera Preview and displayed camera
orientation only. Camera centers/movement scale, original camera files, source
poses, contact correction and FBX/BVH animation remain unchanged. With the option
off, FlowHMR retains its saved VGGT-Omega tracking and ExMR its saved SimpleVO
tracking without this additional operation. Failures keep the original camera
and display a notice.

The preview uses the saved calibration (or its optional display-only background
rotation refinement), without fitting cameras to corrected bodies. Correction can intentionally move bones away from the video. Inference,
calibration and body-fit errors remain visible. Calibrations with lens distortion
currently use a pinhole overlay and show a limitation warning. Missing/stale
camera data shows a notice; prepare the selected preview instead of substituting
another result. Camera Preview is unavailable in PEAR live mode.

## Development and license

See [Build from source](BUILD.md) for environment setup, editing and release builds.
Original RGB MoCap code/documentation uses the [MIT License](../LICENSE).
Third-party code, derived modifications, weights and body data keep their own
terms; see [Third-party notices](../THIRD_PARTY_NOTICES.md).

## PEAR live preview

**Experimental support. Tracking and pose estimates may be unstable.**

Run `install.cmd` with the app closed; PEAR support is included. PEAR reuses
the shared runtime and adds only pinned source/utilities; no compiler is required.
PEAR tracks webcam or video input in real time; it may skip frames to limit delay.
Prepare its checkpoint, detector and registered SMPL-X 2020/FLAME bundle in Model
Manager. VGGT and GeoCalib are additional files for Camera Align.
Select **PEAR · Live (Experimental)** and click **Start Live Tracking**. It attempts
to align one or more fixed cameras before tracking; keep the person and floor
visible during alignment. If alignment fails or its support/models are missing,
tracking continues from the selected camera with a visible warning. It does not
fuse unaligned views or apply aligned live foot lock. Retry with **Camera Align**.
**Stop** ends tracking and releases the models and webcam. With **Webcam: None**,
open a video and seek/play its timeline to infer the current frame. Alternatively,
select a connected webcam before starting. While tracking is running, input
changes automatically align the new configuration. Ordinary seeks and thumbnail changes reuse the loaded tracking model. Rotation and smoothing apply to both sources. Under **One Euro Filter**, select
**Bone Rotation** and **World Position** independently (both default on). Its
World Position checkbox controls position smoothing; the separate top-level
World Position switch controls whether the pelvis moves at all.
Selecting a webcam shows an image preview without loading PEAR; Start is needed
for pose tracking. After Stop, reselect a webcam or Refresh to preview again.
Webcam capture stays connected during Start, model loading and Camera Align.
Opened videos can play, pause and seek during loading/alignment; alignment uses
the sample time selected when Camera Align was clicked.
All webcam selectors are grouped below the video in a numbered Camera / Device /
Rotation table, with Count and Refresh together above it. Each webcam has its own
Rotation field beside its selector. For
videos, select a thumbnail and use the first Rotation field; other videos retain
their angles. Changing any angle requires Camera Align again.
The live view shows fixed-length body and finger bones without per-frame body
mesh calculation. **Reset Bones** recalibrates their lengths from the displayed
pose, pauses video playback and realigns cameras while preserving the viewport camera.
**World Position** is on by default. Turn it off to hold the pelvis at its current
position, including height, while retaining pelvis rotation and joint motion.
**Foot Lock** defaults on, becomes active after successful alignment and applies small causal leg/pelvis corrections with smooth lock/release transitions. Fixed lengths and ankle/toe orientations remain; turning World Position off keeps the pelvis fixed, even if a foot cannot reach. Poor calibration and source jitter can still cause sliding. There is no Find person, recording, offline correction, retargeting or animation export
in this live mode. Switching models releases the live worker and device.
For multiple fixed views, select several synchronized videos or set the webcam
**Count** and assign distinct devices to its slots. Thumbnails select the main
preview. Start attempts **Camera Align** using VGGT/GeoCalib, including one camera;
alignment errors no longer block selected-camera tracking. Reset Bones also
triggers an alignment attempt. Synchronization and
background camera/floor estimates are approximate. See the
[PEAR guide](models/PEAR.md) for setup, resources and limitations.


## International filenames

Video, project, character and export paths support Unicode filenames, including
Japanese, Korean, Chinese, Cyrillic and Western European accents. The Windows
launchers and Python workers use explicit UTF-8, and image file I/O avoids
OpenCV's Windows filename code-page limitation. The interface remains English.

If a text-encoding error occurs, record the complete error and the operation
that triggered it. Updating code is preferable to renaming all files or changing
the Windows system language. VMD retains its format-defined CP932 name fields;
this restriction is separate from the filename used to save it.
