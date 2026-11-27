# RGB MoCap command line

Run `RGBMoCapCLI.cmd` from the installed Windows distribution. It uses the
desktop's workers, saved settings and project results. Run `install.cmd`
first for ExMR, FlowHMR and GVHMR. GEM-X, PromptHMR, MAMMA, SAM 3.1 and Unreal
Live Link have separate setup commands. See the [user guide](USER_GUIDE.md)
and [model guides](models/README.md) for setup and required files.

## Reconstruct and inspect

```powershell
.\RGBMoCapCLI.cmd check --model flowhmr --json
.\RGBMoCapCLI.cmd models list --workflow flowhmr --json
.\RGBMoCapCLI.cmd reconstruct "D:\clips\walk.mp4" --model flowhmr --project "D:\projects\walk" --camera moving
.\RGBMoCapCLI.cmd reconstruct "D:\clips\duet.mp4" --model exmr --project "D:\projects\duet" --find-person --person 1 --person 2
.\RGBMoCapCLI.cmd project inspect "D:\projects\duet" --model exmr --json
.\RGBMoCapCLI.cmd open "D:\projects\walk"
```

| Model key | Reconstruction options |
| --- | --- |
| `exmr` | Static or moving camera, smoothing and motion correction. |
| `flowhmr` | Camera mode, sampling every 1, 2, 5, 10 or 30 frames, Fast SAM and HaMeR finger refinement. |
| `gvhmr` | Fixed camera, with optional HTD, FootMR and HandFlow. |
| `gemx` | Native SOMA reconstruction and motion correction in the dedicated runtime. |
| `prompthmr` | People count, interaction and camera mode. Moving camera requires the optional PromptHMR camera runtime. |
| `mamma` | Synchronized camera views with repeated `--camera-video`. Supply saved camera calibration with `--calibration`, or use automatic alignment. |

Run ViDiHand2 reconstruction in the desktop workspace. Its checkpoints
are available under `models list --workflow vidihand2`. [ViDiHand2 guide](FIRST_PERSON_ARMS.md)

Without `--project`, reconstruction uses the desktop's default project
location. An existing workspace must match the source video. `--start` and
`--end` use source-video seconds. `--rife` enables interpolation for eligible
lower-frame-rate input. See `reconstruct --help` for model, correction and
compute options. Omitted options use saved project settings, saved user
defaults or built-in defaults, as applicable.

For ExMR, FlowHMR and GVHMR, `--find-only` saves detected people without
reconstructing them. Use `--find-person` and repeat `--person` with numeric
IDs to select people for reconstruction. Add `--tracker sam31` for optional
SAM 3.1 tracking. Set `--sam31-prompt` for a text prompt, `--sam31-time` for
the marker time and repeat `--sam31-point X Y` for normalized image points.

## Correct, edit contacts and export

```powershell
.\RGBMoCapCLI.cmd correct "D:\projects\walk" --model flowhmr --preset strong
.\RGBMoCapCLI.cmd contacts list "D:\projects\walk" --model flowhmr --json
.\RGBMoCapCLI.cmd export "D:\projects\duet" --all-takes --output-dir "D:\out" --dry-run
.\RGBMoCapCLI.cmd export "D:\projects\duet" --all-takes --format blend --combined --output "D:\out\duet"
.\RGBMoCapCLI.cmd export "D:\projects\walk" --model flowhmr --fbx-profile ue5_mannequin --ue5-target quinn --output "D:\out\walk.fbx"
```

- Commands that operate on saved takes accept a workspace, model folder or
  take. Use `project inspect` to find take IDs and readiness. Select a group
  with `--take ID`, `--person ID` or `--all-takes`. Group correction updates
  available takes against the shared floor and applies explicit actor
  settings to the selected takes.
- `contacts set PROJECT RANGES.json` accepts two arrays for feet or four
  arrays for feet and hands, ordered left foot, right foot, left hand and
  right hand. Each range is a zero-based, end-exclusive `[start, end)` pair.
  Run `correct` after setting or resetting manual contacts.
- `export` uses the saved motion result unless you specify `--result`.
  `--output` names one output and `--output-dir` generates filenames for
  individual takes. `--format blend` writes a Blender bundle directory.
  Add `--combined` for a shared timeline or `--whole-scene` for a native
  MAMMA scene. FBX and BVH use file paths. Existing export destinations
  require `--overwrite`. The export plan reports unavailable takes and
  how to prepare them.
- `prepare-preview` refreshes saved rig and scene data. `preview PROJECT
  --output PREVIEW.mp4` renders the desktop viewport. It needs an
  OpenGL-capable Windows session and accepts an optional `--frames` limit.
  `project copy PROJECT --output NEW_FOLDER` copies the workspace and media.

## Refine camera rotation

Use `--vggt-opt` or `--no-vggt-opt` to set the lower-memory VGGT-Omega camera
path for FlowHMR and MAMMA. The built-in default is on. FlowHMR Static Camera
and MAMMA manual calibration bypass VGGT-Omega.
Combine this with `--compute low_memory` for CPU block offload.
[VGGT memory measurements](PERFORMANCE.md#vggt-camera-memory-with-opt)

FlowHMR and ExMR support `--refine-camera-rotation` and
`--no-refine-camera-rotation`. On `reconstruct`, either flag saves the
choice in the workspace. The refinement runs on the CPU when the saved
camera is used. It skips static cameras and keeps the original rotation for
samples that fail validation.

```powershell
.\RGBMoCapCLI.cmd reconstruct "D:\clips\walk.mp4" --model flowhmr --camera moving --refine-camera-rotation --project "D:\projects\walk"
.\RGBMoCapCLI.cmd preview "D:\projects\walk" --model flowhmr --refine-camera-rotation --output "D:\out\walk.mp4"
.\RGBMoCapCLI.cmd export "D:\projects\walk" --model flowhmr --format blend --no-refine-camera-rotation --output "D:\out\walk-original.blend"
```

Preview and animated Blender exports use the workspace setting when the
flag is omitted. On those commands, either flag overrides the saved choice
for that output. An explicit preview flag also shows the camera in the 3D
viewport. Blender bundles record accepted samples in `camera_report.json`.
Camera centers, body motion, and the source camera files stay unchanged.

## Face, character and live output

```powershell
.\RGBMoCapCLI.cmd face track "D:\projects\walk" --model flowhmr
.\RGBMoCapCLI.cmd face neutral "D:\projects\walk" --model flowhmr --person 1 --frame 5
.\RGBMoCapCLI.cmd export "D:\projects\walk" --model flowhmr --face livelink_csv --output "D:\out\walk_face.fbx"
.\RGBMoCapCLI.cmd character "D:\projects\walk" --model flowhmr --asset "D:\rigs\character.fbx" --output "D:\out\character.fbx"
.\RGBMoCapCLI.cmd reference-fbx "D:\projects\walk" --model flowhmr --output "D:\out\reference.fbx"
.\RGBMoCapCLI.cmd stream "D:\projects\walk" --model flowhmr --target unreal --face --wait 15 --loop
```

- `face inspect` reports tracking coverage. Frame numbers for `face neutral`
  start at 1. `face reset-neutral` clears the selected person's neutral frame.
  `export --face txt` or `export --face livelink_csv` adds facial data to the
  export. Facial export and Unreal streaming use smoothed curves by
  default. `export --raw-face` selects raw curves. The export result lists
  face files and reports any face-export errors alongside saved body files.
- `character` retargets one take to an FBX, VRM or PMX character. The output
  extension selects FBX or, for a PMX character, VMD. For VMD, use
  `--vmd-placement origin` to center the start and keep travel, or
  `--vmd-placement in-place` to remove horizontal travel. The default keeps the
  original placement. Use `--vmd-root-bone center` to put root translation and
  rotation on センター and leave the master root free for placement in MMD.
  This requires a compatible PMX hierarchy. The default is `root`.
  See the [VMD import notes](USER_GUIDE.md#character-retargeting) when verifying
  the master root in Blender.
  `reference-fbx` exports the body streaming reference rig.
- Use `stream --target blender` with the Blender receiver. `--port` overrides
  the Blender port. The receiver can request a bake of the saved take. Use
  `stream --target unreal` with the Live Link sender and receiver, and add
  `--face` for actors with face data. `--wait` allows time to connect and
  `--seconds` limits the run. Ctrl+C stops a loop. See the
  [desktop streaming setup](USER_GUIDE.md).

The former `pear` command now points to [RGB MoCap Live](https://github.com/toyxyz/rgbmocap_live)
for real-time tracking.

## Model files and batch jobs

Use `models list` to find model keys and shared workflows.
`models download KEY` runs the normal download worker. Use
`models import KEY FILE` for a manual download, with `--replace` to replace a
ready file. `models remove KEY --dry-run` lists affected paths and workflows.
`models remove KEY --yes` performs the removal.

`batch QUEUE.json` accepts a version 1 JSON file containing arrays of CLI
arguments. Run it with `--dry-run` to check the plan. A dependent export can
be marked deferred until its reconstruction or correction has run. Jobs run
sequentially and stop at the first error by default. `--keep-going` continues
after a worker or input failure. Outputs from completed jobs remain available.

```json
{
  "version": 1,
  "jobs": [
    ["correct", "D:\\projects\\walk", "--model", "flowhmr", "--preset", "strong"],
    ["export", "D:\\projects\\walk", "--model", "flowhmr", "--format", "bvh",
      "--output", "D:\\out\\walk.bvh"]
  ]
}
```

Commands with `--dry-run` plan their operations without writing project or
output files. `--json` emits a result with `schema_version` and `ok`.
Progress and worker logs go to stderr. Exit status is 0 for success, 1 for
a runtime or processing failure, 2 for invalid arguments or inputs, and 130
for Ctrl+C.

Writing operations share the desktop and setup lock. Close the desktop
before starting one from the same installation. Use a single installation
to write to a workspace. After cancellation, existing worker receipts
determine which stages can be reused.
