# Command line (ExMR)

`RGBMoCapCLI.cmd` runs the single-video ExMR workflow without the window:
Find person (optional), Reconstruct 3D, Motion Correction and FBX export. It
starts the same worker stages, with the same arguments, environment and request
files, that the app uses for those buttons. A project made from the command
line opens in the app with its results, receipts and caches, and the reverse.

It is meant for batches and automation. Review the motion in the app as usual.

## Requirements

- A completed `install.cmd` installation. The command line uses the same
  private runtimes and the same `models`, `projects`, `cache` and
  `config\local.json` as the app.
- The ExMR model files prepared in **Model Manager**. Missing files stop the
  run before any worker starts and are listed by name.

## Examples

Run from the app folder, in Command Prompt or PowerShell:

```cmd
RGBMoCapCLI.cmd reconstruct "D:\clips\walk.mp4" --fbx "D:\out\walk.fbx"
RGBMoCapCLI.cmd reconstruct "D:\clips\walk.mp4" --fbx-profile ue5_mannequin --fbx "D:\out\walk.fbx"
RGBMoCapCLI.cmd reconstruct "D:\clips\two.mp4" --find-person --people 1,2 --fbx "D:\out\two.fbx"
RGBMoCapCLI.cmd correct "<project>\exmr" --preset strong --fbx "D:\out\walk_strong.fbx"
RGBMoCapCLI.cmd export "<project>\exmr" --fbx "D:\out\walk.fbx"
```

- `reconstruct` prepares the video, tracks one person and runs ExMR, Motion
  Correction and the motion export.
  - The project goes to `projects\` with the same name the app would give it.
  - Running the same video again reuses the completed stages.
- With `--find-person`, the Default tracker finds the people first. The checked
  people are then reconstructed (Person 1 by default; `--people` picks others),
  and each take gets its own FBX file.
- `correct` is **Apply Correction**: it saves the options and recomputes the
  corrected motion.
  - For a take found with Find person, or for its people group folder, the
    floor options are shared. Every reconstructed take in the group is
    corrected again, as in the app.
- `export` writes an FBX of an existing take. A people group folder is
  refused with the list of its take folders.
- `--json`, `--log`, `--quiet` and `--compute` can go before or after the
  command.
- `--json` prints a summary with the project, takes and FBX paths on stdout.
- `--log` appends the full worker output to a file.
- The exit code is 0 on success and 1 on any failure; Ctrl+C stops the
  running worker and its child processes.

## Options and defaults

The defaults are the app's built-in defaults for new work, and any starting
settings saved with **Defaults...** are applied as they are at **Open Video**.
Command-line options take precedence.

| App control | Option | Default |
| --- | --- | --- |
| Camera | `--moving-camera` | Static Camera |
| Smooth Motion | `--no-smoothing` | On |
| Enable Correction | `--no-correction` | On |
| Lock Preset | `--preset weak\|normal\|strong` | Normal |
| Correction Mode | `--mode flat\|multi_level` | Flat Ground |
| Foot / Hand | `--foot-lock 0\|1`, `--hand-lock 0\|1` | Foot on, Hand off |
| Auto Floor Height | `--floor <m>` (manual floor) | Automatic |
| Pelvis Height Offset | `--pelvis-offset <m>` | 0 |
| Resource Mode | `--compute` | Automatic |
| Trim | `--start`, `--end` (seconds) | Whole video |
| FBX profile | `--fbx-profile generic\|ue5_mannequin` | Generic / Blender |
| Target mannequin | `--ue5-target manny\|quinn` | Manny |
| Root movement | `--in-place` | Root Motion |
| Include Mannequin Mesh | `--include-mesh` | Off |
| Rest pose (generic) | `--rest-pose t\|a` | T-pose |

Resource Safety follows the app's saved choice. Face capture, RIFE
interpolation, SAM 3.1, BVH and Blender export are not available from the
command line.

## Running beside the app

The command line does not take the app's lock, so it can run while the window
is open. Do not process the same project from both at the same time, and keep
in mind that two reconstructions share the GPU's memory.

## Scope

Only ExMR is supported. The other models, the People panel's manual edits
(names, merges, contact-lock timing) and multi-camera work remain in the app.
A take edited in the app can still be exported or re-corrected from the command
line, and keeps those edits.
