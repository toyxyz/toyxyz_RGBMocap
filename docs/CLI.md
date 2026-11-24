# RGB MoCap command line

`RGBMoCapCLI.cmd` runs ExMR from the installed Windows distribution. It uses
the desktop's worker runtimes, project files, cached results and source export
rules. Complete `install.cmd` and prepare the ExMR files in Model Manager.

Run the examples from the RGB MoCap folder in PowerShell.

```powershell
.\RGBMoCapCLI.cmd check --json
.\RGBMoCapCLI.cmd reconstruct "D:\clips\walk.mp4" --model exmr --project "D:\projects\walk"
.\RGBMoCapCLI.cmd reconstruct "D:\clips\duo.mp4" --project "D:\projects\duo" --find-person --person 1 --person 2
.\RGBMoCapCLI.cmd project inspect "D:\projects\duo" --model exmr --json
.\RGBMoCapCLI.cmd correct "D:\projects\walk" --preset strong
.\RGBMoCapCLI.cmd export "D:\projects\duo" --all-takes --output-dir "D:\out" --dry-run
.\RGBMoCapCLI.cmd export "D:\projects\duo" --all-takes --output-dir "D:\out"
.\RGBMoCapCLI.cmd export "D:\projects\walk" --fbx-profile ue5_mannequin --ue5-target quinn --output "D:\out\walk.fbx"
.\RGBMoCapCLI.cmd open "D:\projects\walk"
```

## Projects and takes

- `reconstruct` accepts a source video and an optional destination workspace.
  An existing destination must match the source video. Without `--project`,
  it uses the same default location as the desktop. Use `--start` and `--end`
  for a range in seconds relative to the supplied video.
- `--find-person` runs the Default tracker. Repeat `--person` with a numeric
  ID to set the selected people. When omitted, it reconstructs the checked
  people. New detections start with the first person checked.
- `project inspect`, `correct` and `export` accept the workspace, model folder
  or saved take. Use `project inspect` to list saved take IDs and readiness.
  For a group, select a scope with `--take ID`, `--person ID` or `--all-takes`.
  Repeat `--take` or `--person` to select several. A single take path already
  identifies that take.
- `correct` keeps project options you omit. A correction on a people group
  updates the available reconstructed takes against the shared scene floor.
  The selected take(s) receive the explicitly supplied actor options.
- `export` uses the saved motion selection unless you pass
  `--result initial` or `--result corrected_initial`. Use `--output` for one
  take or `--output-dir` for automatic filenames. The generated names include
  person IDs. An existing destination requires `--overwrite`. The report
  identifies incomplete or out-of-date takes and how to repair them.
- `open` hands a video or project to a new desktop window and reports the
  launch request. Close the existing window for this installation first.

## Settings, automation and safety

The priority is explicit options, saved project settings, saved user defaults,
then built-in defaults. See each command's `--help` for camera, smoothing, RIFE,
correction and resource mode options. Installation and optional runtime setup
use the existing setup commands.

`--dry-run` checks inputs without writing project or output files. Take
selection that depends on person detection is listed as deferred.
`--json` emits one JSON result with `schema_version` and `ok`. Progress and
worker logs go to stderr. Exit status is 0 for success, 1 for a runtime or
processing failure, 2 for invalid arguments or inputs, and 130 for Ctrl+C.

The CLI shares the desktop and install/update lock for writing operations.
Close the desktop before reconstruction, correction or export from the same
installation. Use a single installation to write to a workspace. If a job is
cancelled, the workers' existing receipts determine which stages can be reused.

This first version covers offline ExMR and source FBX. Use the desktop for
other models, face capture, character retargeting, other export formats,
preview video, live capture, streaming and visual edits.
