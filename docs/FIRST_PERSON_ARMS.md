# ViDiHand2 first-person hands, head and optional arms

ViDiHand2 is an experimental desktop workflow for first-person hand and head
capture. Prepare the ViDiHand2 source and model files below. Then
choose **ViDiHand2** in **AI Motion Model**, open a first-person video, choose
a camera mode and select **Reconstruct 3D**. Start with a short clip. The worker
limits the decoded RGB sequence to 2 GiB, and CUDA memory requirements vary
with the clip and the GPU. Use the timeline trim for longer input.

The Movie Clip overlay shows the tracked hands and fingers. The 3D Viewport
shows both hands as MANO meshes and a head bone driven by camera translation
and rotation. Use **MANO Mesh** and **Bone Overlay** to toggle the hand surfaces
and the skeleton independently.
The head includes a fixed camera mount offset. **Camera** shows the saved
camera path and starts enabled when opening a ViDiHand2 take. **Camera
Preview** projects the saved hands using the matching camera and intrinsics.
Use **Floor Grid** to show the world reference plane at Y = 0.

- **Automatic / Moving** estimates the camera track with VGGT-Omega and
  requires its shared checkpoint. It uses VGGT-Ω OPT on CUDA GPUs with native
  BF16 support and standard VGGT-Omega on other supported NVIDIA GPUs.
  Monocular camera scale and trajectory can drift.
- **Fixed Camera** uses a stationary camera pose and estimated pinhole intrinsics.
- **Import Camera JSON** accepts ViDiHand2 calibration with matching image
  dimensions, pinhole intrinsics and Z-up world-to-camera extrinsics in metres.
  With intrinsics alone, the workspace uses fixed camera coordinates.
  Reopening a take uses its saved camera track. To reconstruct a manually
  calibrated clip after reopening it, select its calibration JSON again.

After reconstruction, select **Adjust Camera** to open the first processed
frame with the camera gizmo. Drag an outer arrow to move along a world axis.
Drag an inner ring to rotate around that axis.
The camera track, head and hands update together in the 3D preview.

Select **Apply** to save the adjustment across the take. **Reset** returns to
the camera track before manual adjustment and can be saved with **Apply**.
**Cancel** discards the preview. Existing takes keep their previous Tilt Offset
as part of the saved camera track.

## Arm IK

Use a fixed or steady first-person camera for the best results. Enable
**Estimate Arms After Tracking** to include shoulders and elbows in a new
reconstruction. For a saved take, select **Apply Arm IK** to reuse the
prepared video, hand tracks and camera. After using **Adjust Camera**, run
**Apply Arm IK** again if you want to fit the shoulders to the new orientation.

Set **Upper Arm Length** and **Forearm Length** in centimeters before
applying IK. They default to 32 cm and 28 cm and apply to both arms throughout
the take. A successful fit saves the lengths with the arm track, restores them
when the project is opened, and uses them in the viewport and native FBX.
Enable **Auto Arm Length** to let the solver estimate a shared arm reach
from both wrist tracks and the camera. It uses a high percentile of the
observed reach and includes sustained extensions across the take. It uses
neighboring frames to reject isolated tracking spikes, then splits the
result using the default upper-arm to forearm proportion. It compares torso
pitches for estimated cameras so a camera orientation error is less likely
to inflate the length. The estimate is saved after a successful fit. Turn the
option off to edit the lengths. Candidate lengths are checked against the
whole take to reduce gaps when the hands are tracked. Large changes in camera
position or uncertain wrist depth can still leave frames outside the arm's
reach or without a plausible elbow angle.

The solver anchors the shoulders to the camera, estimates elbow direction
from the video and smooths the elbow swivel within the two-bone joint limits.
It keeps the tracked wrists and finger poses. A local shoulder prior may be
loaded from `.setup/vidihand2-shoulder-prior.npz`. The solver uses
camera-relative shoulder offsets if the file is absent.
With manually entered lengths, if an automatic or fixed camera yields zero
valid arm poses, the solver tries a shared torso pitch correction for the clip. It keeps the saved
camera alignment, hand positions and selected arm lengths. Imported
calibrations use their saved orientation to fit the arms.

The arm bones appear in the **Bone Overlay** and **Camera Preview** when a
valid pose exists. Use **File → Export...** to save the native first-person
skeleton as FBX. With Arm IK, the skeleton includes the camera root and each
solved arm chain. FBX tracks hold the last solved arm through gaps and keep the hands at
their saved positions. This can leave a gap at the wrist on unsolved frames.
The viewport shows just the hand for that side during these gaps.

The 3D Viewport and FBX use flexion and spread at the four finger roots
and flexion alone at the inner joints. The thumb root also retains an
estimated opposition twist. The wrist and knuckle positions are anchored to
ViDiHand2, and the finger chain follows the constrained rotations.

During a tracking gap, each visible hand keeps its last world-space pose in
the 3D Viewport and FBX. The viewport shows a hand from its first tracked frame.
For hand-only takes, FBX uses the first tracked pose for earlier keys and includes the wrist and
finger chains of each tracked hand plus the camera-driven head. Existing
ViDiHand2 projects read their saved hands and camera when opened.
The first reconstruction with mesh preview support prepares a local MANO
preview cache from the installed right-hand model. Existing takes can use this
cache when reopened. The MANO mesh is a viewport preview. The FBX contains
bones and animation keys and can be imported into Blender.

## Local runtime setup

1. Run `install.cmd` to prepare the standard Windows components.
2. In a new Command Prompt, run `git --version`. If Git is unavailable, install
   Git for Windows with command-line PATH support and open a new terminal.
   [Git for Windows](https://git-scm.com/download/win)
3. Close RGB MoCap and run `Setup_ViDiHand2.cmd` from the app folder. Setup
   fetches the pinned ViDiHand2 source with Git and checks CUDA BF16 attention
   through the installed FlowHMR runtime.
4. Open RGB MoCap and prepare the checkpoints in **Model Manager > ViDiHand2**.

If fresh setup reports **WinError 2**, first confirm `git --version` and
`where.exe git` in the same terminal, then retry setup. A base installation can
succeed even if Git is unavailable to this terminal. If Git works, read the
final setup error to distinguish a failed source download from a CUDA check.
Rerun `install.cmd` if setup reports a missing FlowHMR profile.

The default Windows worker shares the installed Python, PyTorch and CUDA
packages with FlowHMR. Setup obtains only the upstream source. The pinned
source and local settings stay in `.setup/` when updating RGB MoCap.
[Official ViDiHand source](https://github.com/NTUYWANG103/ViDiHand/tree/23cf80c292f2fb2db20caa1dee96516a63ee30a7)

### Optional external runtime

An existing `.setup/first-person-arms.json` still selects its external
Windows or WSL environment. To switch to the shared Windows runtime,
back up the file and set its contents to `{"mode": "managed"}`.

For an external WSL runtime, use the distro name shown by
`wsl --list --verbose` and absolute paths inside that distro. Install the
upstream requirements in that environment and confirm
`torch.cuda.is_available()` returns `True`.
[Official ViDiHand installation](https://github.com/NTUYWANG103/ViDiHand/tree/23cf80c292f2fb2db20caa1dee96516a63ee30a7#installation)

~~~json
{
  "mode": "wsl",
  "distribution": "Ubuntu",
  "python": "/home/user/ViDiHand/.venv/bin/python",
  "upstream": "/home/user/ViDiHand"
}
~~~

The app also accepts a native CUDA environment. Use the Windows interpreter
and repository paths for native mode.

~~~json
{
  "mode": "native",
  "python": "C:/ViDiHand/.venv/Scripts/python.exe",
  "upstream": "C:/ViDiHand"
}
~~~

Review the upstream model and MANO terms for your intended use. The
checkpoints are obtained separately through Model Manager.
[Third-party notices](../THIRD_PARTY_NOTICES.md#ai-tools-and-models) · [Citation](CITATIONS.md#vidihand2)

## Model Manager

Open **Model Manager > ViDiHand2**. **Auto Download** and **Download All**
can fetch the two public checkpoints from the
[official ViDiHand2 release](https://huggingface.co/WANG-XIXI/ViDiHand/tree/5117c7a34939cd6a2af0da12859237a38ede4276/vidihand_v2).
For an existing download, use **Import...** in the corresponding row. **Manual
Download...** links to the supported files.

| File | Use |
| --- | --- |
| `vdm.safetensors` (6.07 GB) | Video diffusion backbone and VAE |
| `projector.safetensors` (41.3 MB) | Hand decoder |
| `MANO_RIGHT.pkl` | Shared hand model. Sign in on the official MANO site, extract `mano_v1_2.zip`, and import the right-hand file. |
| `vggt_omega_1b_512.pt` | Shared camera model for **Automatic / Moving**. Use **Manual Download...** and **Import...**. |

Native and WSL workers read verified Model Manager files directly. Older
runtime installations can still use existing files in their own
`checkpoints/vidihand_v2/` and `checkpoints/mano/` folders. These files stay in
the runtime folder and are outside the Model Manager's storage inventory.
