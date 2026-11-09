# GEM-X

In our tests, GEM-X processed clips quickly, with relatively low overall motion quality.

GEM-X reconstructs the body and fingers of one primary person from one
fixed-camera video. RGB MoCap preserves the native 77-joint SOMA skeleton,
estimated body shape and evaluated SOMA MID surface. This workflow uses native
SOMA data throughout. The app applies one vertical offset to place the clip
near the display floor while retaining the original motion and frame-to-frame
height changes. The floor is an estimate.

## Setup

1. Run `install.cmd` on Windows and close RGB MoCap.
2. Run `Setup_GEMX.cmd` from the app folder. Review the upstream terms shown
   by the command. Allow **12 GiB** free during setup, plus space for the model
   files. Setup uses the app's existing ExMR CPython 3.12.10 base and installs
   its own PyTorch 2.10 CUDA 12.8, Transformers 5.3, ONNX Runtime and SOMA-X
   0.3.3 packages. The pinned [NVIDIA Warp wheel](https://pypi.org/project/warp-lang/1.19.0/)
   requires a Turing or newer NVIDIA GPU and an R580-series or newer driver.
   The setup uses prebuilt wheels and does not need Visual Studio or a CUDA compiler.
3. Reopen RGB MoCap, choose **GEM-X**, then open **Model Manager**. **Download
   All** gets nine required public files from NVIDIA and OpenMMLab. You can
   also import exact files previously downloaded. For manual detector import,
   extract `end2end.onnx` from the specified HumanArt YOLOX archive.

The installed runtime used **2.13 GiB** of additional storage with shared CUDA
files on the tested Windows installation. On a filesystem that cannot link
those files, the runtime uses about **5.08 GiB**. The models add **7.11 GiB**.
Setup reuses the ExMR interpreter and only shares DLLs from an exact SHA-256
allowlist. The different Torch binaries and Python packages are isolated.
Blender and FFmpeg use the base app's installation. `install.cmd` and
`update.cmd` do not run optional GEM-X setup. If the app asks for setup after
an update, rerun `Setup_GEMX.cmd`.

## Reconstruct and export

1. Use **File > Open Video**. Keep the camera fixed and the intended person
   visible. Choose a trim of 4–1,800 frames at the working rate of 30 fps.
2. Click **Reconstruct 3D**. GEM-X detects the primary person automatically
   and runs 2D observations, body features, temporal motion and native geometry
   evaluation. Play the result in the normal timeline. **Bone Overlay** shows
   the Native77 rig. Use **Save Project As** to retain the media and results.
3. Use **File > Export** and choose **FBX**, **BVH** or **Blender project + video**.
   FBX and BVH export the selected original or applied correction as a 77-joint
   SOMA skeleton with body and finger motion at 30 fps. The BVH uses the
   **Generic / Blender** profile, metres and Y up. **Place at origin** centers
   the first root horizontally while preserving height and travel.
   **Add T-pose before motion** inserts one frame using the shaped SOMA rest pose
   at the first motion frame's root position. The Blender bundle includes the
   editable rig, separate evaluated surface cache and synchronized **Input Video**
   scene. Keep the `.blend`, `surface.pc2` and `video.mp4` together.
4. Import a humanoid character through **File > Import Character**, right-click
   its row and choose **Retarget > Person 1**. Native SOMA rotations are
   transferred relative to the character's rest pose. The character keeps its
   bone lengths, and root travel is scaled by its leg lengths. Export the
   retargeted character in its usual format.

This integration supports fixed-camera, offline capture. Moving-camera
tracking, Camera Preview and separate EgoHand inference are unavailable for
GEM-X. The camera setup estimates motion scale and placement. Overlap or an unclear subject can
affect automatic selection.

## Motion Correction

**Enable Correction** selects the corrected result. New reconstructions run
correction when this option is enabled. For an existing result, enable it and
click **Apply Correction**. You can toggle between the original and corrected
poses at the same frame. Adjust the Foot and Hand lock intervals in the timeline
and apply again. Foot is enabled by default and Hand is optional. Flat Ground
uses one floor, and Multi Level retains individual support heights.

Automatic timing uses the movement of the SOMA soles and palms. Review the
intervals where the performer slides deliberately or uses an elevated support.
The shared anatomical solver uses the actor's measured joint positions and
bone lengths. SOMA-X rebakes the bones and surface together from those changes.
Applying correction uses the saved motion and four SOMA geometry assets. It
uses the existing base CPU solver and GEM-X CUDA runtime without running video
inference again. The corrected surface uses about 6.2 MiB/s of additional
project storage. FBX, BVH, Blender export and character retargeting use the selected pose.

## Resources and troubleshooting

**Automatic** uses an image batch of four with sufficient free VRAM and one
under memory pressure. **Low Memory** uses one. An actual CUDA allocation
failure can lower the batch and retry. **Resource Safety** uses the app's
existing memory monitoring and PyTorch allocation policy. The mode does not
change motion frame count or model precision.

One fixed-camera 9.8-second clip (294 frames) took about **104 s** through
verification, inference and mesh evaluation on an RTX 5090. Peak PyTorch
allocated memory was **3.48 GiB**. This is a measured clip, rather than a
minimum GPU requirement. The native mesh cache uses about **6.2 MiB per
second** of motion at 30 fps. Keep the project folder to preserve it.

Setup logs are in `.setup/logs/gemx-*.log`. Keep `runtime_gemx/`,
`.setup/gemx-runtime.json` and `runtime_exmr/`. After setup finishes,
`.setup/gemx-downloads/` can be cleared. Repair may download the files
again. If a model fails verification, reimport its official pinned file.

## Sources and terms

- The pinned upstream source retains the [Apache-2.0 license](https://github.com/NVlabs/GEM-X/blob/4aef1452e5df171deff4f293c08e643eb450981e/LICENSE)
  and its [component attributions](https://github.com/NVlabs/GEM-X/blob/4aef1452e5df171deff4f293c08e643eb450981e/ATTRIBUTIONS.md),
  including OpenMDW 1.1 and the SAM License where specified.
- NVIDIA's [GEM-X model card](https://huggingface.co/nvidia/GEM-X) identifies
  the NVIDIA Open Model License for its model. [SOMA-X assets](https://huggingface.co/nvidia/SOMA-X)
  are Apache-2.0. The [SOMA-X Python package](https://pypi.org/project/py-soma-x/0.3.3/)
  retains its own notices.
- The detector comes from the [OpenMMLab RTMPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose).
  Its HumanArt YOLOX weights retain the provider's applicable terms. This
  app downloads runtime source and model assets locally from their providers.
  Review the terms of the components and your footage before distributing outputs.

See [download pins](../../installer/gemx-models.json),
[runtime pins](../../installer/gemx-runtime-lock.json),
[third-party notices](../../THIRD_PARTY_NOTICES.md) and [research citations](../CITATIONS.md).
