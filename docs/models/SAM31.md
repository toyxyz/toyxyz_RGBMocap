# Optional SAM 3.1 person tracking

SAM 3.1 is a **Find person** tracker for **ExMR, FlowHMR and GVHMR** on
Windows. It fills the People list using a text prompt and optional clicks.
The selected motion model then reconstructs the people you check.
**Default** remains available and uses the standard person tracker.
MAMMA, PromptHMR and GEM-X use their own tracking workflows. PEAR uses its
live detector.

## Setup

1. Run `install.cmd` in the app folder. Close RGB MoCap, then run this command
   from the same folder and review the upstream terms when prompted.

   ```cmd
   Setup_SAM31.cmd
   ```

   Setup needs Git, internet access and an NVIDIA CUDA GPU with native
   bfloat16 support. It reuses the installed ExMR Python and Torch and adds
   a separate `runtime_sam31/` overlay. Setup downloads pinned wheels and
   verifies CUDA attention. [Meta SAM terms](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/LICENSE)

2. Reopen RGB MoCap, select **ExMR**, **FlowHMR** or **GVHMR**, then choose
   **SAM 3.1** in the selector next to **Find person**. Open **Model Manager**
   and click **Auto Download** on **SAM 3.1 Person Tracking (Optional)**.
   Use that row directly. **Download All** skips the optional checkpoint.

3. For manual download, use the public Comfy-Org FP16 file
   `sam3.1_multiplex_fp16.safetensors` and click **Import...** on the same row.
   The file is about **1.75 GB (1.63 GiB)** and is checked by SHA-256.
   The default location is `models/sam31/`. The linked public file is the
   one the app downloads and does not require a Meta checkpoint access token.
   [Comfy-Org FP16 checkpoint](https://huggingface.co/Comfy-Org/sam3.1/blob/5febd4769e8802cdfdb75e1f733abd8c68434a85/checkpoints/sam3.1_multiplex_fp16.safetensors)

## Track people

1. Open a video and trim it to at most **1,800 working frames** at 30 fps
   (about one minute). If you plan to mark people, pause at a clear frame
   inside the trim where they are visible.
2. Select **SAM 3.1** and click **Find person**. Enter a text prompt and
   optionally mark people in the dialog.
   - **Text only**. Leave the default `person` or use a short description
     such as `person in a red shirt`. Tracking starts at the beginning of
     the trim and keeps matches.
   - **Text and clicks**. With the same text prompt, click once on the torso
     of each person you want to keep. The text finds candidates in the marked
     frame and each click selects a matching mask. The tracker follows those
     people across the trim. Right-click to undo the last marker or use
     **Clear Markers** to remove them all.
3. Click **Track**. When the People list is populated, check the people you
   want to process and click **Reconstruct 3D**.

## VRAM use

SAM 3.1 uses more VRAM than the Default YOLOv8x + OSNet tracker. In separate
single-video SAM 3.1 tracking tests, the `person` prompt used **11.70 to
14.01 GiB** peak PyTorch-allocated VRAM on clips of 286 to 568 working frames.
The exact increase over Default has yet to be measured on matching inputs.

In a separate MAMMA experiment with three 14.2-second videos, peak allocation
during mask generation was **10.68 GiB with SAM 3.1** and **1.77 GiB with
SAM 2.1** under the tested settings. Those measurements describe the mask
stage in that experiment. The app's MAMMA workflow uses SAM 2.1.

These figures describe observed peaks. The VRAM needed for a given run can
vary with the footage and tracking settings. Find person runs separately from
3D reconstruction.

## Troubleshooting

- If the app asks for SAM 3.1 setup, close it and run `Setup_SAM31.cmd` again.
  If the optional checkpoint row says **Missing**, download or import that
  file in Model Manager.
- A broad prompt can include people in the background. Use a more specific
  description or mark the intended people in a clear frame. Long occlusions
  can still cause a lost track or identity switch. If a marker misses, retry
  on a frame where the torso is visible.
- If memory runs low, use a shorter trim or switch to **Default** and run
  **Find person** again. Resource Safety applies to SAM 3.1. Videos and
  prompts are processed locally.

See [recorded validation](../VALIDATION.md#sam-31-person-discovery-2026-11-10)
for the tested workflows.
