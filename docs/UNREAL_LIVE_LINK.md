# Stream motion to Unreal Live Link

This optional Windows Git/CMD feature sends the selected offline people's bones
through Unreal's built-in **Message Bus Source**. It follows playback and
scrubbing; it does not run live video inference. The default Stream target
remains **None**. Blender streaming keeps its existing port and add-on.

For file-based animation on stock Manny or Quinn, see
[UE5 Mannequin FBX](USER_GUIDE.md#ue5-mannequin-fbx).

## Prebuilt downloads

Each Windows 64-bit bundle contains the RGB MoCap Connect 0.5.0 receiver,
the RGB MoCap sender, install instructions and checksums. Match the exact
Epic Games Launcher engine build.

| Unreal Engine | BuildId | Download |
| --- | ---: | --- |
| 5.6.1 | `43139311` | [Bundle](https://github.com/toyxyz/toyxyz_RGBMocap/releases/download/unreal-live-link-win64-v0.5.0/RGBMoCapLiveLink_UE5.6.1_Win64.zip) · [SHA-256](https://github.com/toyxyz/toyxyz_RGBMocap/releases/download/unreal-live-link-win64-v0.5.0/RGBMoCapLiveLink_UE5.6.1_Win64.zip.sha256) |
| 5.7.4 | `47537391` | [Bundle](https://github.com/toyxyz/toyxyz_RGBMocap/releases/download/unreal-live-link-win64-v0.5.0/RGBMoCapLiveLink_UE5.7.4_Win64.zip) · [SHA-256](https://github.com/toyxyz/toyxyz_RGBMocap/releases/download/unreal-live-link-win64-v0.5.0/RGBMoCapLiveLink_UE5.7.4_Win64.zip.sha256) |
| 5.8.4 | `55116800` | [Bundle](https://github.com/toyxyz/toyxyz_RGBMocap/releases/download/unreal-live-link-win64-v0.5.0/RGBMoCapLiveLink_UE5.8.4_Win64.zip) · [SHA-256](https://github.com/toyxyz/toyxyz_RGBMocap/releases/download/unreal-live-link-win64-v0.5.0/RGBMoCapLiveLink_UE5.8.4_Win64.zip.sha256) |

These packages use your installed Unreal Editor. The prebuilt route runs
without a local C++ build. For a different Unreal build, use
[source setup](#build-the-sender-locally).

## Recommended MetaHuman connection

Use [RGB MoCap Connect](../unreal_plugin/RGBMoCapMetaHuman/README.md)
for the direct source-to-target panel. Use the receiver ZIP from the
[matching bundle](#prebuilt-downloads), or
[build the receiver](../unreal_plugin/RGBMoCapMetaHuman/README.md#build-and-verification).
Close Unreal and place the extracted `RGBMoCapMetaHuman` folder in your
project's `Plugins` folder. Replace the entire existing plugin folder when upgrading. Then complete sender setup below, reopen Unreal and use
**Tools > RGB MoCap Connect**. Set **Target type** to **MetaHuman** in 0.5.0
or later, select a source row and one MetaHuman actor, then click **Connect
selected actor**. Each row displays its targets and Face switch.
The later reference-FBX/IK instructions describe the optional manual workflow.
The old sample project is no longer distributed.

## Connect a UE5 Manny or Quinn mannequin

1. Install an RGB MoCap Connect **0.5.0 or newer** build matching your Unreal
   version. Set up the app sender as described below.
2. Place a UE5 Manny or Quinn Skeletal Mesh Actor or Character Blueprint in
   your level. Its mannequin mesh must use **Use Animation Blueprint** mode
   and positive uniform scale. A standalone mesh can leave **Anim Class** empty.
3. In **Tools > RGB MoCap Connect**, set **Target type** to **SK Mannequin
   (Manny/Quinn)**. Select a live person row and the actor, then click
   **Connect selected actor**.
4. Repeat for other actors. **Connected targets** shows each actor and its type.
   Changing **Target type** sets the type for the next connection. **Disconnect
   source** restores the original animation for the actors on that row.

The plugin reads the selected mesh's rest pose and bone lengths. It drives
body, fingers and the mannequin's hand and foot IK guides directly. **Ground
Contact** uses an estimated flat floor in component space. The mannequin
receives body motion only. On a shared source row, **Face** applies to the
connected MetaHumans. The receive-side setting does not change which Live Link
subjects the app publishes.

## Prepare once

### Install a prebuilt sender

Use the current RGB MoCap `main` checkout and complete the normal
`install.cmd` setup. For an existing installation, close the app and run
`update.cmd` first to get `--prebuilt` support. Extract the outer bundle and
keep the inner sender ZIP intact. Open PowerShell in the RGB MoCap app folder
and run the following, adjusting the path for your extracted sender ZIP.

```powershell
.\Setup_UnrealStream.cmd --prebuilt "C:\Downloads\RGBMoCapSender_UE5.8.4_Win64.zip"
```

Setup finds the exact installed engine, checks the package against the app
source and starts the sender to verify the connection. The prebuilt route uses
the installed Unreal editor without a Visual Studio or Windows SDK build.
If the engine is installed outside the default Epic Games folder, add
`--engine` with the installation root. Extract the separate RGB MoCap Connect
receiver ZIP into the receiving project's `Plugins` folder.
If setup reports an engine mismatch, check the engine patch and the
`BuildId` in `Engine/Binaries/Win64/UnrealEditor.modules`. If it reports a
sender source mismatch after an app update, use a sender ZIP for the updated
source, or [build the sender locally](#build-the-sender-locally).

### Build the sender locally

1. Complete the normal RGB MoCap `install.cmd` setup first; this creates
   `runtime\python.exe`. Install Unreal Engine **5.6 or later** and, in Visual
   Studio Installer, add Visual Studio 2022 **Desktop development with C++**,
   including the MSVC C++ toolchain and a Windows SDK. An Unreal installation
   alone does not establish that these compiler components are installed.
   The sender and receiver have been built on stock UE **5.6.1**, **5.7.4**
   and **5.8.4**. Use a receiver ZIP built for your engine's exact BuildId
   or build the receiver source with your project's engine. See the plugin's
   [Build and verification](../unreal_plugin/RGBMoCapMetaHuman/README.md#build-and-verification).
2. Close RGB MoCap. In File Explorer, open its app folder containing
   `RGBMoCap.cmd`, `install.cmd` and `Setup_UnrealStream.cmd`, then double-click
   **Setup_UnrealStream.cmd**. Keep the command window open until setup finishes.
   Alternatively, open PowerShell in that folder and run:

   ```powershell
   .\Setup_UnrealStream.cmd
   ```

   Automatic discovery checks `Program Files\Epic Games\UE_5.*`. For a custom
   installation or several installed engines, explicitly select the engine used
   by the receiving project:

   ```powershell
   .\Setup_UnrealStream.cmd --engine "C:\Program Files\Epic Games\UE_5.6"
   ```

   Pass the Unreal installation root containing the `Engine` folder, not an
   editor executable or the receiving project's folder. In Command Prompt, use
   the same command without the leading `.\`.

3. Setup builds RGB MoCap's small provider module against that installation,
   starts it to verify the app/provider handshake, and records a source/module/
   engine receipt under `runtime_unreal_stream`. It downloads no models or
   Unreal binaries. The installed Unreal Engine must remain available.
   Each engine build has a separate sender build folder. The last successful
   setup selects the active sender. Temporary compiler intermediates are removed
   after success; developers can retain them with `--keep-build-cache`.
4. Successful setup ends with **Live Link support is ready. In Stream choose
   Unreal Live Link, then check people.** Reopen RGB MoCap and choose
   **Stream > Target > Unreal Live Link**. This setup builds the app sender;
   installing the MetaHuman Connect ZIP in the receiving project is separate.
5. If setup reports **Live Link setup failed**, do not treat the final
   press-any-key prompt as success. Read the preceding error: run `install.cmd`
   if the Python runtime is missing, select `--engine` if discovery failed, or
   repair the Visual Studio C++ workload/Windows SDK if compilation failed.
   If UnrealBuildTool reports action paths over 260 characters, use a shorter
   app folder such as `C:\RGBMoCap`, or use the matching prebuilt bundle.
   Retry after correcting the cause. Inspect/redact logs before sharing them;
   preserve projects, models and existing setup receipts.
6. If the provider source or Unreal installation changes, rerun setup. A failed
   build/startup restores the previous receipt; changed source still needs a
   successful setup. Ordinary motion playback does not require rebuilding.

The current app and plugin use source-qualified subject evaluation, matching
rig/reference IDs and body/face frame revisions. Update them together and rerun
Setup_UnrealStream.cmd after a source change. The local bridge build is frozen
against source edits and verified before its receipt is accepted.

This support is optional and separate from the inference runtimes.
`install.cmd` and `update.cmd` leave it to `Setup_UnrealStream.cmd`. Epic's
terms apply to the separately installed Unreal Engine.

## Send motion

1. Open a project with prepared offline bone motion.
2. In **Stream > Target**, choose **Unreal Live Link** and check the people to
   send. These checkboxes are independent of People visibility.
3. In the receiving Unreal project, enable the built-in **Live Link** and
   **UDP Messaging** plugins, restart if requested, then open
   **Window > Virtual Production > Live Link**.
4. Choose **Add Source > Message Bus Source > RGB MoCap**.
   The app's status changes from starting/waiting to connected only after the
   provider reports a real Live Link connection.
5. Play, pause or scrub the RGB MoCap timeline. For a single pass, turn off
   **Loop** next to the playback buttons. Playback stops at the last frame of
   the trimmed clip and Live Link keeps sending the pose for each checked
   person with motion at that frame. Pausing also holds the current pose.
   Unchecking a person or moving to a frame without their motion removes
   their subjects. Returning to valid motion declares them again.

**None**, switching to Blender or closing the app stops the owned
provider process. **Retry connection** restarts it after a startup/connection
error. After restarting the provider, wait for it to start, then recreate the Message
Bus Source if Unreal does not reconnect. Logs are under `cache/unreal_stream`; check the app Console
for the exact log path. The Blender port controls apply only to Blender.

Use one active RGB MoCap sender for the default-name workflow. Multiple app
instances advertise the same provider name, so a name-based preset can select
the wrong instance. Set unused app Stream targets to **None** before connecting.

This version uses local IPv4 loopback UDP messaging with automatic port selection,
plus multicast discovery. For this same-computer workflow, the receiving
project's UDP Messaging **Unicast Endpoint** can be `127.0.0.1:0` and its
**Multicast Endpoint** `230.0.0.1:6666`. An explicitly selected different
network adapter, blocked UDP or a firewall can prevent discovery. Remote-machine,
packaged-game and Shipping builds were not verified.

## Export the matching source rig FBX

This is the source-avatar workflow used with Unreal's standard IK retargeting
tools. RGB MoCap supplies the source pose and a matching reference; the receiving
project handles its own target character and retarget settings.

1. Prepare motion, open **Stream**, choose **Unreal Live Link**, and highlight
   the desired person in the list. Seek inside the ready take you intend to use.
   The person does not have to be checked or connected to export its reference.
   Clicking a checkbox enables streaming; click the person's name to choose the
   reference export. If only one person is listed, export selects them automatically.
   Missing selection or ready motion is explained in a popup before file selection.
2. Click **Export Reference FBX...** and choose an output file outside the
   source project data. The app uses its already installed Blender runtime in a
   CPU worker. The Live Link provider itself is not required for this export.
3. The FBX contains the exact stream reference skeleton and a small procedural,
   rigidly skinned bone-display mesh. Every bone has weighted vertices so Unreal
   can import a complete Skeletal Mesh. It contains no human surface or motion.
   No A/T-pose, symmetry, origin-placement or ordinary motion-export setting is
   applied to this reference.
4. The default filename includes the person Subject and the short rig ID.
   Keep one reference per different rig/rest. Different people, shapes, models
   or takes can have different references. Export again if the rig changes.

SMPL-based prepared streams use 52 bones. GEM-X preserves SOMA Native77,
including its original terminal joints; the exporter adds no extra end bones.
Do not reuse a 52-bone source mesh for Native77. The **_Reference** Subject and
FBX describe the same rest skeleton; stock Live Link does not construct a mesh
or automatically configure a retargeter from that Subject.

### Import and connect with standard Unreal tools

1. Import the FBX as a **Skeletal Mesh**, with **Skeleton = None** to create a
   separate source Skeleton. Do not choose the MetaHuman skeleton. Import mesh/
   skin weights, leave animation and **Create Physics Asset** off, and use import
   scale **1.0**. The file is
   already Z-up centimetres; do not add a 100x scale or force a new front axis.
   Check that Hips is the root and that 52 or 77 bones were retained.
2. Right-click the source Skeletal Mesh and create an **Animation Blueprint**.
   Connect **Live Link Pose > Output Pose** and select the matching person
   Subject without **_Reference**. First confirm this source rig follows the
   app's pose, root position and fingers before configuring the target.
3. Create an **IK Rig** for the source mesh. Choose **Hips** as Retarget Pelvis,
   and define spine, neck, clavicle, arm, leg, toe and finger chains.
   Create/select the target character's IK Rig and an **IK Retargeter**.
   Source chain hints are below; mirror the left side for the right.

   | Chain | SMPL 52 | SOMA Native77 |
   | --- | --- | --- |
   | Spine | Spine > Spine2 | Spine1 > Chest |
   | Neck/head | Neck > Head | Neck1 > Head |
   | Clavicle | LeftShoulder | LeftShoulder |
   | Arm | LeftArm > LeftHand | LeftArm > LeftHand |
   | Leg | LeftUpLeg > LeftFoot | LeftLeg > LeftFoot, via LeftShin |
   | Toe | LeftToeBase | LeftToeBase |

   Native77 has extra finger joints and terminal bones. Inspect the actual
   hierarchy instead of copying the 52-bone finger mapping unchanged.
4. Align the source and target facing, then match their retarget poses before
   testing a real pose. The reference is the stream's bind, which is not
   necessarily the same as the target A-pose. Configure pelvis/root translation
   separately. A source Hips bone represents the pelvis, not a floor-level root.
   For the previously tested MetaHuman, Root Motion used
   **Generate From Target Pelvis + Snap To Ground** after Pelvis Motion/FK.
   Retarget options and quality still depend on the chosen target.
5. In a copy of the character Blueprint, add the source Skeletal Mesh component
   and assign its Live Link Animation Blueprint. Make the target Body a child
   of that source component. The target Animation Blueprint uses
   **Retarget Pose From Mesh > Output Pose**, selecting the IK Retargeter and
   the parent skeletal mesh source (called **Use Attached Parent** in some
   engine versions).
6. Validate in **Play/Simulate** first. For a hidden source mesh, use
   **Always Tick Pose and Refresh Bones** so hiding it does not stop evaluation.
   Preserve the MetaHuman's Body-to-Face/clothing relationships and its normal
   post-process/corrective configuration. Editor-only preview updating may also
   need the built-in **Live Link Skeletal Animation** component.

Receiving a green Subject verifies the transport, not the target retarget or
cloth simulation. Test the source rig, target Body and clothing in that order.
Unreal may reorder bone indices during FBX import; Live Link matches by bone
name. The named parent hierarchy and local rest transforms must agree.

References: [Epic's runtime IK retargeting](https://dev.epicgames.com/documentation/en-us/unreal-engine/runtime-ik-retargeting-in-unreal-engine),
[OptiTrack's source-avatar/MetaHuman workflow](https://docs.optitrack.com/v3.2/plugins/optitrack-unreal-engine-plugin/unreal-engine-optitrack-live-link-plugin/quick-start-guide-real-time-retargeting-in-unreal-engine-with-live-link-content).

## Subjects and retargeting

Each person has a readable `RGBMoCap_<person name>_<8-character ID>` animation
subject, for example `RGBMoCap_Person_1_a29d53a7`. The short identity keeps
duplicate person names distinct and remains the same across seeking, take
changes and reconnecting. Renaming the person changes the subject name.
After updating from the earlier full-hash names, select the new subject once
in the source Animation Blueprint. The native bridge does not need rebuilding.
Hover over the person in the app Stream list to see that exact subject name.
A companion `..._Reference` animation subject carries the parent-local rest
pose and `BoneLength_<bone name>` curves, including terminal bones. Motion
contains `RGBMocapFrame` and source-frame/person/rig metadata. Units are
centimetres, with Z up, parent-local translations and quaternion rotations.
The reference subject is refreshed at 10 Hz; the current pose at about 30 Hz.
No mesh, face-expression stream or Blender Bake command is sent through this target.

Receiving a subject does not automatically create a skeletal mesh, Animation
Blueprint or MetaHuman retargeter. Use a source rig whose hierarchy and rest
transforms match the reference subject. In the verified test:

- A hidden source skeletal component uses an Animation Blueprint with
  **Live Link Pose**, selecting the person's motion subject.
- The MetaHuman Body is a child of that source component, and its Animation
  Blueprint uses **Retarget Pose From Mesh**, with the parent skeletal mesh
  component and the configured IK Retargeter.
- Source and target IK Rigs define matching body/finger chains. Align their
  reference facing before editing the retarget pose. A pelvis root is not a
  ground root; in the tested MetaHuman retargeter, **Pelvis Motion > FK Chains >
  Root Motion**, with **Generate From Target Pelvis + Snap To Ground**, keeps
  the separate target root on the ground.

The stream uses the verified pilot's fixed facing convention: source
`(X,Y,Z) -> (-Z,X,Y)`, then a **-90 degree world yaw applied equally to the
rest and animated skeleton**. This is a coordinate convention, not a per-frame
pose correction. Different imported source rigs can require a matching basis/
retarget-pose adjustment. The app does not change recorded motion or existing
export settings. Terrain contact and arbitrary target-rig retarget quality
remain separate from successful packet reception.

## Implementation and measured overhead

`mocap/stream_ui.py` shares existing selected-result/playhead sampling with
Blender. `mocap/unreal_stream.py` converts on a background thread and retains
only the latest snapshot. A bounded, token-authenticated localhost TCP channel
feeds an app-owned Unreal commandlet; the official `ILiveLinkProvider`
publishes through the stock Message Bus protocol. The optional MetaHuman Connect plugin provides direct targets; the stock manual workflow also remains available.
Limits are 128 people, 256 bones per person and 4 MiB per local snapshot.
Only people with a prepared stream rig are listed.

On the development Windows PC with UE 5.7.4, a warm provider start took **2.77 s**,
provider peak working RAM was **859 MiB**, and steady CPU averaged **0.77% of one
CPU core** over six seconds. Converting two 52-bone people took **0.64 ms median /
0.93 ms p95** over 100 calls, excluding GUI, network and receiving Unreal work.
The commandlet uses NullRHI; no new AI model or GPU inference is started.
The retained local module/project occupied about 59 MiB after compiler-cache
cleanup, excluding the existing Unreal installation. These are observations,
not minimum requirements or a guarantee on other PCs.

See [validation](VALIDATION.md#unreal-live-link-body-streaming-2026-11-19)
and [Epic's Live Link provider documentation](https://dev.epicgames.com/documentation/en-us/unreal-engine/live-link-plugin-development-in-unreal-engine).

## Optional MetaHuman face streaming

The app also sends saved **Track Face** ARKit-style coefficients for each checked
person as `<motion Subject>_Face`. This is an Animation-role curve Subject,
separate from the body and `_Reference` Subjects. It follows the current take,
source time, playback and seeking. Run Track Face first; selecting a stream target
does not run face inference. The capture checkbox controls new capture, so an
existing valid face archive can still stream when that checkbox is off.

In **RGB MoCap Connect**, open **Tools > RGB MoCap
Connect**, select a source and MetaHuman actor, and enable **Face** on that
source row. The checkbox is off by default and applies to the MetaHumans bound
to that row.
Each person can use a different MetaHuman and Face setting. `_Face` Subjects do
not appear as extra people in this panel. This plugin is distributed as the separate ZIP linked above.

The plugin maps the 52 coefficients using the MetaHuman's standard additive
ARKit mapping pose asset and keeps its existing facial postprocess/RigLogic.
Head and neck motion remain driven by the body. The mapping normally resolves
`PA_MetaHuman_ARKit_Mapping` in the assembled MetaHuman content; older content
can use `mh_arkit_mapping_pose`. If absent, the row explains the problem; set
**Face Mapping Pose** on the actor's RGB MoCap component to the matching additive
asset. MetaHuman assets are not included in the plugin ZIP.

A missed face detection holds that person's last successful expression. Before
the first detection, outside a valid face archive, or when face data are absent,
body streaming continues without a facial sample. A missing/stale live stream
holds the last received expression; an explicit no-face sample returns the
plugin's expression controls to neutral. Disabling Face or disconnecting restores
the previous face animation class. Existing Blueprint playback state is not
preserved across reinitialization. Face Overlay visibility does not control
streaming. MediaPipe does not measure `tongueOut`; it remains zero. This is saved
timeline playback, not a live camera face-capture mode.

After updating this sender, close/restart RGB MoCap and rerun
**Setup_UnrealStream.cmd** if requested. Setup uses source-versioned bridge
folders and verifies startup before replacing the receipt. A failed verification
restores the previous receipt; a source mismatch still requires successful setup.
Restart Unreal after replacing the receiver plugin. Body-only Live Link and the
existing TXT/Live Link Face CSV exports keep their existing behavior.


MetaHuman Connect preserves simultaneous body/face editor playback: the
MetaHuman Face initialization callback can disable editor updates on Body and
its followers. Reapply the connection's owned preview tick settings after face
initialization/restoration; retain the original recovery baseline. Validate
actual body transforms across seeks and continuous Face-enabled playback,
plus facial deformation, rather than relying on Streaming status alone.
Receiver 0.5.0 uses the same sender protocol as 0.4.0. If upgrading from
an older sender, update RGB MoCap and rerun `Setup_UnrealStream.cmd`.
Close Unreal before replacing the project plugin folder.


### Smoothed face playback

Current app playback always sends One Euro-smoothed ARKit coefficients and
displays smoothed face-overlay landmarks. Both use one asynchronous, verified
archive cache, computed in source-frame/FPS order separately for each person.
Pauses, loops, reverse seeks and playback-rate changes reuse the same results.
Invalid detections hold the last valid filtered sample; a detection gap longer
than 0.25 seconds resets the filter on reacquisition. The first detected sample
seeds the filter directly. Preparation never sends raw samples as a fallback;
body streaming continues while the face cache is being prepared.

Capture archives are preserved. The Export dialog's Smooth Face checkbox still
controls only exported TXT/CSV: on uses the same coefficient filter as streaming,
off exports unsmoothed capture values. Saved neutral calibration applies in
both modes. After updating the app, restart it to use the shared face cache.


### Neutral face calibration

In RGB MoCap, select a person in People, seek to a successfully tracked neutral
frame and click **Set Neutral** beside Track Face. The corrected expression
coefficients are sent automatically for that person; no Unreal/plugin setting
or rebuild is needed. **Reset** removes that person's baseline. The setting is
saved with the capture, survives reopening/Save Project As, and applies to its
takes. Preview landmarks and TXT/CSV exports use the same saved reference.
With playback smoothing enabled, calibration is derived from the selected
smoothed frame so that frame sends exactly zero expression. Raw capture data
are preserved. Choose an eyes-open, relaxed-mouth frame; held/missing detections
are not eligible as neutral frames.


## Security and publication

Live Link UDP Messaging is intended for local/trusted networks. It does not
provide application-level encryption or authentication against hostile network
participants. Restrict UDP Messaging endpoints and firewall access to intended
peers. The receiver's matching source/revision checks prevent accidental mixing;
they are not a hostile-network authentication guarantee.

App person checkboxes and Target None control outgoing data. When face capture
exists, selected people include facial coefficients; the plugin Face switch
controls applying them, not publication. Transmitted fields include person
labels, short subject IDs, rest/pose bones, lengths and optional face values.
Video, mesh vertices, source paths, capture archives and calibration files are
not transmitted. The loopback launch nonce uses the owned child environment,
not command-line arguments. Local logs/projects remain private data; inspect
them before sharing. See [plugin security notes](../unreal_plugin/RGBMoCapMetaHuman/SECURITY.md).
