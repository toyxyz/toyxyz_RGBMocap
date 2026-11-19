# Stream body bones to Unreal Live Link

This optional Windows Git/CMD feature sends the selected offline people's bones
through Unreal's built-in **Message Bus Source**. It follows playback and
scrubbing; it does not run live video inference. The default Stream target
remains **None**. Blender streaming keeps its existing port and add-on.

## Download the sample project

[Download the Unreal Live Link sample ZIP](https://github.com/toyxyz/toyxyz_RGBMocap/releases/download/unreal-live-link-sample-v1/RGBMoCap_Unreal_LiveLink_Sample_UE5.7_v1.zip)
([SHA-256](https://github.com/toyxyz/toyxyz_RGBMocap/releases/download/unreal-live-link-sample-v1/RGBMoCap_Unreal_LiveLink_Sample_UE5.7_v1.zip.sha256)). Extract into a short writable path and open
`RGBMoCap_LiveLink_Sample.uproject` in UE **5.7.4**. The project contains its
MetaHuman assembly, matching 52-bone source, stock Live Link Animation Blueprints,
20-chain IK Rigs/Retargeter, and comparison map. No receiver plugin or C++ project
module is required. The app sender still needs the separate setup below.

In `Assets/ABP_RGBMoCap_LiveSource`, choose your running app's motion Subject and
Compile/Save. The distributed default is **None**; do not select `_Reference`.
The sample contains no recorded motion or video. Its rig is one specific SMPL
52-bone reference: verify the rest/rig identity, not just the bone count. Export
and configure a matching reference for other shapes/takes/models; Native77 needs
its own setup. The ZIP includes an English README, validation records, per-file
hash manifest and separate Epic asset notices. It omits engine binaries, build
outputs, caches, model weights and local project/user state.

## Prepare once

1. Install Unreal Engine **5.7 or later**, Visual Studio 2022 **Desktop
   development with C++**, and a Windows SDK. The tested engine is **5.7.4**;
   later versions are not independently verified.
2. Close RGB MoCap and run **Setup_UnrealStream.cmd** in the app folder.
   For a custom Unreal installation, run:

   ```cmd
   Setup_UnrealStream.cmd --engine "D:\Epic Games\UE_5.7"
   ```

3. Setup builds RGB MoCap's small provider module against that installation,
   starts it to verify the app/provider handshake, and records a source/module/
   engine receipt under `runtime_unreal_stream`. It downloads no models or
   Unreal binaries. The installed Unreal Engine must remain available.
   Temporary compiler intermediates are removed after success; developers can
   retain them with `--keep-build-cache`.
4. If the provider source or Unreal installation changes, rerun setup. A failed
   build/startup does not leave a successful readiness receipt.

This support is optional and separate from the inference runtimes.
`install.cmd` and `update.cmd` do not compile it. Epic's terms apply to the
separately installed Unreal Engine; RGB MoCap distributes its own bridge source.

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
5. Play, pause or scrub the RGB MoCap timeline. Pausing holds the current pose
   while continuing the Live Link heartbeat. An unchecked person, or a person
   without motion at the current frame, has its subjects removed; returning to
   valid motion declares them again.

**None**, switching to Blender, entering PEAR or closing the app stops the owned
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
publishes through the stock Message Bus protocol. There is no new receiver plugin.
Limits are 128 people, 256 bones per person and 4 MiB per local snapshot.
Only people with a prepared stream rig are listed; PEAR is excluded.

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
