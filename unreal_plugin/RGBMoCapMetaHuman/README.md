# RGB MoCap MetaHuman Connect

Directly connect selected RGB MoCap offline motion streams to assembled
MetaHuman actors. Each source row shows its connected targets and optional Face
switch. No source-mesh import, reference FBX or IK Retargeter is required for
this plugin workflow. The app's existing manual Live Link workflow remains
available separately.

## Install

1. [Download the UE 5.7.4 / Win64 plugin ZIP](https://github.com/toyxyz/toyxyz_RGBMocap/releases/download/unreal-metahuman-connect-v0.4.0/RGBMoCapMetaHuman_UE5.7.4_Win64_v0.4.0.zip)
   and [SHA-256](https://github.com/toyxyz/toyxyz_RGBMocap/releases/download/unreal-metahuman-connect-v0.4.0/RGBMoCapMetaHuman_UE5.7.4_Win64_v0.4.0.zip.sha256).
2. Close Unreal. Extract the `RGBMoCapMetaHuman` folder into your project's
   `Plugins` folder. Create `Plugins` if needed. Replace an older plugin folder
   as a complete folder; keep your project assets separately.
3. Open the project and enable **RGB MoCap MetaHuman Connect**, **Live Link** and
   **UDP Messaging** if necessary; restart when requested.

The prebuilt modules were compiled with stock Unreal Engine **5.7.4**, Win64,
BuildId **47537391**. Other engine builds may need compilation from source.
The descriptor uses compatible EngineVersion 5.7.0. The ZIP includes source and
authored DLLs, not Unreal Engine, characters, Epic content or a sample project.

## Prepare RGB MoCap

Install Unreal Engine 5.7.4 and Visual Studio 2022 Desktop development with C++
plus a Windows SDK. Close RGB MoCap and run `Setup_UnrealStream.cmd` in its app
folder. This compiles and verifies the optional sender against your local
engine; install.cmd/update.cmd do not build it. Rerun setup after sender source
or engine changes. The current sender and receiver must be updated together for
the face revision checks. No model is downloaded by sender setup.

## Connect

1. Place your assembled MetaHuman Blueprint actors in the level.
2. In RGB MoCap, load prepared offline motion, choose **Stream > Unreal Live
   Link** and check the people to send. None remains the default target.
3. Open **Tools > RGB MoCap Connect** in Unreal. It adds/reuses the app's stock
   Message Bus source automatically. Select a source row, then select exactly
   one MetaHuman actor in the level and click **Connect selected actor**.
4. Repeat for another person/actor. **Connected MetaHuman** displays each target
   beside its source. Click a target cell to select its actors in the level.
5. Use a Realtime editor viewport or Play/Simulate and play/scrub the app.
   Connection editing is disabled during Play/Simulate.

Reassigning an actor reuses its connection component. **Disconnect source**
restores all actors on that source row; other rows remain connected. Connections
support editor Undo/Redo. Stopped sources retain their target rows so resuming
is understandable. Source selection is explicit, never silently moved to
another person. **Reconnect Live Link** recreates the app-named Message Bus
sources if reception stays stale after an app restart. Use one active app sender
with this default provider-name workflow.

## Face

Enable **Face** on a source row to apply that person's saved Track Face data.
Face is off by default; it can be set before connecting. The receiver maps the
ARKit-style coefficients using the target's additive MetaHuman ARKit mapping
pose asset, preserving its facial postprocess/RigLogic. Body drives head/neck.
Do not enable the MetaHuman actor's global UseLiveLink switch for this workflow.

The app uses One Euro-smoothed coefficients and landmark previews. **Set Neutral**
beside Track Face sets a per-person reference at the current successful capture
frame; **Reset** clears it. Settings are saved with the app project, and the
same reference applies to face export. Original capture data are preserved.
Missing detections hold the last expression; explicit no-face input clears to
neutral, and stale transport holds the last safe expression. Body keeps working
without face data. MediaPipe does not estimate tongueOut.

Standard assembled MetaHuman mapping assets are found automatically. If missing,
the status explains the problem; set **Face Mapping Pose** on the actor's RGB
MoCap component to the compatible additive ARKit pose asset. No mapping or
character assets are distributed here. Disabling Face or disconnecting restores
the previous face class. Reinitialization does not preserve prior montage or
state-machine playback state.

## Body and ground contact

Supported source skeletons are RGB MoCap SMPL-based 52-bone and SOMA Native77.
The target needs standard MetaHuman Body pelvis/thigh/calf/foot/ball names,
Animation Blueprint mode and positive uniform scale. The solver measures actual
target rest lengths. It transfers rest-relative rotations and anatomical limb
directions, scales travel by body proportions and uses analytic leg reach.
Ground Contact and Floor Offset on the connection component control an estimated
flat floor in component-space centimetres. This does not trace world geometry,
solve uneven terrain, add temporal sticky foot locks or remove existing source
slip. Clothing/face postprocess relationships are preserved.

## Build and verification

For another engine build, copy this source plugin into a C++ project's Plugins
folder and build that project's Editor target using its matching Unreal/Visual
Studio toolchain. Do not ship a stale prebuilt DLL under a different BuildId.
See [VALIDATION.md](VALIDATION.md) and [SECURITY.md](SECURITY.md).

Authored code is MIT licensed. Unreal Engine and separately obtained assets stay
under their own terms. The body solver was informed by the workflow of
[Toyxyz Blender Retargeter](https://github.com/toyxyz/toyxyz-blender-retargeter);
its Python code is not included or translated into this plugin.
