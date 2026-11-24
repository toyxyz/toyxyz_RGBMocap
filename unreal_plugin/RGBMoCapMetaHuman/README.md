# RGB MoCap Connect

Connect RGB MoCap motion streams to assembled MetaHumans or UE5 Manny and Quinn
actors. Each source row shows its connected actors and target types. MetaHumans
can also receive tracked expressions. The plugin retargets motion using the
receiving project's existing character meshes. The [manual Live Link
workflow](https://github.com/toyxyz/toyxyz_RGBMocap/blob/main/docs/UNREAL_LIVE_LINK.md)
remains available.

## Install

1. Download the [prebuilt bundle](https://github.com/toyxyz/toyxyz_RGBMocap/blob/main/docs/UNREAL_LIVE_LINK.md#prebuilt-downloads)
   for your exact Unreal Engine version and BuildId. Extract it to find the
   RGB MoCap Connect receiver ZIP. Use [Build and verification](#build-and-verification)
   to build for another engine build.
2. Close Unreal. Extract the `RGBMoCapMetaHuman` folder into your project's
   `Plugins` folder. Create `Plugins` if needed. Replace the entire existing
   `RGBMoCapMetaHuman` folder when upgrading.
3. Open the project and enable **RGB MoCap Connect**, **Live Link** and
   **UDP Messaging** if necessary. Restart when requested.

The bundles target Epic Games Launcher UE **5.6.1**, **5.7.4** and **5.8.4**
on Win64. The receiver ZIP contains source and authored DLLs. Use your own
Unreal project and character assets.

## Prepare RGB MoCap

Update RGB MoCap to the current `main` version. Run
`Setup_UnrealStream.cmd --prebuilt <sender.zip>` from the app folder with the
inner sender ZIP from the same bundle. The installed editor hosts the sender
without a local build. For another engine build, use the
[local build option](https://github.com/toyxyz/toyxyz_RGBMocap/blob/main/docs/UNREAL_LIVE_LINK.md#build-the-sender-locally).
Sender setup runs separately from `install.cmd` and `update.cmd`. Rerun
sender setup after an engine or sender source change.

## Connect

1. Place the MetaHuman Blueprints, mannequin Skeletal Mesh Actors or
   mannequin Character Blueprints in the level.
2. In RGB MoCap, load prepared offline motion, choose **Stream > Unreal Live
   Link** and check the people to send. None is the default target.
3. Open **Tools > RGB MoCap Connect** in Unreal. It adds or reuses the app's
   Message Bus source. Set **Target type** to **MetaHuman** or **SK Mannequin
   (Manny/Quinn)**, select a source row and one actor in the level, and click
   **Connect selected actor**.
4. Repeat to connect another actor. **Connected targets** displays each actor
   and its type beside the source. Click the target cell to select its actors.
   Changing **Target type** applies to the next connection.
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

Enable **Face** on a source row to apply that person's saved Track Face data
to the connected MetaHumans. The switch is off by default and can be set before
connecting a MetaHuman. Manny and Quinn receive body motion only. The receiver maps the
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
The selected mesh needs **Use Animation Blueprint** mode and positive uniform
scale. A standalone mannequin mesh may leave **Anim Class** empty. Manny and
Quinn use the UE5 mannequin skeleton and its standard IK guides. The solver
measures actual target rest lengths, transfers rest-relative rotations and limb
directions, scales travel and calculates leg reach. On mannequins it preserves
scaled ankle spacing and updates the hand and foot IK guides.
Ground Contact and Floor Offset on the connection component control an estimated
flat floor in component-space centimetres. This does not trace world geometry,
solve uneven terrain, add temporal sticky foot locks or remove existing source
slip. Clothing/face postprocess relationships are preserved.

## Build and verification

To build from source, copy this plugin into a C++ project's Plugins
folder and build that project's Editor target using its matching Unreal/Visual
Studio toolchain. Do not ship a stale prebuilt DLL under a different BuildId.
See [VALIDATION.md](VALIDATION.md) and [SECURITY.md](SECURITY.md).

Authored code is MIT licensed. Unreal Engine and separately obtained assets stay
under their own terms. The body solver was informed by the workflow of
[Toyxyz Blender Retargeter](https://github.com/toyxyz/toyxyz-blender-retargeter);
its Python code is not included or translated into this plugin.
