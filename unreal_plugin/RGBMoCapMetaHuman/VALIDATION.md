# Verification scope

## 0.6.1 connected-take bake

UE 5.8.4 baked an actual paused 1,675-sample take to the connected MetaHuman
and Manny skeletons. A separate 100-sample MetaHuman take used saved face
tracking. Its body and face sequences both reopened with 100 samples at
30 fps. The Face Skeleton sequence had 216 curves, and RigLogic moved the
facial bones when it played. With Face off, missing saved expressions or
a Manny target, the body asset still saved by itself.

A 33.9-second MetaHuman take also played through an actual Level Sequence.
Setting **Force Custom Mode** on the Face animation section restored eye
blink and jaw motion in the assembled character. See
[Bake and Face playback](../../docs/UNREAL_LIVE_LINK.md#bake-a-connected-take).

## 0.5.2 A-pose clavicle rest alignment

UE 5.8.4 evaluated both receivers on the same MetaHuman body reference
skeleton and 1,675 saved FlowHMR frames. At a representative lowered-arm
frame, the source clavicles sat about 8 degrees below horizontal. Version
0.5.1 placed them about 26 degrees below horizontal and moved the shoulder
joints about 4.3 cm below the A-pose in chest space. Version 0.5.2 recovered
approximately 8 degrees below horizontal and kept both joints within 0.2 cm
of their A-pose height in that frame.

A 12-degree one-sided clavicle rotation was preserved on MetaHuman and
Manny. Lowering the upper arms by 70 degrees in isolation did not move
either clavicle. Rotating the source root did not change the target's
chest-relative clavicle angles. A T-pose target produced the same transforms
before and after the change over 22 test poses.

A separate native solver test used 294 saved GEM-X frames on the same
MetaHuman asset, Manny and a T-pose target. Native77 clavicle elevations
matched the source within 0.01 degrees on all three. The SMPL replay and
controls produced identical transforms after the Native77 change.

## 0.5.1 clavicle rest calibration

The offline measurement used a 495-frame FlowHMR take with a T-pose Mixamo
target and a 2,267-frame take with an A-pose PMX target. Blender baked both.
The T-pose result matched the previous chest-relative transfer. The A-pose
target received about 3.2 degrees of reference lift. In both takes the tested
pelvis, spine and leg joint positions matched the baseline.

A UE 5.8.4 commandlet compared 248 saved poses on each stock mannequin and
a T-pose control. The geometry-derived lift was about 7.9 degrees for Manny
and 7.1 degrees for Quinn. A 12-degree one-sided shrug remained 12 degrees.
The T-pose control and tested leg channels matched the baseline.

## Prebuilt Windows bundles

Clean app and Unreal projects loaded the paired archives on Epic Games Launcher
UE 5.6.1 (BuildId 43139311), 5.7.4 (47537391) and 5.8.4 (55116800).
Sender setup completed its startup handshake. The installed editor received
the test body's animation, reference pose and face subjects over Live Link.
UnrealBuildTool did not run during sender setup or the scripted receiver check.
These checks use synthetic frames. Actor deformation tests are described below.

## 0.5.0 mannequin and mixed targets

Tested in Unreal Engine 5.7.4 / Win64 with the live app sender and saved
motion for two people. The real Slate panel connected a stock Manny
Skeletal Mesh Actor and a stock Quinn Character Blueprint. It rejected
wrong target types and displayed each actor's type beside its source.

A timeline seek changed both mannequins' hand and finger poses. The hand
and foot IK guides matched their driven joints in component space. Manny
shared a source with an assembled MetaHuman receiving tracked-face test
coefficients. The MetaHuman used the face animation class while Manny
and Quinn stayed body-only. Disconnect restored the original animation
on the mixed row while Quinn kept streaming on the other row. Undo
restored the mannequin connection and its target type.

The native solver check used 51 samples from a 52-bone recording and 22
Native77 samples on each stock mannequin mesh and two leg-length variants.
Leg lengths stayed fixed. The IK guides followed their driven joints.
Reachable samples preserved the horizontal ankle spacing at target scale.
Reverse-order evaluation reproduced the forward poses.

Saving an active connection restored the original animation class in the
saved level and resumed live preview in the editor. A separate load verified
that both mannequin target types and their enabled state were serialized.

## 0.4.0 MetaHuman release checks

Release 0.4.0 targets stock Unreal Engine 5.7.4 / Win64, BuildId 47537391.
Checks use authored diagnostics and separately installed MetaHuman content;
diagnostics, character assets, recorded motion and private results are excluded
from both this source folder and the installation ZIP.

Verified behaviors in the release audit:

- Native sender and receiver compile, and local sender startup authenticates.
- Two-person source/target panel connection, reassignment, disconnect, target
  selection, actor rename/deletion, Undo, stream stop/resume and map switching.
- Independent Face controls; actual jaw/blink bone deformation, held/no-face
  samples, resume, body/head alignment and restoration on disable/disconnect.
- Body and face simultaneously follow timeline seeks and continuous playback.
- Recovery through duplication, save suspension, Undo and Blueprint reconstruction.
- Real saved face data arrive at Unreal with the app's smoothed coefficients;
  neutral calibration sends zero at its reference frame and repeat seeking gives
  identical values. The neutral integration fixture is an independent file copy.
- Per-person calibration, held-frame rejection, capture-time mapping, reset and
  reload, immutable raw archives, TXT/CSV behavior and smoothing/export equality.
- Bounded loopback authentication rejects a wrong nonce; nonce is absent from
  launch arguments, invalid status is rejected, shutdown reaps the owned process.
- Setup rejects source changes during compilation and retains a previous receipt;
  generated build outputs do not alter source identity.

The final archive is read back for CRC, SHA-256 and per-file identity, scanned
for private paths/credential patterns in text and ASCII/UTF-16 native strings,
then extracted and loaded in a content-only project without the diagnostic
module. Public content uses a strict source/document/native-module allowlist.
The exported body rig/fingers are compared with independently retained solver
expectations. Runtime source and modules in the ZIP match the tested build.

This does not establish universal target compatibility, uneven-terrain contact,
source-motion accuracy, arbitrary custom face mappings, hostile-network security
or support for other engine builds. The flat-floor sole estimate and original
animation-instance playback-state limitations are described in README.md.
