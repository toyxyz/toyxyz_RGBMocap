# Verification scope

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
