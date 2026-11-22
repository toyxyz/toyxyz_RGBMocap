# Experimental PEAR VMC output

Select PEAR, set a destination IP and UDP port in **VMC Output (Experimental)**,
and click **Start VMC Output**. Start Live Tracking separately. The output button
changes to **Stop VMC Output** while enabled. Both start orders are supported.
The default destination is `127.0.0.1:39539`; use the receiver's LAN IP for a
different computer. IP and port can be edited while output is stopped.

The sender emits 52 body/finger bone transforms over OSC/UDP using Unity
HumanBodyBones names, metre positions and XYZW quaternions. Root is identity;
Hips carries translation once. Bone transforms are parent-local and preserve the
PEAR source rig. Receivers with a different avatar bind pose/axes need retargeting;
VMC packet compatibility does not guarantee correct motion on every VRM avatar.
Jaw, eyes, blendshapes and meshes are not transmitted.

Accepted preview poses retain filtering, World Position and live foot correction.
The sender uses a bounded latest-only mailbox and a separate conversion/network
thread. Each OSC bundle is at most 1400 bytes; a pose spans multiple datagrams.
UDP delivery is unacknowledged and lost datagrams can produce partial poses.

Missing people, shutdown and a 0.75-second pose timeout emit `/VMC/Ext/OK 0`.
Receivers that ignore this status may keep showing their last pose. Fresh poses
resume OK=1. Stop tracking, leaving PEAR and closing the app stop output; worker
replacement for camera alignment suspends output until fresh poses arrive.
Settings are transient. No dependencies, models or runtime packages are changed.

## Validation

Run `python tests/test_pear_vmc.py` with NumPy available. The six model-free tests
cover FK round trips, aligned-basis equivalence, rotation edge cases, OSC types,
bone names, packet sizes, invalid inputs, real localhost UDP, missing/timeout
status and sender shutdown. They do not require licensed body data or a GPU.

The developer's Qt control smoke test passed against the installed GUI runtime.
The existing Operator V6 OSC codec/decoder accepted generated packets as root
plus 52 bone records in a container-stub test. The user subsequently confirmed
live PEAR motion received and displayed in Operator V6. Avatar retarget accuracy,
other receivers, physical LAN delivery and end-to-end latency are not verified.

VMC specification: https://protocol.vmc.info/english.html
