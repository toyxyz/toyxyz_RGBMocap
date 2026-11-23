# Stream VMC output (experimental)

## Controls

Open the **Stream** tab and choose **Target > VMC**. Enter the receiving IP/UDP
port (default `127.0.0.1:39539`) and click **Start VMC Output**. Click **Stop VMC
Output** to stop. IP/port are editable when output is stopped.

For ExMR, FlowHMR, GVHMR, MAMMA and PromptHMR, prepare the chosen motion preview
and select one person in the Stream list. VMC follows the current timeline,
including pause/scrub, selected result and scene placement. GEM-X uses its prepared
Native77 motion with a dedicated mapping. Merely choosing an offline model does
not perform inference, load missing motion or make that model real time.

PEAR uses the same Stream controls; Start Live Tracking remains separate. The
People tab is disabled while PEAR is active; Stream stays visible. There is no
second VMC panel in PEAR Settings. PEAR uses accepted live preview packets after
filtering/position lock/foot lock. Entering PEAR stops the previous stream; select
VMC and Start to begin the new source. Leaving PEAR or stopping live tracking
stops output. Blender streaming stays restricted to prepared offline motion.

VMC has no per-pose actor ID, so this version selects one actor per destination.
Checking a different actor replaces the previous selection. A missing take or
person sends unavailable status, not the last actor's animation. Offline pause
keeps refreshing the current pose four times per second; playback
updates sample the current playhead, without an inference or transmission queue.
The send thread retains only the latest small pose. The adapter cache retains
one rig, no mesh or timeline cache. Settings are session-only.

## Skeleton contract

52 body/finger names, parent-local metre positions and XYZW quaternions in Unity
coordinates. Root is identity; Hips carries translation once. Prepared motion
preserves the original source bone axes and positions, including any source
bind-pose convention. Receivers need bind/retarget correction where their avatar
differs. Finger motion is present only to the extent the source model estimated
it; e.g. PromptHMR's neutral fingers do not become tracked fingers.

SOMA uses LeftLeg/LeftShin rather than Mixamo LeftUpLeg/LeftLeg, a three-bone
spine and extra neck/finger/end joints. The adapter remaps names and collapses
unmapped helpers through world transforms, preserving the mapped pivots.
Face/eye/jaw/terminal helpers are not sent. No source matrices are overwritten.

Each OSC bundle is at most 1400 bytes; a pose spans multiple datagrams. Delivery
is unacknowledged and partial UDP loss can leave a partial pose. `/VMC/Ext/OK 0`
is sent on loss/stop; receivers that ignore it may retain their last pose. This
feature does not guarantee accurate avatar retargeting or end-to-end latency.

## Verification

- Original six PEAR/OSC/UDP tests remain available in `tests/test_pear_vmc.py`.
- Five `tests/test_vmc_rig.py` tests cover common prepared skeletons, SOMA naming
  and collapsed neck, scene transforms, exact FK reconstruction and invalid data.
- A developer Qt harness checked actual controls and localhost UDP with synthetic
  motion: source/model changes, playhead changes, exclusive actor selection, take
  gaps, PEAR routing, Start/Stop and Blender listener configuration. It is not a
  full application, physical-camera or Blender bake test.
- Existing local cached FlowHMR, ExMR and GEM-X results were sampled at first,
  middle and final frames. All mapped to 52 bones; FK pivot errors were below
  0.000001 m before OSC float packing. This is adapter consistency, not tracking
  accuracy. No cache data is included in this patch.
- Actual newly integrated application playback, physical camera and receiver
  rendering still need user verification. GVHMR/MAMMA/PromptHMR use the shared
  prepared-rig format; no cached result or new inference was tested for them.

Original PEAR-only live output was previously confirmed by the user in Operator
V6. Protocol: https://protocol.vmc.info/english.html
