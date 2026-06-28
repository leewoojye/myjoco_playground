# dVRK PSM/ECM Assets for MyJoCo

This folder contains a lightweight MuJoCo scene for a dVRK-style PSM teleoperation demo.

Upstream reference assets are copied from `jhu-dvrk/dvrk_model`:

- Source: https://github.com/jhu-dvrk/dvrk_model
- License: MIT, preserved in `LICENSE.upstream`
- Copied meshes: Si PSM/ECM base links and PSM/ECM instrument STL files
- Copied URDF/Xacro references: `upstream_urdf/Si`

The MyJoCo XML files are intentionally simplified for a stable RCM-constrained teleoperation portfolio demo:

- `psm.xml`: dVRK-style PSM, ECM camera body, position actuators, named tool/shaft sites
- `scene_psm_peg_needle.xml`: surgical board, peg reach targets, needle approach target, active target marker

The simplified MJCF is a research/portfolio simulator asset, not a clinically accurate dynamics model.
