# dVRK PSM Assets for MyJoCo

This folder contains the SurRoL-derived PSM assets used by the MyJoCo needle-reach teleoperation demo.

Current runtime files:

- `psm_surrol.xml`: SurRoL `psm_RL.urdf`-derived PSM chain adapted to the same MyJoCo joint/site/actuator names
- `scene_psm_surrol_needle_reach.xml`: table, tray, needle proxy, target marker, and camera setup
- `surrol_psm`: SurRoL source URDF, visual meshes, and license

Upstream reference:

- Source: https://github.com/med-air/SurRoL
- License: MIT, preserved in `surrol_psm/LICENSE.surrol`
- Source URDF: `surrol_psm/psm_RL.urdf`

This MJCF is a research simulator asset, not a clinically accurate dynamics model.
