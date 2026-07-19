# SurRoL dVRK MJCF assets

This directory contains PSM and ECM assets converted from the SurRoL
`SR-VPPV` branch at commit `10f7c8fbb8f6585cb41e09a2ff9231b126266765`.

Source URDF and mesh directories:

- `psm/psm.urdf`, `psm/meshes/`
- `ecm/ecm.urdf`, `ecm/meshes/`

Generated MJCF files:

- `psm/psm.xml`
- `ecm/ecm.xml`

The conversion follows RussellWiz/MUJOCO's `Src/trans_urdf2xml.py` method:
it adds MuJoCo compiler settings to a temporary URDF, loads that URDF with
MuJoCo, and writes canonical MJCF with
`mujoco.mj_saveLastXML`.

RussellWiz's `<default><mesh inertia="shell"/></default>` insertion is omitted.
MuJoCo's documented URDF extension only accepts `compiler`, `option`, and
`size`, and MuJoCo 3.6 silently ignores that MJCF-only default block.

`strippath="false"` is explicitly set so MuJoCo 3.6 keeps SurRoL's nested
`meshes/visual` and `meshes/collision` paths.

The RussellWiz aliases replace three PSM mesh references that are missing from
the SurRoL branch. Their original millimeter scale is removed because the alias
targets are already stored in meters.

Run the conversion again from the repository root with:

```bash
venv/bin/python assets/robots/dvrk/surrol_mjcf/convert_urdf_to_mjcf.py
```

Each generated MJCF includes:

- one equality constraint for every URDF mimic relation;
- one position actuator for every non-mimic joint;
- `ctrl_<joint>` actuator names whose controls are desired joint positions;
- RussellWiz gains and force limits: revolute `kp=60`, insertion `kp=400`, and
  gripper `kp=15`;
- RussellWiz control-stability settings: `damping=2`, `armature=0.02`,
  `frictionloss=0.01`, and the `implicitfast` integrator at a 2 ms timestep.

The resulting counts are seven actuators and five equalities for PSM, and four
actuators and four equalities for ECM.

Equality constraints are enforced by MuJoCo during `mj_step`. They do not
project manually assigned, inconsistent `qpos` values during `mj_forward`
alone. To replace manual mimic synchronization, write desired driver positions
to `data.ctrl` and advance the simulation with `mj_step`.

Upstream sources:

- SurRoL: https://github.com/med-air/SurRoL/tree/SR-VPPV/Data_driven_scene_simulation/surrol/assets
- RussellWiz/MUJOCO: https://github.com/RussellWiz/MUJOCO

The copied SurRoL files remain covered by `LICENSE.surrol`.
