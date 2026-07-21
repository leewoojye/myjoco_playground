# dVRK Si patient-side MJCF

This directory contains the Si tower, four SUJs, three PSMs, and one ECM converted to a single MuJoCo model.

Source files under `dvrk_model/` come from [`jhu-dvrk/dvrk_model`](https://github.com/jhu-dvrk/dvrk_model) commit `1744e52cedc33a750016f1a5a919e8129140bc21`. The upstream CISST license is preserved in `dvrk_model/LICENSE`.

The converter expands the current lowercase component Xacros separately, prefixes each arm's generic joint names, merges the arms at the SUJ mounting points, and adds MuJoCo position actuators and mimic equalities. The stale upstream `patient_cart.urdf.xacro` is not used.

```bash
venv/bin/python assets/robots/dvrk_si_patient_side/convert_dvrk_si_to_mjcf.py
```

Generated files:

- `dvrk_si_patient_side.urdf`: expanded and merged URDF
- `dvrk_si_patient_side.xml`: converted MJCF used by the scene

Imported mesh collision geoms are retained but disabled while validating the full assembly. Re-enable only the task-relevant tool collision geoms before grasp/contact experiments.
