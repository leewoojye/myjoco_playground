# 01. CPG-Residual Natural Walker

This experiment treats a hand-authored CPG gait as a structured prior and learns
a small residual parameter vector on top of it. The goal is not maximum forward
speed; it is a more natural-looking gait under physics constraints.

The default demo now runs on the Berkeley Humanoid model from Google DeepMind's
MuJoCo Menagerie. The old planar toy walker is still available with
`--robot toy`.

## Research Question

Can a low-dimensional residual controller improve a procedural walking rhythm in
terms of forward progress, balance, energy use, and smoothness?

## Files

- `toy_walker.xml`: demo-local MuJoCo planar biped asset.
- `assets/mujoco_menagerie/berkeley_humanoid`: official Berkeley Humanoid asset.
- `cpg_residual_core.py`: simulation loop, CPG target generation, reward metrics.
- `berkeley_cpg_core.py`: Berkeley Humanoid CPG mapping and virtual pelvis stabilizer.
- `train_cem_residual.py`: dependency-light CEM optimizer for residual parameters.
- `run_demo.py`: evaluates either the base CPG or a saved residual policy.
- `config.json`: default experiment settings.
- `REFERENCES.md`: related work.

## Run

From the repository root:

```bash
source venv/bin/activate
python sim_with_mujoco/demo/01_cpg_residual_natural_walker/run_demo.py
```

Berkeley Humanoid with trained CPG residual:

```bash
python sim_with_mujoco/demo/01_cpg_residual_natural_walker/run_demo.py --robot berkeley --policy-json sim_with_mujoco/demo/01_cpg_residual_natural_walker/berkeley_policy_residual.json
```

Berkeley Humanoid headless metric check:

```bash
python sim_with_mujoco/demo/01_cpg_residual_natural_walker/run_demo.py --robot berkeley --headless --duration 5 --policy-json sim_with_mujoco/demo/01_cpg_residual_natural_walker/berkeley_policy_residual.json
```

Planar toy walker with the learned residual:

```bash
python sim_with_mujoco/demo/01_cpg_residual_natural_walker/run_demo.py --robot toy --policy-json sim_with_mujoco/demo/01_cpg_residual_natural_walker/policy_residual.json
```

Train a small residual vector:

```bash
python sim_with_mujoco/demo/01_cpg_residual_natural_walker/train_cem_residual.py --robot berkeley
python sim_with_mujoco/demo/01_cpg_residual_natural_walker/train_cem_residual.py --robot toy
python sim_with_mujoco/demo/01_cpg_residual_natural_walker/run_demo.py --robot toy --policy-json sim_with_mujoco/demo/01_cpg_residual_natural_walker/policy_residual.json
```

## Portfolio Angle

The story is: `CPG baseline -> learned residual -> smoother/natural gait
metrics`. This is deliberately different from directly reproducing FreeMusco:
the contribution is a structured control prior plus lightweight learning inside
MyJoCo.

For Berkeley Humanoid, this is a CPG preview with a virtual pelvis stabilizer,
not a fully learned balance controller. It is intended as the starting point for
replacing the stabilizer with a residual or transition policy.
