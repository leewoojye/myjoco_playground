# 02. Natural Motion Transition Generator

This experiment focuses on motion transitions rather than steady-state walking.
It generates smooth command-conditioned transitions such as:

- stand -> walk
- walk -> stop
- walk -> recover
- recover -> walk

## Research Question

Can MyJoCo generate natural-looking motion transitions by blending structured
gait primitives with smooth temporal envelopes and physics feedback?

## Files

- `transition_core.py`: transition schedules, smooth blending, rollout loop.
- `run_transition_sequence.py`: runs a hand-designed transition program.
- `generate_transition_dataset.py`: saves transition rollouts as `.npz`.
- `config.json`: default timings and perturbation settings.
- `REFERENCES.md`: related work.

This experiment reuses the demo-local walker asset in
`../01_cpg_residual_natural_walker/toy_walker.xml`. It does not modify files
outside `sim_with_mujoco/demo`.

## Run

```bash
source venv/bin/activate
python sim_with_mujoco/demo/02_natural_motion_transition_generator/run_transition_sequence.py
```

Generate a small dataset for later model fitting:

```bash
python sim_with_mujoco/demo/02_natural_motion_transition_generator/generate_transition_dataset.py --episodes 32
```

## Portfolio Angle

The demo story is: natural motion is not just a gait cycle. It is also the
quality of starting, stopping, changing speed, and recovering without abrupt
joint snaps.
