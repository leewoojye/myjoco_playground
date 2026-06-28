![Simulator preview](/assets/images/myjoco3.png)

## Simulator Overview

The current branch focuses on a dVRK PSM teleoperation demo for surgical robot simulation, while keeping the original kinematic and dynamic simulator entries for general robot-control experiments.

### dVRK PSM Needle Reach Teleoperation

The main entry file is:

```text
sim_with_mujoco/demo/dvrk_psm_teleop_demo.py
```


Run the GUI demo from the project root:

```bash
python -m sim_with_mujoco.demo.dvrk_psm_teleop_demo
```


The demo uses the SurRoL-derived PSM model by default:

```text
assets/robots/dvrk/scene_psm_surrol_needle_reach.xml
assets/robots/dvrk/psm_surrol.xml
```


The task is a reach-only surgical preview: move the PSM tool tip toward the green needle target while keeping the shaft close to the RCM constraint. Grasping, gauze retrieval, and needle pickup are intentionally disabled in the current entry.

### Controls

| Input | Behavior |
| --- | --- |
| W / S | Move the commanded tool-tip target along world z |
| A / D | Move the commanded tool-tip target along world y |
| Q / E | Move the commanded tool-tip target along world x |
| Mouse drag | Rotate, pan, or move the MuJoCo free camera |
| Mouse wheel | Zoom around the needle target focus |
| V or M | Switch to the fixed overview camera |
| F | Return to the free camera |

The upper-left overlay reports the current task, action vector, tip error, RCM error, joint-limit margin, and safety status.

## Installation

Create and activate a conda environment:

```bash
conda create -n my_robotics python=3.12
conda activate my_robotics
```


Install dependencies from the project root:

```bash
pip install -r requirements.txt
```


Core runtime dependencies include MuJoCo, GLFW, NumPy, and SciPy.

## Core Implementation

| Area | Current implementation |
| --- | --- |
| dVRK model | SurRoL psm_RL.urdf-derived PSM chain adapted to MyJoCo MJCF naming, sites, and actuators |
| Task scene | Table, tray, red needle proxy, green needle reach target, active mocap target marker |
| Input viewer | SurrolKeyboardViewer, a lightweight GLFW/MuJoCo viewer with SurRoL-style keyboard preview input |
| Teleoperation target | Keyboard input creates small task-space target increments around the current tool-tip pose |
| IK | solve_dvrk_rcm_ik converts the desired world-frame tip target into the PSM RCM frame and solves yaw, pitch, and insertion targets |
| Actuation | Active joints are updated with a kinematic servo preview for stable SurRoL-like teleoperation |
| Mimic joints | Passive pitch-linkage and jaw visual joints are synchronized manually because the converted MJCF does not encode URDF mimic joints |
| Metrics | Tip error, RCM error, joint-limit margin, and forbidden-contact count are reported through surgical task utilities |

The active PSM joints used by the demo are:

```text
psm_yaw
psm_pitch
psm_insertion
psm_roll
psm_wrist_pitch
psm_wrist_yaw
```

## Simulator Structure

```text
sim_with_mujoco/
  demo/
    dvrk_psm_teleop_demo.py       dVRK PSM needle-reach teleoperation entry
    pd_control_demo.py            PD-control demo
  environment/
    env.py                        MuJoCo model/data/viewer wrapper
  tasks/surgical/
    safety_metrics.py             tip, RCM, joint-limit, contact metrics
    target_sequence.py            timed reach-target sequence helper
  utils/
    dvrk_ik.py                    dVRK RCM-frame PSM IK
    ik.py                         DLS multi-target IK
    ik_qp.py                      differential IK
    dynamics.py                   computed torque and PD helpers
    collision.py                  MuJoCo contact helpers
    mj.py                         MuJoCo id mapping helpers
  viewer/
    surrol_keyboard_viewer.py     SurRoL-style keyboard preview viewer
    viewer.py                     generic MuJoCo viewer wrapper

assets/robots/dvrk/
  psm_surrol.xml                  SurRoL-derived PSM MJCF
  scene_psm_surrol_needle_reach.xml
                                  current default surgical reach scene
  surrol_psm/                     copied SurRoL source URDF and license
```


<!-- ## Current Limitations

- This is a research/portfolio simulator asset, not a clinically accurate dVRK dynamics model.
- The current surgical demo is reach-only. Needle grasping, gauze retrieval, suturing, and contact-rich manipulation are not implemented in the active entry.
- Keyboard input is a preview substitute for master-device teleoperation. It is not equivalent to real dVRK MTM or haptic-device pose control.
- The current IK prioritizes tip-position reach and RCM consistency. Tool orientation is not strongly constrained, so wrist motion can absorb part of the commanded target displacement.
- Passive mimic joints from the SurRoL URDF are synchronized in Python because the MJCF conversion does not yet include an equality-constraint replacement for URDF mimic behavior.
- Scene contacts and needle geometry are simplified proxies. -->

## Asset Sources

Primary upstream reference:

- SurRoL: https://github.com/med-air/SurRoL

## References

- MuJoCo XML modeling documentation
- MuJoCo computation and API documentation
- MuJoCo visualization documentation
- Gymnasium MuJoCo environment API
- Modern Robotics, Kevin M. Lynch and Frank C. Park
- Drake Differential IK: https://drake.mit.edu/doxygen_cxx/group__planning__kinematics.html
- robosuite Controllers: https://robosuite.ai/docs/modules/controllers.html
- dm_control: https://github.com/google-deepmind/dm_control
- ROBOTIS MuJoCo Menagerie assets: https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie
- robosuite assets: https://github.com/ARISE-Initiative/robosuite

## Tech Stack

- Python
- MuJoCo
- GLFW
- NumPy
- SciPy
- matplotlib
