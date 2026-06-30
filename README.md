![Simulator preview](/assets/images/img6.png)

## Simulator Overview

기존 MuJoCo 기반 simulator MyJoCo 위에 SurRoL 연구를 참고하여 dVRK PSM model, RCM(Remote Center of Motion) 기반 제어 관점, needle reach task를 추가했습니다. 특히 SurRoL의 PSM teleoperation 흐름과 RCM frame에서의 도구 끝점 제어 방식을 참고하면서, 일반적인 manipulator 제어와 다른 수술로봇 특화 제어 구조를 실험할 수 있도록 구성했습니다. 본 실험을 통해 의료특화 로봇이 왜 별도의 kinematic constraint와 task setup(예. 성공 기준인 tip error 정의)을 필요로 하는지 직접 확인할 수 있었습니다.

뼈대가 되는 기본 로봇시뮬레이터와 보다 자세한 기술적 설명은 다음 포스팅에서 확인 가능합니다. https://leewoojye.github.io/robotics/research/2026/06/03/myjoco3.html

### dVRK PSM Needle Reach Teleoperation

main entry file:

```text
sim_with_mujoco/demo/dvrk_psm_teleop_demo.py
```

프로젝트 루트에서 다음 명령으로 GUI demo를 실행합니다:

```bash
python -m sim_with_mujoco.demo.dvrk_psm_teleop_demo
```

demo는 기본적으로 SurRoL PSM model을 사용합니다:

```text
assets/robots/dvrk/scene_psm_surrol_needle_reach.xml
assets/robots/dvrk/psm_surrol.xml
```

현재는 needle reach task만을 수행한 상태이며 grasping, gauze retrieval, needle pickup task로까지의 확장을 목표로 하고 있습니다. PSM tool tip을 초록색 needle target 쪽으로 이동시키면서 shaft가 RCM constraint에서 크게 벗어나지 않도록 제어합니다.

### Controls

| Input | Behavior |
| --- | --- |
| W / S | tool-tip target을 world z 방향으로 이동 |
| A / D | tool-tip target을 world y 방향으로 이동 |
| Q / E | tool-tip target을 world x 방향으로 이동 |

<!-- | Mouse drag | MuJoCo free camera 회전, pan, 이동 |
| Mouse wheel | needle target focus 기준 zoom |
| V or M | fixed overview camera로 전환 |
| F | free camera로 복귀 | -->

좌측 상단 overlay에는 현재 task, action vector, tip error, RCM error가 표시됩니다.

## Installation

conda environment를 생성하고 활성화합니다:

```bash
conda create -n my_robotics python=3.12
conda activate my_robotics
```

프로젝트 루트에서 dependency를 설치합니다:

```bash
pip install -r requirements.txt
```

## Core Implementation

| Area | Current implementation |
| --- | --- |
| dVRK model | SurRoL psm_RL.urdf 기반 PSM chain을 MyJoCo MJCF naming, site, actuator 구조에 맞게 구성 |
| Input viewer | SurRoL 방식 keyboard input을 제공하는 단순한 GLFW/MuJoCo viewer인 SurrolKeyboardViewer class 사용 |
| Teleoperation target | keyboard input을 현재 tool-tip pose 주변의 작은 task-space target increment로 변환 |
| IK | solve_dvrk_rcm_ik에서 목표 world-frame tip target을 PSM RCM frame으로 변환한 뒤 yaw, pitch, insertion target 계산 |
| Actuation | 안정적인 SurRoL teleoperation preview를 위해 active joint를 kinematic servo 방식으로 갱신 |
| Mimic joints | 변환된 MJCF가 URDF mimic joint를 직접 표현하지 않으므로 pitch-linkage와 jaw visual joint를 Python에서 동기화 |
| Metrics | surgical task utility를 통해 tip error, RCM error를 계산 |

<!-- | Metrics | surgical task utility를 통해 tip error, RCM error, joint-limit margin, forbidden-contact count를 계산 | -->

## Simulator Structure

```text
sim_with_mujoco/
  demo/
    dvrk_psm_teleop_demo.py       dVRK PSM needle-reach teleoperation entry
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
```

<!-- ## Current Limitations

- This is a research/portfolio simulator asset, not a clinically accurate dVRK dynamics model.
- The current surgical demo is reach-only. Needle grasping, gauze retrieval, suturing, and contact-rich manipulation are not implemented in the active entry.
- Keyboard input is a preview substitute for master-device teleoperation. It is not equivalent to real dVRK MTM or haptic-device pose control.
- The current IK prioritizes tip-position reach and RCM consistency. Tool orientation is not strongly constrained, so wrist motion can absorb part of the commanded target displacement.
- Passive mimic joints from the SurRoL URDF are synchronized in Python because the MJCF conversion does not yet include an equality-constraint replacement for URDF mimic behavior.
- Scene contacts and needle geometry are simplified proxies. -->

## References

- MuJoCo documentation
- Modern Robotics, Kevin M. Lynch and Frank C. Park
- Drake Differential IK: https://drake.mit.edu/doxygen_cxx/group__planning__kinematics.html
- robosuite Controllers: https://robosuite.ai/docs/modules/controllers.html
- dm_control: https://github.com/google-deepmind/dm_control
- SurRoL: https://github.com/med-air/SurRoL

## Tech Stack

- Python
- MuJoCo
- GLFW
- NumPy
- SciPy
- matplotlib
