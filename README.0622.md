
![Simulator preview](/assets/images/myjoco3.png)

## DEMO

<!-- ### 26/06/22
https://youtu.be/ZT2nsVZF6J0

00:03 ~ 01:14 Kinematic simulator Demo

01:17 ~ 02:32 Dynamic simulator (position) Demo

02:35 ~ 03:29 Dynamic simulator (motor) Demo

### 26/06/12
https://youtu.be/5F9DRPQdj8Y

00:03 ~ 01:03 Kinematic simulator Demo

01:06 ~ 02:30 Dynamic simulator (position) Demo

02:33 ~ 03:31 Dynamic simulator (motor) Demo -->

## Simulator Overview

MyJoCo(feat. MuJoCo)는 MuJoCo-free 구시뮬레이터를 MuJoCo-based 시뮬레이터로 재탄생한 프로젝트입니다. MJCF을 제외한 MyJoCo 코드는 vibe coding 없이 from scratch로 제작되었습니다.

Check out the more detailed implementation journey here !

https://leewoojye.github.io/research/2026/06/03/myjoco2.html

[update] A newsletter feature has been added to personal blogs. If you would like to receive the newsletter, please subscribe !

## How to Use

Create and activate a conda environment:

```bash
conda create -n my_robotics python=3.12
conda activate my_robotics
```

Install the required Python packages from the project root:

```bash
pip install -r requirements.txt
```

Run the simulator:

**Dynamic simulator using position actuator**

```bash
python3 -m sim_with_mujoco.dynamic_simulator_position
```

**Dynamic simulator using motor actuator**

```bash
python3 -m sim_with_mujoco.dynamic_simulator_motor
```

**Kinematic simulator**

```bash
python3 -m sim_with_mujoco.kinematic_simulator
```

Use the right-side GUI panels to move the right hand target and control the right-hand grasp sliders.

## Core Implementation

<!-- | 엔트리 파일 | 상태 갱신 방식 | 궤적 형성 방식 | ctrl 입력 | 물리 계산 | 용도 |
| --- | --- | --- | --- | --- | --- |
| dynamic_simulator_position.py | IK 목표 관절각을 position actuator ctrl로 전달 | panel target을 task-space pose로 보간한 뒤 IK 수행 | ctrl = target qpos, mujoco position servo가 torque 계산 | mj_step 사용, qfrc_bias를 qfrc_applied에 더해 중력 보상 실험 | baseline dynamic simulator |
| dynamic_simulator_motor.py | actual/reference state 기반 differential IK로 관절 목표를 갱신 | panel target을 바로 pose target으로 쓰고 매 부분 IK 수행 | arm은 computed torque, finger는 motor PD torque를 data.ctrl에 입력 | mj_step 사용, 접촉 여부에 따라 finger gain 조절 | motor actuator 기반 torque-level teleoperation / grasping 실험 |
| kinematic_simulator.py | IK 결과를 data.qpos에 직접 대입 | panel target을 task-space pose로 보간한 뒤 IK 수행 | torque 계산 없이 ctrl만 qpos와 동기화 | mj_forward 사용, data.time 수동 증가 | IK와 trajectory 동작 확인용 kinematic simulator | -->

| 영역 | 구현 내용 |
| --- | --- |
| Environment, Viewer class | model, data, viewer, mujoco API wrapper를 묶어서 관리하는 Environment class / GLFW와 mujoco rendering API를 묶은 Viewer class, camera 조종 패널 추가, event handler에서 polling 중심 구조로 변경 (reference: dm_control) |
| Kinematic simulator | IK 결과를 data.qpos에 직접 반영하고 mj_forward로 상태를 갱신 |
| Dynamic simulator | IK 결과를 actuator ctrl에 넣고 mj_step으로 mujoco dynamics 진행 |
| 시뮬레이션 공통 | rendering, polling, trajectory generation, simulation 시간축 분리 및 적절한 주기(ex. trajectory_duration, poll_interval) 탐색 |
| Multi target IK | multi target의 jacobian과 error를 쌓는 get_stacked_ik 함수, damped least squares로 IK 계산, 클리핑 로직 최적화 |
| Differential IK | actual state 기준으로 multi target jacobian을 구성하고, least-squares로 qvel target과 다음 q target 계산 |
| Trajectory | pose interpolation(시작점 속도/가속도가 비영으로 부드러운 궤적 전환 도모), joint-space interpolation |
| Dynamics utility | computed torque, PD controller 모듈을 구현하고, mujoco timestep마다 각각 팔과 손가락 제어를 담당 (reference: robosuite) |
| Collision utility | mujoco data.contact 기반 robot-table, finger-object 접촉 판정 |
| mujoco utility | 편의를 위한 mujoco API wrapper (ex. joint id, dof id, actuator id 매핑) |
| Assets | motor actuator로 구성된 ffw MJCF 파일 추가, 손가락 마디 사이에 self-collision을 exclude 태그로 임시 방지 |

<!-- | Planning(현재 미사용) | joint trajectory planner: 궤적들 간 qvel을 공유 + singularity check (reference: ROS2) | -->

### Simulator Structure

<!-- ```text
sim_with_mujoco/
  dynamic_simulator_position.py position actuator 기반 dynamic entry
  dynamic_simulator_motor.py    motor actuator 기반 torque-control entry
  kinematic_simulator.py        qpos 직접 갱신 기반 kinematic entry
  environment/
    env.py                      model, data, viewer wrapper
  mjcf/
    parser.py                   MJCF parser
  utils/
    ik.py                       DLS multi-target IK
    ik_qp.py                    differential IK
    dynamics.py                 CT, PD, task-space control
    kinematics.py               finger / kinematics helper
    collision.py                contact helper
    math3d.py                   transform / twist helper
    mj.py                       mujoco id mapping helper
    planning.py                 waypoint / trajectory helper
  viewer/
    viewer.py                   renderer wrapper
    glfw_panel.py               hand target panel
    gui_panel.py                camera control panel
sim/model/motion/
  trajectory.py                 task / joint trajectory math
``` -->

## Limitation (Future Implementation Plan)

<!-- - IK 행렬이 단순히 stack한 구조라 task 간 우선순위가 없고 직관적으로 position IK를 적용한 왼손의 target error를 과소평가할 것 같습니다. 실제로 시뮬레이션 상에서 왼손은 오른손에 비해 자리를 이탈하는 경우가 많았습니다. 
- IK의 damping과 dq 제한이 singularity, collision 조건까지 만족시키지 않습니다. 별도 planning 모듈을 추가하거나 Differential IK 모듈 제약을 보완해야 합니다.
- 향후에 task-space PD외에 mass matrix, null-space posture control을 포함한 operational space controller를 구현해야 합니다.
- finger interpolation은 grasp synergy를 모델링하지 않고 open/close alpha를 관절 목표로 직접 매핑해서 손의 실제 닫힘을 충분히 표현하지 못합니다.
- viewer diagnostics는 visual contact와 body BVH에 의존하고 있어 q_des saturation, ctrl saturation, qfrc_bias 보상 여부 같은 controller 내부 상태를 즉시 확인하기 어렵습니다. timestep별 mjData 대시보드도 만들면 좋을 것 같습니다. 생성된 궤적(ref)을 얼마나 준수했는지 평가하는 지표도 시각화할 수 있습니다.
- dm_control과 달리 현재 Environment wrapper는 state snapshot과 restore 경계가 약해서 IK용 임시 MjData, simulation MjData, viewer가 보는 MjData가 섞여있습니다.
- 현재 trajectory duration은 고정값인데, 인접 velocity 등을 바탕으로 동적으로 바꿔볼 수 있습니다.
- 현재 충돌 감지 모듈은 kinematic rollback 중심이라 접촉면을 따라 미끄러짐을 표현하지 못합니다. 또한 robot-table hard collision에 초점이 맞춰진 모듈을 확장해야 합니다. 한편 kinematic mode 기능 범위가 헷갈려 사용처를 더 조사해야 합니다.
- 점진적으로 강화학습 데모를 붙여봅니다. -->

## References

- mujoco XML modeling documentation
- mujoco computation and API documentation
- mujoco visualization documentation
- Gymnasium mujoco environment API
- Modern Robotics, Kevin M. Lynch and Frank C. Park: Ch. 3 Rigid-Body Motions, Ch. 5 Velocity Kinematics and Statics, Ch. 6 Inverse Kinematics, Ch. 8 Dynamics of Open Chains, Ch. 9 Trajectory Generation, Ch. 11 Robot Control, Ch. 12 Grasping and Manipulation
- Drake Differential IK: https://drake.mit.edu/doxygen_cxx/group__planning__kinematics.html
- robosuite Controllers: https://robosuite.ai/docs/modules/controllers.html
- dm_control: https://github.com/google-deepmind/dm_control
- ROBOTIS mujoco Menagerie assets: https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie
- robosuite assets: https://github.com/ARISE-Initiative/robosuite

## Tech Stack

- MuJoCo
- GLFW
- SciPy
- matplotlib
- Python
- NumPy
