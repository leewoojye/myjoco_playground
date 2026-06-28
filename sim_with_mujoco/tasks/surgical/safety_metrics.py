from dataclasses import dataclass

import mujoco
import numpy as np

from sim_with_mujoco.utils.dvrk_ik import shaft_rcm_error


@dataclass
class SurgicalSafetyMetrics:
    tip_error: float
    rcm_error: float
    joint_margin: float
    forbidden_contacts: int

    def status(self, rcm_limit=0.003):
        if self.forbidden_contacts > 0:
            return "COLLISION"
        if self.rcm_error > rcm_limit:
            return "RCM WARN"
        return "OK"


def joint_limit_margin(model, data, joint_ids):
    margins = []
    for joint_id in joint_ids:
        if not model.jnt_limited[joint_id]:
            continue
        q = data.qpos[model.jnt_qposadr[joint_id]]
        lower, upper = model.jnt_range[joint_id]
        margins.append(min(q - lower, upper - q))
    if not margins:
        return float("inf")
    return float(min(margins))


def count_named_contact(model, data, name_fragment):
    count = 0
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        if name_fragment in geom1 or name_fragment in geom2:
            count += 1
    return count


def compute_surgical_metrics(
    model,
    data,
    tip_site_id,
    shaft_start_site_id,
    shaft_end_site_id,
    rcm_pos,
    target_pos,
    joint_ids,
):
    tip_pos = data.site_xpos[tip_site_id]
    rcm_error, _, _ = shaft_rcm_error(model, data, shaft_start_site_id, shaft_end_site_id, rcm_pos)
    return SurgicalSafetyMetrics(
        tip_error=float(np.linalg.norm(target_pos - tip_pos)),
        rcm_error=float(np.linalg.norm(rcm_error)),
        joint_margin=joint_limit_margin(model, data, joint_ids),
        forbidden_contacts=count_named_contact(model, data, "forbidden"),
    )
