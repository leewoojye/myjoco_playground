from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


ASSET_DIR = Path(__file__).resolve().parent
ROBOTS = ("psm", "ecm")

COMPILER_ATTRIBUTES = {
    "autolimits": "true",
    "balanceinertia": "true",
    "strippath": "false",
    "inertiafromgeom": "true",
    "alignfree": "true",
    "saveinertial": "true",
    "discardvisual": "false",
    "boundmass": "1e-8",
    "boundinertia": "1e-10",
}

# The SR-VPPV PSM URDF references three meshes that are absent upstream.
MESH_ALIASES = {
    "meshes/visual/CDF_base_no_shaft.obj": "meshes/visual/tool_roll_link.STL",
    "meshes/collision/CDF_base_no_shaft.obj": "meshes/collision/tool_roll_link.STL",
    "meshes/visual/CDF_base.obj": "meshes/visual/main_insertion_link_3.obj",
}

EQUALITY_SOLREF = "0.005 1"


def add_psm_sites(mjcf_root: ET.Element) -> None:
    worldbody = mjcf_root.find("worldbody")
    tool_body = mjcf_root.find(".//body[@name='psm_tool_yaw_link']")
    if worldbody is None or tool_body is None:
        raise ValueError("Converted PSM is missing its worldbody or tool-yaw body")

    ET.SubElement(
        worldbody,
        "site",
        {
            "name": "psm_rcm_site",
            "pos": "0 0.4864 0",
            "size": "0.008",
            "rgba": "0 0.85 0.35 1",
        },
    )
    ET.SubElement(
        tool_body,
        "site",
        {
            "name": "psm_tool_tip_site",
            "pos": "0 0.0102 0",
            "quat": "0.5 -0.5 -0.5 -0.5",
            "size": "0.006",
            "rgba": "1 0.25 0.1 1",
        },
    )


def exclude_ecm_mesh_overlap(mjcf_root: ET.Element) -> None:
    contact = ET.SubElement(mjcf_root, "contact")
    ET.SubElement(
        contact,
        "exclude",
        {
            "body1": "ecm_pitch_end_link",
            "body2": "ecm_tool_link",
        },
    )


def build_augmented_urdf(source_urdf: Path) -> Path:
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected a URDF robot root, got {root.tag!r}")

    mujoco_block = root.find("mujoco")
    if mujoco_block is None:
        mujoco_block = ET.Element("mujoco")
        root.insert(0, mujoco_block)

    compiler = mujoco_block.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_block, "compiler")
    for name, value in COMPILER_ATTRIBUTES.items():
        compiler.attrib.setdefault(name, value)

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename or (source_urdf.parent / filename).exists():
            continue
        alias = MESH_ALIASES.get(filename)
        if alias is None or not (source_urdf.parent / alias).exists():
            raise FileNotFoundError(f"Missing mesh referenced by {source_urdf}: {filename}")
        mesh.set("filename", alias)
        mesh.attrib.pop("scale", None)

    temp_file = tempfile.NamedTemporaryFile(
        dir=source_urdf.parent,
        prefix=f"{source_urdf.stem}_mujoco_",
        suffix=".urdf",
        delete=False,
    )
    augmented_urdf = Path(temp_file.name)
    temp_file.close()
    tree.write(augmented_urdf, encoding="utf-8", xml_declaration=True)
    return augmented_urdf


def actuator_parameters(joint_name: str, joint_type: str) -> tuple[str, str]:
    if joint_type == "prismatic":
        return "400", "120"
    if "jaw" in joint_name or "gripper" in joint_name:
        return "15", "20"
    return "60", "80"


def add_control_elements(source_urdf: Path, output_mjcf: Path) -> None:
    urdf_root = ET.parse(source_urdf).getroot()
    mjcf_tree = ET.parse(output_mjcf)
    mjcf_root = mjcf_tree.getroot()

    urdf_joints = [joint for joint in urdf_root.findall("joint") if joint.get("type") != "fixed"]
    joint_index = {joint.get("name"): index for index, joint in enumerate(urdf_joints)}
    mjcf_joints = {
        joint.get("name"): joint
        for joint in mjcf_root.findall(".//joint")
        if joint.get("name")
    }
    for joint in mjcf_joints.values():
        joint.set("damping", "2")
        joint.set("armature", "0.02")
        joint.set("frictionloss", "0.01")

    if source_urdf.stem == "psm":
        add_psm_sites(mjcf_root)
    elif source_urdf.stem == "ecm":
        exclude_ecm_mesh_overlap(mjcf_root)

    option = mjcf_root.find("option")
    if option is None:
        option = ET.Element("option")
        mjcf_root.insert(1, option)
    option.set("timestep", "0.002")
    option.set("iterations", "80")
    option.set("integrator", "implicitfast")

    mimic_joints: list[tuple[int, str, str, float, float]] = []
    mimic_names: set[str] = set()
    driver_rank: dict[str, int] = {}
    for index, joint in enumerate(urdf_joints):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        dependent = joint.get("name")
        driver = mimic.get("joint")
        if dependent not in mjcf_joints or driver not in mjcf_joints:
            raise ValueError(f"Unknown mimic relation: {dependent} -> {driver}")
        multiplier = float(mimic.get("multiplier", "1"))
        offset = float(mimic.get("offset", "0"))
        mimic_joints.append((index, dependent, driver, multiplier, offset))
        mimic_names.add(dependent)
        driver_rank[driver] = min(driver_rank.get(driver, joint_index[driver]), index)

    equality = ET.SubElement(mjcf_root, "equality")
    for _, dependent, driver, multiplier, offset in mimic_joints:
        ET.SubElement(
            equality,
            "joint",
            {
                "name": f"eq_{dependent}",
                "joint1": dependent,
                "joint2": driver,
                "polycoef": f"{offset:g} {multiplier:g} 0 0 0",
                "solref": EQUALITY_SOLREF,
            },
        )

    actuated_joints = [
        (index, joint)
        for index, joint in enumerate(urdf_joints)
        if joint.get("name") not in mimic_names
    ]
    actuated_joints.sort(key=lambda item: driver_rank.get(item[1].get("name"), item[0]))

    actuator = ET.SubElement(mjcf_root, "actuator")
    for _, urdf_joint in actuated_joints:
        joint_name = urdf_joint.get("name")
        mjcf_joint = mjcf_joints.get(joint_name)
        if mjcf_joint is None:
            raise ValueError(f"Joint missing from converted MJCF: {joint_name}")
        joint_range = mjcf_joint.get("range")
        if joint_range is None:
            raise ValueError(f"Actuated joint has no range: {joint_name}")
        kp, force = actuator_parameters(joint_name, urdf_joint.get("type", "revolute"))
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"ctrl_{joint_name}",
                "joint": joint_name,
                "kp": kp,
                "ctrlrange": joint_range,
                "forcelimited": "true",
                "forcerange": f"-{force} {force}",
            },
        )

    ET.indent(mjcf_tree, space="  ")
    mjcf_tree.write(output_mjcf, encoding="unicode")
    with output_mjcf.open("a", encoding="utf-8") as output:
        output.write("\n")


def convert(source_urdf: Path, output_mjcf: Path) -> None:
    augmented_urdf = build_augmented_urdf(source_urdf)
    try:
        model = mujoco.MjModel.from_xml_path(str(augmented_urdf))
        mujoco.mj_saveLastXML(str(output_mjcf), model)
        add_control_elements(source_urdf, output_mjcf)
    finally:
        augmented_urdf.unlink(missing_ok=True)


def main() -> None:
    for robot in ROBOTS:
        source_urdf = ASSET_DIR / robot / f"{robot}.urdf"
        output_mjcf = ASSET_DIR / robot / f"{robot}.xml"
        convert(source_urdf, output_mjcf)
        print(f"Converted {source_urdf} -> {output_mjcf}")


if __name__ == "__main__":
    main()
