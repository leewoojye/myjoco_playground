from __future__ import annotations

import copy
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import xacro


ASSET_DIR = Path(__file__).resolve().parent
DVRK_MODEL_DIR = ASSET_DIR / "dvrk_model"
OUTPUT_URDF = ASSET_DIR / "dvrk_si_patient_side.urdf"
OUTPUT_MJCF = ASSET_DIR / "dvrk_si_patient_side.xml"
GENERATED_MESH_DIR = ASSET_DIR / "generated_meshes"

COMPILER_ATTRIBUTES = {
    "angle": "radian",
    "autolimits": "true",
    "balanceinertia": "true",
    "strippath": "false",
    "inertiafromgeom": "true",
    "alignfree": "true",
    "saveinertial": "true",
    "discardvisual": "false",
    "fusestatic": "false",
    "boundmass": "1e-8",
    "boundinertia": "1e-10",
}

SUJ_INSTANCES = (
    ("PSM1_", "0 0.228 0.528", "0 0 1.57079632679"),
    ("PSM2_", "0 -0.228 0.528", "0 0 -1.57079632679"),
    ("PSM3_", "-0.223 0 0.528", "0 0 -3.14159265359"),
    ("ECM_", "0.223 0 0.528", "0 0 0"),
)

ARM_INSTANCES = (
    ("PSM1", "psm.urdf.xacro"),
    ("PSM2", "psm.urdf.xacro"),
    ("PSM3", "psm.urdf.xacro"),
    ("ECM", "ecm.urdf.xacro"),
)

CONTROLLED_JOINTS = (
    "PSM1_yaw",
    "PSM1_pitch",
    "PSM1_insertion",
    "PSM1_roll",
    "PSM1_wrist_pitch",
    "PSM1_wrist_yaw",
    "PSM1_jaw",
)
ACTUATOR_KV = {
    "PSM1_yaw": "2",
    "PSM1_pitch": "2",
    "PSM1_insertion": "10",
    "PSM1_roll": "0.5",
    "PSM1_wrist_pitch": "0.5",
    "PSM1_wrist_yaw": "0.5",
    "PSM1_jaw": "0.5",
}
HOME_JOINT_POSITIONS = {
    "SUJ_PSM1_J0": 0.005,
    "SUJ_PSM1_J1": -0.8792,
    "SUJ_PSM1_J2": 0.3702,
    "SUJ_PSM1_J3": 1.6385,
    "PSM1_yaw": -0.0472,
    "PSM1_pitch": -1.0466,
    "PSM1_insertion": 0.24,
    "PSM1_jaw": 0.30,
    "SUJ_PSM2_J0": 0.10,
    "SUJ_PSM2_J2": 3.0,
    "PSM2_pitch": -1.20,
    "SUJ_PSM3_J0": 0.20,
    "SUJ_PSM3_J1": 1.50,
    "SUJ_PSM3_J2": 2.80,
    "SUJ_PSM3_J3": -2.80,
    "SUJ_PSM3_J4": -3.0,
    "PSM3_pitch": -1.20,
    "SUJ_ECM_J0": 0.10924,
    "SUJ_ECM_J1": 1.54406,
    "SUJ_ECM_J2": 0.27367,
    "SUJ_ECM_J3": 0.139,
    "ECM_yaw": 2.2676,
    "ECM_pitch": -1.47363,
}
EQUALITY_SOLREF = "0.005 1"


def stage_xacros(temp_dir: Path) -> Path:
    staged_model = temp_dir / "dvrk_model"
    shutil.copytree(DVRK_MODEL_DIR / "urdf", staged_model / "urdf")

    replacement = str(staged_model)
    for source in (staged_model / "urdf").rglob("*.xacro"):
        text = source.read_text(encoding="utf-8")
        source.write_text(text.replace("$(find dvrk_model)", replacement), encoding="utf-8")
    return staged_model


def build_suj_wrapper(staged_model: Path, temp_dir: Path) -> Path:
    common = staged_model / "urdf" / "common.urdf.xacro"
    suj_column = staged_model / "urdf" / "Si" / "suj_column.urdf.xacro"
    suj = staged_model / "urdf" / "Si" / "suj.urdf.xacro"
    instances = "\n".join(
        f'  <xacro:suj prefix="{prefix}" parent_link="SUJ_column" xyz="{xyz}" rpy="{rpy}"/>'
        for prefix, xyz, rpy in SUJ_INSTANCES
    )
    wrapper = temp_dir / "suj_wrapper.urdf.xacro"
    wrapper.write_text(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"dvrk_si_patient_side\" xmlns:xacro=\"http://www.ros.org/wiki/xacro\">\n"
        f"  <xacro:include filename=\"{common}\"/>\n"
        f"  <xacro:include filename=\"{suj_column}\"/>\n"
        f"  <xacro:include filename=\"{suj}\"/>\n"
        "  <xacro:suj_column/>\n"
        f"{instances}\n"
        "</robot>\n",
        encoding="utf-8",
    )
    return wrapper


def process_xacro(path: Path, mappings: dict[str, str] | None = None) -> ET.Element:
    document = xacro.process_file(str(path), mappings=mappings or {})
    return ET.fromstring(document.toxml())


def prefix_arm_joints(root: ET.Element, arm: str) -> None:
    prefix = f"{arm}_"
    names: dict[str, str] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name:
            continue
        names[name] = name if name.startswith(prefix) else f"{prefix}{name}"

    for joint in root.findall("joint"):
        name = joint.get("name")
        if name:
            joint.set("name", names[name])
        mimic = joint.find("mimic")
        if mimic is not None:
            driver = mimic.get("joint")
            if driver in names:
                mimic.set("joint", names[driver])


def merge_arm(base: ET.Element, arm_root: ET.Element, arm: str) -> None:
    mounting_point = f"{arm}_mounting_point"
    prefix_arm_joints(arm_root, arm)

    for link in arm_root.findall("link"):
        if link.get("name") != mounting_point:
            base.append(copy.deepcopy(link))
    for joint in arm_root.findall("joint"):
        base.append(copy.deepcopy(joint))


def add_mujoco_compiler(root: ET.Element) -> None:
    mujoco_block = ET.Element("mujoco")
    ET.SubElement(mujoco_block, "compiler", COMPILER_ATTRIBUTES)
    root.insert(0, mujoco_block)


def normalize_mesh_paths(root: ET.Element) -> None:
    prefix = "package://dvrk_model/"
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and filename.startswith(prefix):
            mesh.set("filename", f"dvrk_model/{filename[len(prefix):]}")


def make_mesh_basenames_unique(root: ET.Element) -> None:
    """Avoid MuJoCo's URDF mesh-name collisions between repeated link_N.STL files."""
    GENERATED_MESH_DIR.mkdir(exist_ok=True)
    aliases: dict[str, str] = {}
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        if filename not in aliases:
            source = ASSET_DIR / filename
            relative = source.relative_to(DVRK_MODEL_DIR / "meshes")
            stem = re.sub(r"[^A-Za-z0-9]+", "_", str(relative.with_suffix(""))).strip("_")
            destination = GENERATED_MESH_DIR / f"{stem}{source.suffix}"
            shutil.copy2(source, destination)
            aliases[filename] = str(destination.relative_to(ASSET_DIR))
        mesh.set("filename", aliases[filename])


def remove_degenerate_geometries(root: ET.Element) -> None:
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for element in list(link.findall(tag)):
                box = element.find("geometry/box")
                if box is None:
                    continue
                size = [float(value) for value in box.get("size", "").split()]
                if len(size) == 3 and not any(size):
                    link.remove(element)


def validate_unique_names(root: ET.Element) -> None:
    for tag in ("link", "joint"):
        names = [element.get("name") for element in root.findall(tag)]
        duplicates = sorted({name for name in names if name and names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate {tag} names: {duplicates}")


def build_urdf() -> ET.Element:
    with tempfile.TemporaryDirectory(prefix="dvrk_si_xacro_") as temp:
        temp_dir = Path(temp)
        staged_model = stage_xacros(temp_dir)
        root = process_xacro(build_suj_wrapper(staged_model, temp_dir))

        for arm, filename in ARM_INSTANCES:
            arm_root = process_xacro(
                staged_model / "urdf" / "Si" / filename,
                {"arm": arm, "parent_link_": f"{arm}_mounting_point"},
            )
            merge_arm(root, arm_root, arm)

    add_mujoco_compiler(root)
    normalize_mesh_paths(root)
    make_mesh_basenames_unique(root)
    remove_degenerate_geometries(root)
    validate_unique_names(root)
    return root


def write_urdf(root: ET.Element) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT_URDF, encoding="utf-8", xml_declaration=True)


def actuator_parameters(joint_name: str, joint_type: str) -> tuple[str, str]:
    if joint_name.startswith("SUJ_"):
        return ("1000", "1200") if joint_type == "prismatic" else ("500", "600")
    if joint_type == "prismatic" or "insertion" in joint_name:
        return "400", "160"
    if "jaw" in joint_name:
        return "20", "30"
    return "80", "120"


def add_site(parent: ET.Element, name: str, pos: str, rgba: str) -> None:
    ET.SubElement(parent, "site", {"name": name, "pos": pos, "size": "0.006", "rgba": rgba})


def home_positions(urdf_joints: list[ET.Element]) -> dict[str, float]:
    positions = {joint.get("name"): 0.0 for joint in urdf_joints}
    positions.update(HOME_JOINT_POSITIONS)
    for joint in urdf_joints:
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        positions[joint.get("name")] = float(mimic.get("offset", "0")) + float(
            mimic.get("multiplier", "1")
        ) * positions[mimic.get("joint")]
    return positions


def postprocess_mjcf(urdf_root: ET.Element) -> None:
    tree = ET.parse(OUTPUT_MJCF)
    root = tree.getroot()
    root.set("model", "dvrk_si_patient_side")

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", "0.002")
    option.set("iterations", "80")
    option.set("integrator", "implicitfast")

    mjcf_joints = {
        joint.get("name"): joint
        for joint in root.findall(".//joint")
        if joint.get("name")
    }
    urdf_joints = [joint for joint in urdf_root.findall("joint") if joint.get("type") != "fixed"]

    for name, joint in mjcf_joints.items():
        joint.set("damping", "5" if name.startswith("SUJ_") else "2")
        joint.set("armature", "0.02")
        joint.set("frictionloss", "0.01")

    # MuJoCo imports URDF visuals as group 1 and collisions as group 0. Keep only
    # the visual copy while collision geometry for this assembly is disabled.
    for body in root.findall(".//body"):
        for geom in list(body.findall("geom")):
            if geom.get("group") != "1":
                body.remove(geom)

    mimic_names: set[str] = set()
    equality = ET.SubElement(root, "equality")
    for joint in urdf_joints:
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        dependent = joint.get("name")
        driver = mimic.get("joint")
        if dependent not in mjcf_joints or driver not in mjcf_joints:
            raise ValueError(f"Unknown mimic relation: {dependent} -> {driver}")
        mimic_names.add(dependent)
        multiplier = float(mimic.get("multiplier", "1"))
        offset = float(mimic.get("offset", "0"))
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

    positions = home_positions(urdf_joints)
    for joint in urdf_joints:
        name = joint.get("name")
        if name in mimic_names or name in CONTROLLED_JOINTS:
            continue
        ET.SubElement(
            equality,
            "joint",
            {
                "name": f"lock_{name}",
                "joint1": name,
                "polycoef": f"{positions[name]:g} 0 0 0 0",
                "solref": EQUALITY_SOLREF,
            },
        )

    actuator = ET.SubElement(root, "actuator")
    for urdf_joint in urdf_joints:
        name = urdf_joint.get("name")
        if name not in CONTROLLED_JOINTS:
            continue
        mjcf_joint = mjcf_joints.get(name)
        if mjcf_joint is None:
            raise ValueError(f"Joint missing from converted MJCF: {name}")
        joint_range = mjcf_joint.get("range")
        if joint_range is None:
            raise ValueError(f"Actuated joint has no range: {name}")
        kp, force = actuator_parameters(name, urdf_joint.get("type", "revolute"))
        ET.SubElement(
            actuator,
            "position",
            {
                "name": name,
                "joint": name,
                "kp": kp,
                "kv": ACTUATOR_KV[name],
                "ctrlrange": joint_range,
                "forcelimited": "true",
                "forcerange": f"-{force} {force}",
            },
        )

    worldbody = root.find("worldbody")
    qpos = [positions[joint.get("name")] for joint in worldbody.findall(".//joint")]
    ctrl = [positions[element.get("joint")] for element in actuator]
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "dvrk_home",
            "qpos": " ".join(f"{value:g}" for value in qpos),
            "ctrl": " ".join(f"{value:g}" for value in ctrl),
        },
    )

    rcm_body = root.find(".//body[@name='PSM1_RCM']")
    tip_body = root.find(".//body[@name='PSM1_tool_wrist_sca_ee_link']")
    if rcm_body is None or tip_body is None:
        raise ValueError("Converted model is missing the PSM1 RCM or tool-tip body")
    add_site(rcm_body, "PSM1_rcm_site", "0 0 0", "0 0.85 0.35 1")
    add_site(tip_body, "PSM1_tool_tip_site", "0 0 0.010", "1 0.25 0.1 1")

    ET.indent(tree, space="  ")
    tree.write(OUTPUT_MJCF, encoding="unicode")
    with OUTPUT_MJCF.open("a", encoding="utf-8") as output:
        output.write("\n")


def main() -> None:
    urdf_root = build_urdf()
    write_urdf(urdf_root)
    model = mujoco.MjModel.from_xml_path(str(OUTPUT_URDF))
    mujoco.mj_saveLastXML(str(OUTPUT_MJCF), model)
    postprocess_mjcf(urdf_root)
    mujoco.MjModel.from_xml_path(str(OUTPUT_MJCF))
    print(f"Converted {OUTPUT_URDF} -> {OUTPUT_MJCF}")


if __name__ == "__main__":
    main()
