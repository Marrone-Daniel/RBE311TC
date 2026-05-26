from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


STATIC_APPEARANCE_BODIES = [
    "grooved_table",
    "fr5_fixed_base",
]

ROBOT_APPEARANCE_BODIES = [
    "base_link",
    "shoulder_link",
    "upperarm_link",
    "forearm_link",
    "wrist1_link",
    "wrist2_link",
    "wrist3_link",
    "robotiq_base_link",
    "left_driver",
    "left_coupler",
    "left_spring_link",
    "left_follower",
    "left_pad",
    "right_driver",
    "right_coupler",
    "right_spring_link",
    "right_follower",
    "right_pad",
]

DEFAULT_APPEARANCE_BODIES = STATIC_APPEARANCE_BODIES


def _body_by_name(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    return None


def _remove_asset(asset: ET.Element, tag: str, name: str) -> None:
    for child in list(asset):
        if child.tag == tag and child.get("name") == name:
            asset.remove(child)


def _is_visual_geom(geom: ET.Element) -> bool:
    if geom.get("group") == "3":
        return False
    cls = geom.get("class", "")
    if "collision" in cls or "pad_box" in cls:
        return False
    name = geom.get("name", "")
    if "collision" in name or "_col_" in name:
        return False
    return True


def remove_body(root: ET.Element, name: str) -> bool:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return False

    def rec(parent: ET.Element) -> bool:
        removed = False
        for child in list(parent):
            if child.tag == "body" and child.get("name") == name:
                parent.remove(child)
                removed = True
            elif child.tag == "body":
                removed = rec(child) or removed
        return removed

    return rec(worldbody)


def apply_texture_material_to_fr5_xml(
    model_xml: str | Path,
    *,
    texture_file_relative: str,
    texture_name: str = "fr5_appearance_tex",
    material_name: str = "fr5_appearance_mat",
    target_bodies: list[str] | None = None,
    remove_target_object: bool = True,
) -> dict[str, object]:
    model_xml = Path(model_xml)
    tree = ET.parse(model_xml)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    _remove_asset(asset, "texture", texture_name)
    _remove_asset(asset, "material", material_name)
    ET.SubElement(asset, "texture", {"name": texture_name, "type": "2d", "file": texture_file_relative})
    ET.SubElement(
        asset,
        "material",
        {
            "name": material_name,
            "texture": texture_name,
            "texrepeat": "1 1",
            "texuniform": "true",
            "rgba": "1 1 1 1",
        },
    )

    removed_target = remove_body(root, "target_object") if remove_target_object else False
    remove_body(root, "cube")
    targets = target_bodies or DEFAULT_APPEARANCE_BODIES
    changed_geoms: list[str] = []
    for body_name in targets:
        body = _body_by_name(root, body_name)
        if body is None:
            continue
        for geom in body.iter("geom"):
            if not _is_visual_geom(geom):
                continue
            geom.set("material", material_name)
            geom.attrib.pop("rgba", None)
            changed_geoms.append(geom.get("name") or f"{body_name}:{geom.get('mesh', geom.get('type', 'geom'))}")

    ET.indent(tree, space="  ")
    tree.write(model_xml, encoding="unicode")
    return {"model_xml": model_xml.as_posix(), "removed_target_object": removed_target, "changed_geoms": changed_geoms}
