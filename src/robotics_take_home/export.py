import json
import os
from dataclasses import asdict

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .physics import MassProperties
from .priors import Prior


def _vec(values) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def write_meshes(
    name: str, visual: trimesh.Trimesh, hulls: list[trimesh.Trimesh], mesh_dir: str
) -> tuple[bool, list[str]]:
    os.makedirs(mesh_dir, exist_ok=True)
    # write the obj text only (UVs kept); MuJoCo ignores .mtl, the texture goes in the MJCF
    obj_text, _ = trimesh.exchange.obj.export_obj(visual, return_texture=True)
    with open(os.path.join(mesh_dir, f"{name}_visual.obj"), "w") as f:
        f.writelines(
            line for line in obj_text.splitlines(keepends=True)
            if not line.startswith(("mtllib", "usemtl"))
        )
    texture = getattr(getattr(visual.visual, "material", None), "baseColorTexture", None)
    if texture is not None:
        texture.convert("RGB").save(os.path.join(mesh_dir, f"{name}_texture.png"))

    hull_files = []
    for i, hull in enumerate(hulls):
        filename = f"{name}_collision_{i:02d}.obj"
        hull.export(os.path.join(mesh_dir, filename))
        hull_files.append(filename)
    return texture is not None, hull_files


def model_xml(
    name: str, has_texture: bool, hull_files: list[str], props: MassProperties, prior: Prior
) -> str:
    """A free-floating rigid body meant to be <include>d into any MuJoCo scene."""
    if has_texture:
        texture_asset = (
            f'<texture type="2d" name="{name}_tex" file="{name}_texture.png"/>\n'
            f'    <material name="{name}_mat" texture="{name}_tex"/>'
        )
        visual_appearance = f'material="{name}_mat"'
    else:
        texture_asset, visual_appearance = "", 'rgba="0.2 0.5 0.9 1"'

    hull_assets = "\n".join(
        f'    <mesh name="{f[:-4]}" file="{f}"/>' for f in hull_files
    )
    hull_geoms = "\n".join(
        f'      <geom class="{name}_collision" mesh="{f[:-4]}"/>' for f in hull_files
    )
    I = props.inertia
    full_inertia = _vec([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])

    return f"""<mujoco model="{name}">
  <compiler meshdir="meshes" texturedir="meshes"/>
  <default>
    <default class="{name}_visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" mass="0"/>
    </default>
    <default class="{name}_collision">
      <geom type="mesh" group="3" mass="0" friction="{prior.friction} 0.005 0.0001"/>
    </default>
  </default>
  <asset>
    <mesh name="{name}_visual" file="{name}_visual.obj"/>
{hull_assets}
    {texture_asset}
  </asset>
  <worldbody>
    <body name="{name}">
      <freejoint name="{name}_free"/>
      <!-- mass from prior, distribution from the reconstructed volume ({props.source}) -->
      <inertial pos="{_vec(props.center_of_mass)}" mass="{props.mass:.6g}"
                fullinertia="{full_inertia}"/>
      <geom class="{name}_visual" mesh="{name}_visual" {visual_appearance}/>
{hull_geoms}
    </body>
  </worldbody>
</mujoco>
"""


def scene_xml(name: str, drop_height: float, friction: float) -> str:
    """Preview scene: floor, light, the asset, and a 'drop' keyframe used by the GIF."""
    # MuJoCo's euler="25 35 0" default is intrinsic xyz, which scipy spells "XYZ"
    x, y, z, w = Rotation.from_euler("XYZ", [25, 35, 0], degrees=True).as_quat()
    return f"""<mujoco model="{name}_scene">
  <include file="model.xml"/>
  <worldbody>
    <light pos="0 0 {drop_height * 3:.4g}" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"
          friction="{friction} 0.005 0.0001"/>
  </worldbody>
  <keyframe>
    <key name="drop" qpos="0 0 {drop_height:.6g} {_vec([w, x, y, z])}"/>
  </keyframe>
</mujoco>
"""


def write_package(
    out_dir: str,
    name: str,
    visual: trimesh.Trimesh,
    hulls: list[trimesh.Trimesh],
    props: MassProperties,
    prior: Prior,
    report: dict,
) -> str:
    has_texture, hull_files = write_meshes(name, visual, hulls, os.path.join(out_dir, "meshes"))
    with open(os.path.join(out_dir, "model.xml"), "w") as f:
        f.write(model_xml(name, has_texture, hull_files, props, prior))

    scene_path = os.path.join(out_dir, "scene.xml")
    with open(scene_path, "w") as f:
        f.write(scene_xml(name, drop_height=visual.extents.max() * 2, friction=prior.friction))

    mass = asdict(props)
    mass["center_of_mass"] = props.center_of_mass.tolist()
    mass["inertia"] = props.inertia.tolist()
    with open(os.path.join(out_dir, "properties.json"), "w") as f:
        json.dump({"name": name, "prior": asdict(prior), "mass_properties": mass, **report}, f, indent=2)
    return scene_path
