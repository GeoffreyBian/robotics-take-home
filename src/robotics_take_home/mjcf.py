"""MuJoCo export: model.xml (the asset, includable anywhere) and scene.xml (a preview stage)."""

import numpy as np
from scipy.spatial.transform import Rotation

from .asset import Part, tree_order
from .materials import FLOOR, MaterialTable

# default 0.02 s contact is too soft for small objects (5 cm bolt sank 17% on impact);
# 0.005 s is still >= 2x MuJoCo's default timestep, so it stays stable in any scene.
# solmix makes ours win when mixed with a default-soft floor in someone else's scene
SOLREF = "0.005 1"
SOLMIX = 100
TORSIONAL, ROLLING = 0.005, 0.0001


def _vec(values) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def collision_geom_names(part: Part) -> list[str]:
    count = 1 if part.sphere is not None else len(part.hulls)
    return [f"{part.name}_c{i:02d}" for i in range(count)]


def _body(part: Part, children: dict, parent_origin: np.ndarray, name: str,
          materials: MaterialTable, depth: int) -> str:
    pad = "  " * depth
    origin = part.body_origin()
    lines = [f'{pad}<body name="{part.name}" pos="{_vec(origin - parent_origin)}">']
    j = part.joint
    if j is None:
        lines.append(f'{pad}  <freejoint name="{name}_free"/>')
    elif j.kind == "hinge":
        # degrees: a <compiler angle> here would leak into whatever scene includes us
        limit = f' range="{_vec(np.degrees(j.range))}" limited="true"' if j.range else ""
        lines.append(
            f'{pad}  <joint name="{part.name}_joint" type="hinge" axis="{_vec(j.axis)}"{limit}'
            f' damping="{j.damping:.4g}" stiffness="{j.stiffness:.4g}" springref="0"/>'
        )
    else:
        lines.append(f'{pad}  <joint name="{part.name}_joint" type="ball" damping="{j.damping:.4g}"/>')

    m = part.mass
    I = m.inertia
    lines.append(
        f'{pad}  <inertial pos="{_vec(m.center_of_mass - origin)}" mass="{m.mass:.6g}"'
        f' fullinertia="{_vec([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])}"/>'
    )
    if part.liquid:
        # the pendulum's mass marker; hidden (press 4 in the viewer). The beads show the liquid
        lines.append(
            f'{pad}  <geom class="{name}_visual" mesh="{part.name}_visual" pos="{_vec(-origin)}"'
            f' group="4" rgba="0.15 0.45 0.95 0.6"/>'
        )
    else:
        lines.append(
            f'{pad}  <geom class="{name}_visual" mesh="{part.name}_visual" pos="{_vec(-origin)}"'
            f' material="{name}_{part.material}"/>'
        )
    for geom, material in zip(collision_geom_names(part), part.collision_materials()):
        mu = materials[material].default_friction
        if part.sphere is not None:
            center, radius = part.sphere
            shape = f'type="sphere" size="{radius:.6g}" pos="{_vec(center - origin)}"'
        else:
            shape = f'mesh="{geom}" pos="{_vec(-origin)}"'
        lines.append(f'{pad}  <geom name="{geom}" class="{name}_collision" {shape} friction="{mu} {TORSIONAL} {ROLLING}"/>')
    for child in children.get(part.name, []):
        if child.beads is not None:
            lines += _cavity_walls(child.beads, origin, name, pad)
    for child in children.get(part.name, []):
        lines.append(_body(child, children, origin, name, materials, depth + 1))
    lines.append(f"{pad}</body>")
    return "\n".join(lines)


def _cavity_walls(beads: dict, origin: np.ndarray, name: str, pad: str) -> list[str]:
    """Six hidden boxes around the bead cavity, rigid with the container."""
    lo, hi = beads["lo"] - origin, beads["hi"] - origin
    center, half = (lo + hi) / 2, (hi - lo) / 2
    # thick enough that a fast bead cannot tunnel through in one step, whatever its size
    t = max(2 * beads["radius"], 0.01)
    lines = []
    for axis in range(3):
        for sign in (-1, 1):
            pos, size = center.copy(), half.copy() + t
            pos[axis] += sign * (half[axis] + t)
            size[axis] = t
            lines.append(f'{pad}  <geom class="{name}_cavity" type="box" pos="{_vec(pos)}" size="{_vec(size)}"/>')
    return lines


def _bead_bodies(parts: list[Part], name: str) -> list[str]:
    """Free bodies, so they must sit at the top level of the worldbody."""
    lines = []
    for p in parts:
        if p.beads is None:
            continue
        for i, pos in enumerate(p.beads["positions"]):
            lines.append(
                f'    <body name="{p.name}_bead{i:03d}" pos="{_vec(pos)}"><freejoint/>'
                f'<geom class="{name}_bead" size="{p.beads["radius"]:.5g}"/></body>'
            )
    return lines


def model_xml(name: str, parts: list[Part], textures: dict[str, str], materials: MaterialTable) -> str:
    parts = tree_order(parts)
    children: dict = {}
    for p in parts:
        children.setdefault(p.parent, []).append(p)

    texture_assets = [
        f'    <texture name="{name}_{role}" type="2d" file="{f}" colorspace="{"sRGB" if role == "rgb" else "linear"}"/>'
        for role, f in textures.items() if role != "orm"
    ]
    material_assets = []
    for material in sorted({p.material for p in parts}):
        look = materials[material]
        # no ORM layer: without image-based lighting MuJoCo renders metallic surfaces near
        # black. The USD keeps it, where a real renderer can use it
        layers = "".join(f'<layer texture="{name}_{role}" role="{role}"/>' for role in textures if role != "orm")
        material_assets.append(
            f'    <material name="{name}_{material}" specular="{look.specular}" shininess="{look.shininess}"'
            # alpha < 1 for clear materials (PET bottle), so the liquid inside shows through
            f' rgba="1 1 1 {look.opacity}"'
            + (f">{layers}</material>" if layers else "/>")
        )
    mesh_assets = []
    for p in parts:
        mesh_assets.append(f'    <mesh name="{p.name}_visual" file="{p.name}_visual.obj"/>')
        if p.sphere is None:
            mesh_assets += [f'    <mesh name="{g}" file="{g}.obj"/>' for g in collision_geom_names(p)]

    return f"""<mujoco model="{name}">
  <compiler meshdir="meshes" texturedir="meshes"/>
  <default>
    <default class="{name}_visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" mass="0"/>
    </default>
    <default class="{name}_collision">
      <geom type="mesh" group="3" mass="0" solref="{SOLREF}" solmix="{SOLMIX}"/>
    </default>
    <!-- visual-only liquid: beads (contact bit 2) touch only the cavity walls and each other;
         frictionless so they flow. The liquid's real mass is on the pendulum body -->
    <default class="{name}_bead">
      <geom type="sphere" contype="2" conaffinity="2" condim="1" group="2" mass="0.0001"
            rgba="0.2 0.5 0.95 0.75" solref="0.004 1"/>
    </default>
    <default class="{name}_cavity">
      <geom contype="2" conaffinity="2" condim="1" group="5" mass="0" rgba="0.5 0.5 1 0.2"/>
    </default>
  </default>
  <asset>
{chr(10).join(texture_assets + material_assets + mesh_assets)}
  </asset>
  <worldbody>
{_body(parts[0], children, np.zeros(3), name, materials, 2)}
{chr(10).join(_bead_bodies(parts, name))}
  </worldbody>
</mujoco>
"""


def default_qpos(parts: list[Part], root_pos, root_quat) -> list[float]:
    qpos = list(root_pos) + list(root_quat)
    w, x, y, z = root_quat
    for p in tree_order(parts)[1:]:
        if p.joint.kind == "hinge":
            qpos += [0.0]
        elif p.liquid:
            # undo the root's tilt so the liquid starts hanging level, as liquid at rest would
            qpos += [w, -x, -y, -z]
        else:
            qpos += [1.0, 0.0, 0.0, 0.0]
    # beads ride along with the root's starting pose
    root = Rotation.from_quat([x, y, z, w])
    for p in parts:
        if p.beads is not None:
            for pos in p.beads["positions"]:
                qpos += list(root.apply(pos) + np.asarray(root_pos)) + [w, x, y, z]
    return qpos


def scene_xml(name: str, parts: list[Part], materials: MaterialTable) -> str:
    """Preview stage sized to the asset: floor, lights, and per-material-pair floor friction."""
    lo = np.min([p.visual.bounds[0] for p in parts], axis=0)
    hi = np.max([p.visual.bounds[1] for p in parts], axis=0)
    size = (hi - lo).max()
    center = (lo + hi) / 2
    drop_height = size * 2

    # MuJoCo's euler="25 35 0" default is intrinsic xyz, which scipy spells "XYZ"
    x, y, z, w = Rotation.from_euler("XYZ", [25, 35, 0], degrees=True).as_quat()
    drop = default_qpos(parts, [0, 0, drop_height], [w, x, y, z])

    # MuJoCo combines friction by taking the larger of the two geoms; explicit pairs let
    # every collision geom use the coefficient for its material against the floor's
    excludes = [f'    <exclude body1="world" body2="{p.name}"/>' for p in parts]
    pairs = []
    for p in parts:
        for geom, material in zip(collision_geom_names(p), p.collision_materials()):
            mu = materials.friction(material, FLOOR)
            pairs.append(
                f'    <pair geom1="floor" geom2="{geom}" condim="3" solref="{SOLREF}"'
                f' friction="{mu} {mu} {TORSIONAL} {ROLLING} {ROLLING}"/>'
            )
    tile = size / 3
    key_light = center + np.array([-1.2, -1.6, 2.8]) * size
    fill_light = center + np.array([1.8, 1.2, 1.5]) * size
    return f"""<mujoco model="{name}_scene">
  <include file="model.xml"/>
  <!-- elliptic cones: with the default pyramid, the bolt crept 1.4 mm down a slope it should grip -->
  <option timestep="0.001" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.25 0.25 0.25" specular="0.05 0.05 0.05"/>
    <map znear="0.005"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.78 0.82 0.86" rgb2="0.3 0.33 0.38" width="512" height="512"/>
    <texture name="floor_grid" type="2d" builtin="checker" rgb1="0.62 0.6 0.56" rgb2="0.54 0.52 0.48"
             width="512" height="512" mark="edge" markrgb="0.48 0.46 0.42"/>
    <material name="floor_mat" texture="floor_grid" texrepeat="{1 / (2 * tile):.4g} {1 / (2 * tile):.4g}"
              texuniform="true" reflectance="0.05"/>
  </asset>
  <worldbody>
    <!-- a spotlight sized to the object keeps its shadow map sharp at any scale -->
    <light name="key" pos="{_vec(key_light)}" dir="{_vec(center - key_light)}" cutoff="35" exponent="5"
           diffuse="0.75 0.72 0.68" specular="0.3 0.3 0.3" castshadow="true"/>
    <light name="fill" pos="{_vec(fill_light)}" dir="{_vec(center - fill_light)}" cutoff="60"
           diffuse="0.25 0.27 0.3" specular="0 0 0" castshadow="false"/>
    <geom name="floor" type="plane" size="{max(10, 20 * size):.3g} {max(10, 20 * size):.3g} 0.1" material="floor_mat"/>
  </worldbody>
  <contact>
{chr(10).join(excludes + pairs)}
  </contact>
  <keyframe>
    <key name="drop" qpos="{_vec(drop)}"/>
  </keyframe>
</mujoco>
"""
