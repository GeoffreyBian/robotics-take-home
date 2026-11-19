"""URDF export for PyBullet, Isaac, Drake, ROS. The root link is free-floating by convention.

URDF has no ball joint and no joint springs, so a caster's ball becomes two continuous
joints (swivel then roll) through a tiny dummy link, and hold-pose springs are dropped.
"""

import numpy as np

from .asset import Part, tree_order
from .materials import MaterialTable
from .mjcf import collision_geom_names


def _vec(values) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _inertial(mass: float, com, inertia) -> str:
    I = inertia
    return (
        f'    <inertial>\n      <origin xyz="{_vec(com)}" rpy="0 0 0"/>\n      <mass value="{mass:.6g}"/>\n'
        f'      <inertia ixx="{I[0, 0]:.6g}" iyy="{I[1, 1]:.6g}" izz="{I[2, 2]:.6g}"'
        f' ixy="{I[0, 1]:.6g}" ixz="{I[0, 2]:.6g}" iyz="{I[1, 2]:.6g}"/>\n    </inertial>'
    )


def _link(part: Part, materials: MaterialTable) -> str:
    origin = part.body_origin()
    m = part.mass
    lines = [f'  <link name="{part.name}">', _inertial(m.mass, m.center_of_mass - origin, m.inertia)]
    lines.append(
        f'    <visual>\n      <origin xyz="{_vec(-origin)}" rpy="0 0 0"/>\n'
        f'      <geometry><mesh filename="meshes/{part.name}_visual.obj"/></geometry>\n    </visual>'
    )
    for geom in collision_geom_names(part):
        if part.sphere is not None:
            center, radius = part.sphere
            shape, xyz = f'<sphere radius="{radius:.6g}"/>', center - origin
        else:
            shape, xyz = f'<mesh filename="meshes/{geom}.obj"/>', -origin
        lines.append(
            f'    <collision name="{geom}">\n      <origin xyz="{_vec(xyz)}" rpy="0 0 0"/>\n'
            f'      <geometry>{shape}</geometry>\n    </collision>'
        )
    lines.append("  </link>")
    return "\n".join(lines)


def _joint(name, kind, parent, child, xyz, axis, damping, limits=None) -> str:
    limit = f'\n    <limit lower="{limits[0]:.6g}" upper="{limits[1]:.6g}" effort="1000" velocity="10"/>' if limits else ""
    return (
        f'  <joint name="{name}" type="{kind}">\n    <parent link="{parent}"/>\n    <child link="{child}"/>\n'
        f'    <origin xyz="{_vec(xyz)}" rpy="0 0 0"/>\n    <axis xyz="{_vec(axis)}"/>{limit}\n'
        f'    <dynamics damping="{damping:.4g}" friction="0"/>\n  </joint>'
    )


def urdf(name: str, parts: list[Part], materials: MaterialTable) -> str:
    parts = tree_order(parts)
    origins = {p.name: p.body_origin() for p in parts}
    blocks = [_link(p, materials) for p in parts]
    for p in parts[1:]:
        j = p.joint
        offset = origins[p.name] - origins[p.parent]
        if j.kind == "hinge":
            kind = "revolute" if j.range else "continuous"
            blocks.append(_joint(f"{p.name}_joint", kind, p.parent, p.name, offset, j.axis, j.damping, j.range))
        else:
            swivel = f"{p.name}_swivel"
            blocks.append(f'  <link name="{swivel}">\n{_inertial(1e-4, np.zeros(3), np.eye(3) * 1e-9)}\n  </link>')
            # a caster swivels about z then rolls; a liquid pendulum swings about x then y
            first, second = ([1, 0, 0], [0, 1, 0]) if p.liquid else ([0, 0, 1], [0, 1, 0])
            blocks.append(_joint(f"{p.name}_swivel_joint", "continuous", p.parent, swivel, offset, first, j.damping))
            blocks.append(_joint(f"{p.name}_roll_joint", "continuous", swivel, p.name, [0, 0, 0], second, j.damping))
    return f'<?xml version="1.0"?>\n<robot name="{name}">\n' + "\n".join(blocks) + "\n</robot>\n"
