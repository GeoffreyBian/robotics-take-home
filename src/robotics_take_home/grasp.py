"""Grasp-and-lift with a teleported parallel gripper, as a robot would actually use the asset.

The gripper is two rubber pads on slide joints under a palm on a vertical slide (2F-140-like:
14 cm stroke, ~50 N per finger). It is placed open around the grasp site, closes, lifts and
holds. Strategies: 'body' squeezes across the object's narrow side at about COM height;
'rim' pinches the wall of an open container. Anything else is skipped with a reason.
"""

import os

import mujoco
import numpy as np
import trimesh

from .asset import Part
from .materials import GRIPPER_PAD, MaterialTable
from .mjcf import SOLREF, collision_geom_names

MAX_WIDTH = 0.14
MAX_MASS = 3.0
GRIP_FORCE = 50.0  # N per finger
PAD_THICKNESS = 0.01
PAD_HALF_WIDTH = 0.01
PAD_HALF_HEIGHT = 0.015
CLEARANCE = 0.01  # how far outside the surface the open pads start
CONTAINER_DEPTH = 0.5  # a center ray from above falling this share of the height: open container
START_OVERLAP = 0.002  # m; pads starting deeper than this inside collision geometry


def _visual(parts):
    return trimesh.util.concatenate(
        [trimesh.Trimesh(p.visual.vertices, p.visual.faces, process=False) for p in parts]
    )


def _first_hit(mesh, origin, direction) -> float:
    locations, _, _ = mesh.ray.intersects_location([origin], [direction], multiple_hits=True)
    if not len(locations):
        return np.inf
    return float(np.min((locations - origin) @ np.asarray(direction, float)))


def _section_bounds(mesh, z):
    s = mesh.section([0, 0, 1], [0, 0, z])
    if s is None:
        return None
    pts = s.vertices[:, :2]
    return pts.min(axis=0), pts.max(axis=0)


def plan(parts: list[Part], longest: float) -> dict:
    """Where to grasp: center, closing axis, opening and pad height, or a skip reason."""
    mass = sum(p.mass.mass for p in parts)
    if mass > MAX_MASS:
        return {"strategy": "skipped", "reason": f"{mass:.1f} kg is too heavy for a parallel gripper"}
    visual = _visual(parts)
    lo, hi = visual.bounds
    height = hi[2] - lo[2]
    com = sum(p.mass.mass * p.mass.center_of_mass for p in parts) / mass
    pad_h = min(PAD_HALF_HEIGHT, 0.4 * height)

    # body grasp: widest the object gets over the pad's height, around the COM
    z = float(np.clip(com[2], lo[2] + pad_h + 0.003, hi[2] - pad_h))
    slabs = [b for b in (_section_bounds(visual, zz) for zz in (z - pad_h, z, z + pad_h)) if b is not None]
    if slabs:
        s_lo = np.min([b[0] for b in slabs], axis=0)
        s_hi = np.max([b[1] for b in slabs], axis=0)
        widths = s_hi - s_lo
        axis = int(np.argmin(widths))
        if widths[axis] <= MAX_WIDTH:
            center = np.array([*(s_lo + s_hi) / 2, z])
            return {"strategy": "body", "center": center, "axis": axis,
                    "open": widths[axis] / 2 + CLEARANCE, "pad_h": pad_h, "mass": mass}

    # rim grasp: pinch the wall of an open container, on the side facing -axis.
    # measured on the root part (the box), so an open lid behind it doesn't shift the center
    root = parts[0].visual
    mid = root.bounds.mean(axis=0)
    lo = root.bounds[0]
    depth = _first_hit(visual, [mid[0], mid[1], hi[2] + 0.1], [0, 0, -1]) - (hi[2] + 0.1 - root.bounds[1][2])
    if depth > CONTAINER_DEPTH * (root.bounds[1][2] - root.bounds[0][2]):
        axis = int(np.argmin(root.extents[:2]))  # the long walls face the short axis
        z = root.bounds[1][2] - pad_h - 0.005
        outside = mid.copy()
        outside[2] = z
        outside[axis] = lo[axis] - 0.1
        inward = np.zeros(3)
        inward[axis] = 1.0
        outer = outside[axis] + _first_hit(visual, outside, inward)
        inner_origin = np.array([mid[0], mid[1], z])
        inner = inner_origin[axis] - _first_hit(visual, inner_origin, -inward)
        if np.isfinite(outer) and np.isfinite(inner) and 0 < inner - outer <= MAX_WIDTH:
            center = inner_origin.copy()
            center[axis] = (outer + inner) / 2
            return {"strategy": "rim", "center": center, "axis": axis,
                    "open": (inner - outer) / 2 + CLEARANCE, "pad_h": pad_h, "mass": mass}
        return {"strategy": "skipped", "reason": "could not find a pinchable wall"}
    return {"strategy": "skipped", "reason": "wider than the gripper and not an open container"}


def _scene(package_dir, parts, materials, g, palm_height, lift) -> str:
    with open(os.path.join(package_dir, "model.xml")) as f:
        xml = f.read()
    mesh_dir = os.path.abspath(os.path.join(package_dir, "meshes"))
    xml = xml.replace('meshdir="meshes" texturedir="meshes"', f'meshdir="{mesh_dir}" texturedir="{mesh_dir}"')

    axis = np.zeros(3)
    axis[g["axis"]] = 1.0
    pad_size = np.full(3, PAD_HALF_WIDTH)
    pad_size[g["axis"]] = PAD_THICKNESS / 2
    pad_size[2] = g["pad_h"]
    offset = g["open"] + PAD_THICKNESS / 2
    fingers = []
    for side, sign in (("l", -1), ("r", 1)):
        pos = sign * offset * axis + np.array([0, 0, -palm_height])
        fingers.append(
            f'<body name="finger_{side}" pos="{pos[0]} {pos[1]} {pos[2]}" gravcomp="1">'
            f'<joint name="slide_{side}" type="slide" axis="{-sign * axis[0]} {-sign * axis[1]} 0"'
            f' range="0 {g["open"]}" damping="20" armature="0.01"/>'
            f'<geom name="pad_{side}" type="box" size="{pad_size[0]} {pad_size[1]} {pad_size[2]}" mass="0.05"'
            f' rgba="0.15 0.15 0.15 1"/></body>'
        )
    c = g["center"]
    # a jointed palm, not a mocap body: mocap bodies are teleported with zero velocity, so
    # friction (a velocity constraint) never sees the pads moving and they slide off the object
    gripper = (f'<body name="palm" pos="{c[0]} {c[1]} {c[2] + palm_height}" gravcomp="1">'
               f'<joint name="lift" type="slide" axis="0 0 1" range="-0.01 {lift + 0.01}" damping="50"/>'
               f'<inertial pos="0 0 0" mass="1" diaginertia="0.001 0.001 0.001"/>'
               + "".join(fingers) + "</body>")
    floor = '<geom name="floor" type="plane" size="10 10 0.1"/>'

    excludes = ['<exclude body1="world" body2="finger_l"/>', '<exclude body1="world" body2="finger_r"/>',
                '<exclude body1="finger_l" body2="finger_r"/>', '<exclude body1="palm" body2="finger_l"/>',
                '<exclude body1="palm" body2="finger_r"/>']
    pairs = []
    for p in parts:
        excludes += [f'<exclude body1="finger_{s}" body2="{p.name}"/>' for s in "lr"]
        for geom, material in zip(collision_geom_names(p), p.collision_materials()):
            mu = materials.friction(GRIPPER_PAD, material)
            pairs += [f'<pair geom1="pad_{s}" geom2="{geom}" condim="4" solref="{SOLREF}"'
                      f' friction="{mu} {mu} 0.01 0.0001 0.0001"/>' for s in "lr"]
    kp = GRIP_FORCE / 0.005  # saturates at the force limit 5 mm before the target
    actuators = f'<position name="lift" joint="lift" kp="20000" ctrlrange="-0.01 {lift + 0.01}"/>' + "".join(
        f'<position name="grip_{s}" joint="slide_{s}" kp="{kp}" ctrlrange="0 {g["open"]}"'
        f' forcerange="{-GRIP_FORCE} {GRIP_FORCE}"/>' for s in "lr"
    )
    xml = xml.replace("</worldbody>", f"{floor}\n{gripper}\n</worldbody>")
    xml = xml.replace("</mujoco>", f'<option timestep="0.001" cone="elliptic" impratio="10"/><contact>{"".join(excludes + pairs)}</contact>'
                                   f"<actuator>{actuators}</actuator></mujoco>")
    return xml


def grasp_test(package_dir: str, parts: list[Part], materials: MaterialTable, longest: float) -> dict:
    g = plan(parts, longest)
    if g["strategy"] == "skipped":
        return {"strategy": "skipped", "reason": g["reason"], "passed": True}
    palm_height = 0.1
    lift = longest + 0.05
    model = mujoco.MjModel.from_xml_string(_scene(package_dir, parts, materials, g, palm_height, lift))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    pads = [model.geom("pad_l").id, model.geom("pad_r").id]
    floor = model.geom("floor").id
    object_geoms = [model.geom(n).id for p in parts for n in collision_geom_names(p)]
    result = {"strategy": g["strategy"], "width_m": round(2 * (g["open"] - CLEARANCE), 4),
              "mass_kg": round(g["mass"], 3)}

    def touching(geom_a, geoms_b, min_dist=None):
        for c in data.contact[: data.ncon]:
            pair = {c.geom1, c.geom2}
            if geom_a in pair and pair & set(geoms_b) and (min_dist is None or c.dist < min_dist):
                return True
        return False

    # a pad that starts inside the collision shape means the gripper can't get there at all
    if any(touching(p, object_geoms, -START_OVERLAP) for p in pads):
        result.update({"outcome": "pads start inside collision geometry", "passed": False})
        return result

    root = model.body(parts[0].name).id
    start_root_z = float(data.xpos[root][2])
    palm = model.body("palm").id
    phases = [("settle", 0.5), ("close", 1.0), ("lift", 1.5), ("hold", 1.0)]
    t, rel_start = 0.0, None
    for phase, duration in phases:
        end = t + duration
        while data.time < end:
            if phase != "settle":
                data.ctrl[1:] = g["open"]
            if phase == "lift":
                data.ctrl[0] = lift * min(1.0, (data.time - t) / duration)
            mujoco.mj_step(model, data)
        if phase == "lift":
            rel_start = data.xpos[root] - data.xpos[palm]
        t = end

    held = all(touching(p, object_geoms) for p in pads)
    on_floor = touching(floor, object_geoms)
    slip = float(np.linalg.norm((data.xpos[root] - data.xpos[palm]) - rel_start))
    result.update({
        "lifted_m": round(float(data.xpos[root][2]) - start_root_z, 3),
        "slip_during_hold_m": round(slip, 4),
        "outcome": "lifted" if held and not on_floor else ("dropped" if not held else "still on floor"),
    })
    result["passed"] = result["outcome"] == "lifted"
    return result
