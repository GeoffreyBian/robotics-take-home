"""Checks run against the exported package, so they test exactly what a user would load.

Tolerances are relative to the object's longest side so one set works from bolts to tables.
"""

import os
import re

import mujoco
import numpy as np
import trimesh

from . import grasp
from .asset import Part
from .materials import FLOOR, MaterialTable
from .mjcf import collision_geom_names

COVERAGE_TOL = 0.01  # surface points this close outside a hull still count as covered
MIN_COVERAGE = 0.95
MAX_PENETRATION = 0.01  # once resting
MAX_IMPACT_PENETRATION = 0.05  # peak while landing; soft contacts always give a little
MAX_REST_DRIFT = 0.02
# reconstructed poses aren't always exactly stable (the bolt rolls ~7 deg onto its head),
# so this only catches tipping over, e.g. a chair whose collision base doesn't support it
MAX_REST_TILT_DEG = 15.0
MAX_ENERGY_GAIN = 1.05  # total energy may not exceed its starting value by more than 5%
SETTLE_SPEED = 0.05  # longest sides per second
SETTLE_SPIN = 0.5  # rad/s
PROBE_RADIUS = 0.04
# rays: a hit counts as agreeing if collision and visual first hits are this close
RAY_TOL = 0.03
RAY_GRID = 8
# convex parts always round off some detail (handle cut-outs, chamfers, thread roots), so a
# perfect score isn't expected; 85% still fails a single hull on anything with real gaps
MIN_RAY_AGREEMENT = 0.85
SLIDE_MARGIN_DEG = 4.0
SLIDE_TIME = 2.0
HELD = 0.01  # moved less than 1% of its size: held


def _visual(parts: list[Part]) -> trimesh.Trimesh:
    return trimesh.util.concatenate(
        [trimesh.Trimesh(p.visual.vertices, p.visual.faces, process=False) for p in parts]
    )


def _hull_planes(hull):
    return hull.face_normals, np.einsum("ij,ij->i", hull.face_normals, hull.vertices[hull.faces[:, 0]])


def coverage(parts: list[Part], longest: float, n=4000) -> float:
    """Share of the visual surface inside the collision geometry. Low = missing collision."""
    points, _ = trimesh.sample.sample_surface(_visual(parts), n, seed=0)
    tol = COVERAGE_TOL * longest
    inside = np.zeros(len(points), dtype=bool)
    for part in parts:
        for hull in part.hulls:
            normals, offsets = _hull_planes(hull)
            inside |= ((points @ normals.T - offsets) <= tol).all(axis=1)
        if part.sphere is not None:
            center, radius = part.sphere
            inside |= np.linalg.norm(points - center, axis=1) <= radius + tol
    return float(inside.mean())


def _penetration(data, geoms=None) -> float:
    dist = data.contact.dist[: data.ncon]
    if geoms is not None and len(dist):
        g = data.contact.geom[: data.ncon]
        dist = dist[np.isin(g[:, 0], geoms) | np.isin(g[:, 1], geoms)]
    return float(max(0.0, -dist.min())) if len(dist) else 0.0


def _tilt_deg(q0, q1) -> float:
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(q0, q1)), 0, 1))))


def rest_test(scene_path: str, longest: float, duration: float = 2.0) -> dict:
    """Start in the reconstructed pose on the floor; it should stay put.

    The root free joint is the model's first joint, so qpos[:7] is the root pose.
    """
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q0 = data.qpos[:7].copy()
    penetration = 0.0
    while data.time < duration:
        mujoco.mj_step(model, data)
        if data.time > duration / 2:  # ignore the initial contact transient
            penetration = max(penetration, _penetration(data))

    result = {
        "drift": float(np.linalg.norm(data.qpos[:2] - q0[:2]) / longest),
        "tilt_deg": _tilt_deg(q0[3:7], data.qpos[3:7]),
        "penetration": penetration / longest,
    }
    result["passed"] = (
        result["drift"] < MAX_REST_DRIFT
        and result["tilt_deg"] < MAX_REST_TILT_DEG
        and result["penetration"] < MAX_PENETRATION
    )
    return result


def drop_test(scene_path: str, longest: float, duration: float = 8.0) -> dict:
    """Drop from the 'drop' keyframe; it should settle without gaining energy or sinking.

    8 s because the chair keeps rocking on its base for several seconds after landing.
    """
    model = mujoco.MjModel.from_xml_path(scene_path)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("drop").id)
    mujoco.mj_forward(model, data)
    start_energy = data.energy.sum()
    max_energy, impact, resting = start_energy, 0.0, 0.0
    while data.time < duration:
        mujoco.mj_step(model, data)
        max_energy = max(max_energy, data.energy.sum())
        impact = max(impact, _penetration(data))
        if data.time > duration - 1:
            resting = max(resting, _penetration(data))

    result = {
        "energy_gain": float(max_energy / start_energy),
        "final_speed": float(np.linalg.norm(data.qvel[:3]) / longest),
        "final_spin": float(np.linalg.norm(data.qvel[3:6])),
        "impact_penetration": impact / longest,
        "penetration": resting / longest,
    }
    result["passed"] = (
        result["energy_gain"] < MAX_ENERGY_GAIN
        and result["final_speed"] < SETTLE_SPEED
        and result["final_spin"] < SETTLE_SPIN
        and result["impact_penetration"] < MAX_IMPACT_PENETRATION
        and result["penetration"] < MAX_PENETRATION
    )
    return result


def welded_model_xml(package_dir: str) -> str:
    """model.xml with absolute asset paths and the root weld instead of a free joint."""
    with open(os.path.join(package_dir, "model.xml")) as f:
        xml = f.read()
    mesh_dir = os.path.abspath(os.path.join(package_dir, "meshes"))
    xml = xml.replace('meshdir="meshes" texturedir="meshes"', f'meshdir="{mesh_dir}" texturedir="{mesh_dir}"')
    return re.sub(r"<freejoint[^>]*/>", "", xml)


def probe_test(package_dir: str, parts: list[Part], longest: float, duration: float = 2.0) -> dict:
    """Drop spheres onto the fixed asset and check they end up touching the visual surface.

    gap = distance from the probe's center to the nearest real surface (visual mesh or
    floor) minus its radius. > 0 means it's held up by collision geometry that isn't really
    there (e.g. a convex hull lid over a crate); < 0 means it sank in. Nearest-surface rather
    than a ray straight down, since a probe leaning on a slanted rod has nothing directly below.
    """
    visual = _visual(parts)
    radius = PROBE_RADIUS * longest
    lo, hi = visual.bounds
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    offsets = np.array([[0, 0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    xy = mid[:2] + offsets * half[:2]
    centers = np.column_stack([xy, np.full(len(xy), hi[2] + 2 * radius)])

    # condim 6 adds rolling friction so probes stop where they land instead of rolling off
    probes = "\n".join(
        f'<body name="probe{i}" pos="{x} {y} {z}"><freejoint/>'
        f'<geom type="sphere" size="{radius}" density="500" condim="6" friction="0.8 0.05 0.05"/></body>'
        for i, (x, y, z) in enumerate(centers)
    )
    floor = '<geom type="plane" size="10 10 0.1"/>'
    xml = welded_model_xml(package_dir).replace("</worldbody>", f"{floor}\n{probes}\n</worldbody>")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    while data.time < duration:
        mujoco.mj_step(model, data)
    final = np.array([data.xpos[model.body(f"probe{i}").id] for i in range(len(centers))])

    _, to_mesh, _ = trimesh.proximity.closest_point(visual, final)
    gaps = (np.minimum(to_mesh, final[:, 2]) - radius) / longest
    result = {"gaps": gaps.round(4).tolist(), "max_abs_gap": float(np.abs(gaps).max())}
    # one probe radius of slack absorbs contact against side walls and slopes
    result["passed"] = result["max_abs_gap"] < PROBE_RADIUS
    return result


def _visual_first_hits(visual: trimesh.Trimesh, origins, directions) -> np.ndarray:
    """Distance to the first visual hit per ray, inf on a miss."""
    locations, ray_ids, _ = visual.ray.intersects_location(origins, directions, multiple_hits=True)
    first = np.full(len(origins), np.inf)
    if len(ray_ids):
        dist = np.einsum("ij,ij->i", locations - origins[ray_ids], directions[ray_ids])
        np.minimum.at(first, ray_ids, dist)
    return first


def ray_test(scene_path: str, parts: list[Part], longest: float) -> dict:
    """Cast rays at the asset from four sides and from above, as a gripper or arm would reach.

    Each ray's first hit on the collision geometry is compared with its first hit on the
    visual mesh. 'phantom' = collision stops the ray early (a hull filling the gap under a
    table); 'leak' = collision lets it through where the real surface is.
    """
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # qpos0: every part where the visual mesh is
    visual = _visual(parts)
    lo, hi = visual.bounds
    pad = 0.1 * longest
    tol = RAY_TOL * longest
    s = np.linspace(0.04, 0.96, RAY_GRID)  # stay just inside the bounding box

    rays = {}
    heights = lo[2] + s * (hi[2] - lo[2])
    for axis, sign, label in ((0, 1, "+x"), (0, -1, "-x"), (1, 1, "+y"), (1, -1, "-y")):
        other = 1 - axis
        lateral = lo[other] + s * (hi[other] - lo[other])
        h, l = np.meshgrid(heights, lateral, indexing="ij")
        origins = np.zeros((h.size, 3))
        origins[:, 2] = h.ravel()
        origins[:, other] = l.ravel()
        # start outside on the +sign side and shoot back across the object
        origins[:, axis] = (hi if sign > 0 else lo)[axis] + sign * pad
        direction = np.zeros(3)
        direction[axis] = -sign
        rays[label] = (origins, np.tile(direction, (len(origins), 1)))
    gx, gy = np.meshgrid(lo[0] + s * (hi[0] - lo[0]), lo[1] + s * (hi[1] - lo[1]), indexing="ij")
    origins = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, hi[2] + pad)])
    rays["top"] = (origins, np.tile([0, 0, -1.0], (len(origins), 1)))

    groups = np.zeros(6, dtype=np.uint8)
    groups[3] = 1  # collision geoms only; the floor is group 0 and the visual group 2
    geomid = np.zeros(1, dtype=np.int32)
    result = {"by_direction": {}}
    totals = {"agree": 0, "phantom": 0, "leak": 0}
    for label, (origins, directions) in rays.items():
        visual_d = _visual_first_hits(visual, origins, directions)
        counts = {"agree": 0, "phantom": 0, "leak": 0}
        for o, d, v in zip(origins, directions, visual_d):
            c = mujoco.mj_ray(model, data, o, d, groups, 1, -1, geomid)
            c = np.inf if c < 0 else c
            if np.isinf(c) and np.isinf(v):
                counts["agree"] += 1
            elif c < v - tol:
                counts["phantom"] += 1
            elif c > v + tol:
                counts["leak"] += 1
            else:
                counts["agree"] += 1
        n = len(origins)
        result["by_direction"][label] = round(counts["agree"] / n, 3)
        for k in totals:
            totals[k] += counts[k]
    n = sum(totals.values())
    result.update({k: round(v / n, 3) for k, v in totals.items()})
    result["rays"] = n
    result["passed"] = result["agree"] >= MIN_RAY_AGREEMENT
    return result


def _geom_materials(model, parts: list[Part]) -> dict[int, str]:
    out = {}
    for part in parts:
        for name, material in zip(collision_geom_names(part), part.collision_materials()):
            out[model.geom(name).id] = material
    return out


def slide_test(scene_path: str, parts: list[Part], materials: MaterialTable, longest: float) -> dict:
    """Tilt gravity just below and just above the friction angle atan(mu).

    It should hold below and move above. Tilting gravity instead of the floor keeps the
    scene's floor pairs untouched. Chairs on ball-jointed casters roll at any angle.
    """
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    while data.time < 1.5:  # settle onto the floor first
        mujoco.mj_step(model, data)
    floor = model.geom("floor").id
    geom_material = _geom_materials(model, parts)
    touching = set()
    for c in data.contact[: data.ncon]:
        if floor in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == floor else c.geom1
            if other in geom_material:
                touching.add(geom_material[other])
    if not touching:
        return {"passed": False, "reason": "not resting on the floor"}
    # the weakest contact material decides when it starts to slide
    mu = min(materials.friction(m, FLOOR) for m in touching)
    rolls = bool((model.jnt_type == mujoco.mjtJoint.mjJNT_BALL).any())
    settled_qpos, settled_qvel = data.qpos.copy(), data.qvel.copy()
    half_min = np.min([p.visual.extents for p in parts], axis=0).min() / 2

    outcomes = {}
    friction_angle = np.degrees(np.arctan(mu))
    for label, angle in (("below", friction_angle - SLIDE_MARGIN_DEG), ("above", friction_angle + SLIDE_MARGIN_DEG)):
        theta = np.radians(angle)
        model.opt.gravity[:] = 9.81 * np.array([np.sin(theta), 0.0, -np.cos(theta)])
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:] = settled_qpos, settled_qvel
        mujoco.mj_forward(model, data)
        q0 = data.qpos[:7].copy()
        while data.time < SLIDE_TIME:
            mujoco.mj_step(model, data)
        moved = np.linalg.norm(data.qpos[:3] - q0[:3])
        turned = np.radians(_tilt_deg(q0[3:7], data.qpos[3:7]))
        if moved < HELD * longest and turned < np.radians(5):
            outcome = "held"
        elif rolls:
            outcome = "rolled"
        elif turned < np.radians(10):
            outcome = "slid"
        # rolling covers distance ~ angle x radius; tipping turns without going far
        elif moved > 0.5 * turned * half_min:
            outcome = "rolled"
        else:
            outcome = "tipped"
        outcomes[label] = {"angle_deg": round(float(angle), 1), "outcome": outcome,
                           "moved": round(float(moved / longest), 3)}
    expected_below = ("held", "rolled") if rolls else ("held",)
    return {
        "mu": mu,
        "materials": sorted(touching),
        **outcomes,
        "passed": outcomes["below"]["outcome"] in expected_below and outcomes["above"]["outcome"] != "held",
    }


def export_test(package_dir: str, parts: list[Part]) -> dict:
    """The URDF and USD carry the same masses and joints as the MJCF."""
    ours = sum(p.mass.mass for p in parts)
    result = {"mass_kg": ours}

    # filenames are package-relative (meshes/...), so give MuJoCo the package as meshdir
    with open(os.path.join(package_dir, "model.urdf")) as f:
        text = f.read()
    mesh_dir = os.path.abspath(package_dir)
    text = re.sub(r"(<robot[^>]*>)", rf'\1\n  <mujoco><compiler meshdir="{mesh_dir}" strippath="false" fusestatic="false" discardvisual="true"/></mujoco>',
                  text, count=1)
    try:
        model = mujoco.MjModel.from_xml_string(text)
        names = [model.body(i).name for i in range(1, model.nbody)]
        urdf_mass = float(sum(model.body_mass[i + 1] for i, n in enumerate(names) if not n.endswith("_swivel")))
        result["urdf_mass_kg"] = urdf_mass
        result["urdf_ok"] = abs(urdf_mass - ours) <= 0.01 * ours
    except Exception as e:
        result["urdf_ok"], result["urdf_error"] = False, str(e)[:200]

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(os.path.join(package_dir, "model.usda"))
    usd_mass, joints, bodies = 0.0, 0, 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI):
            usd_mass += UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies += 1
        if prim.IsA(UsdPhysics.Joint):
            joints += 1
    result.update({"usd_mass_kg": usd_mass, "usd_bodies": bodies, "usd_joints": joints})
    result["usd_ok"] = (abs(usd_mass - ours) <= 0.01 * ours and bodies == len(parts) and joints == len(parts) - 1)
    result["passed"] = result["urdf_ok"] and result["usd_ok"]
    return result


def run(package_dir: str, parts: list[Part], prior, materials: MaterialTable) -> dict:
    scene = os.path.join(package_dir, "scene.xml")
    longest = prior.longest_m
    result = {
        "coverage": coverage(parts, longest),
        "rest": rest_test(scene, longest),
        "drop": drop_test(scene, longest),
        "probe": probe_test(package_dir, parts, longest),
        "rays": ray_test(scene, parts, longest),
        "slide": slide_test(scene, parts, materials, longest),
        "grasp": grasp.grasp_test(package_dir, parts, materials, longest),
        "export": export_test(package_dir, parts),
    }
    checks = [result["coverage"] >= MIN_COVERAGE] + [
        result[k]["passed"] for k in ("rest", "drop", "probe", "rays", "slide", "grasp", "export")
    ]
    result["passed"] = all(checks)
    return result
