"""Checks run against the exported package, so they test exactly what a user would load.

Tolerances are relative to the object's longest side so one set works from bolts to tables.
"""

import os
import re

import mujoco
import numpy as np
import trimesh

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


def coverage(visual: trimesh.Trimesh, hulls: list[trimesh.Trimesh], longest: float, n=4000) -> float:
    """Share of the visual surface inside the collision geometry. Low = missing collision."""
    points = visual.sample(n)
    tol = COVERAGE_TOL * longest
    inside = np.zeros(len(points), dtype=bool)
    for hull in hulls:
        offsets = np.einsum("ij,ij->i", hull.face_normals, hull.vertices[hull.faces[:, 0]])
        inside |= ((points @ hull.face_normals.T - offsets) <= tol).all(axis=1)
    return float(inside.mean())


def _penetration(data) -> float:
    dist = data.contact.dist[: data.ncon]
    return float(max(0.0, -dist.min())) if len(dist) else 0.0


def _tilt_deg(q0, q1) -> float:
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(q0, q1)), 0, 1))))


def rest_test(scene_path: str, longest: float, duration: float = 2.0) -> dict:
    """Start in the reconstructed pose on the floor; it should stay put."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q0 = data.qpos.copy()
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


def _probe_scene(package_dir: str, centers: np.ndarray, radius: float) -> str:
    """The asset welded to the world, with spheres hovering over it."""
    with open(os.path.join(package_dir, "model.xml")) as f:
        xml = f.read()
    mesh_dir = os.path.abspath(os.path.join(package_dir, "meshes"))
    xml = xml.replace('meshdir="meshes" texturedir="meshes"',
                      f'meshdir="{mesh_dir}" texturedir="{mesh_dir}"')
    xml = re.sub(r"<freejoint[^>]*/>", "", xml)
    # condim 6 adds rolling friction so probes stop where they land instead of rolling off
    probes = "\n".join(
        f'<body pos="{x} {y} {z}"><freejoint/>'
        f'<geom type="sphere" size="{radius}" density="500" condim="6" friction="0.8 0.05 0.05"/></body>'
        for x, y, z in centers
    )
    floor = '<geom type="plane" size="10 10 0.1"/>'
    return xml.replace("</worldbody>", f"{floor}\n{probes}\n</worldbody>")


def probe_test(package_dir: str, visual: trimesh.Trimesh, longest: float, duration: float = 2.0) -> dict:
    """Drop spheres onto the fixed asset and check they end up touching the visual surface.

    gap = distance from the probe's center to the nearest real surface (visual mesh or
    floor) minus its radius. > 0 means it's held up by collision geometry that isn't really
    there (e.g. a convex hull lid over a crate); < 0 means it sank in. Nearest-surface rather
    than a ray straight down, since a probe leaning on a slanted rod has nothing directly below.
    """
    radius = PROBE_RADIUS * longest
    lo, hi = visual.bounds
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    offsets = np.array([[0, 0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
    xy = mid[:2] + offsets * half[:2]
    centers = np.column_stack([xy, np.full(len(xy), hi[2] + 2 * radius)])

    model = mujoco.MjModel.from_xml_string(_probe_scene(package_dir, centers, radius))
    data = mujoco.MjData(model)
    while data.time < duration:
        mujoco.mj_step(model, data)
    final = data.qpos.reshape(-1, 7)[:, :3]

    _, to_mesh, _ = trimesh.proximity.closest_point(visual, final)
    gaps = (np.minimum(to_mesh, final[:, 2]) - radius) / longest

    result = {
        "gaps": gaps.round(4).tolist(),
        "max_abs_gap": float(np.abs(gaps).max()),
    }
    # one probe radius of slack absorbs contact against side walls and slopes
    result["passed"] = result["max_abs_gap"] < PROBE_RADIUS
    return result


def run(package_dir: str, visual: trimesh.Trimesh, hulls: list[trimesh.Trimesh], longest: float) -> dict:
    scene = os.path.join(package_dir, "scene.xml")
    result = {
        "coverage": coverage(visual, hulls, longest),
        "rest": rest_test(scene, longest),
        "drop": drop_test(scene, longest),
        "probe": probe_test(package_dir, visual, longest),
    }
    checks = [result["coverage"] >= MIN_COVERAGE] + [result[k]["passed"] for k in ("rest", "drop", "probe")]
    result["passed"] = all(checks)
    return result
