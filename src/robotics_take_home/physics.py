"""Mass properties: the prior sets how much, the mass model sets where it sits.

Mass models (prior.mass_model["type"]):
  solid       spread through the closed volume (default; solid wood, steel)
  shell       spread over the surface (thin-walled boxes and crates whose reconstructed
              walls are far thicker than the real ones)
  components  separate closed bodies get different materials (piston vs steel rod)
  liquid      a container shell plus liquid filling the volume up to a height (bottle)
Multi-part assets split the mass by each part's material density x volume (or area).
"""

from dataclasses import dataclass

import numpy as np
import trimesh

from .asset import Joint, Part, halfspace
from .materials import MaterialTable
from .priors import Prior

SHELL_SAMPLES = 20_000
GRAVITY = 9.81


@dataclass
class MassProperties:
    mass: float
    center_of_mass: np.ndarray  # (3,) asset frame
    inertia: np.ndarray  # (3, 3) about the center of mass
    elements: list[str]  # what the mass is made of, for the report


def scale_factor(mesh: trimesh.Trimesh, prior: Prior) -> float:
    # reconstructions have no metric scale; anchor the longest side to the prior
    return prior.longest_m / mesh.extents.max()


def solid_element(mesh: trimesh.Trimesh, mass: float) -> tuple:
    mesh = mesh.copy()
    mesh.density = mass / mesh.volume
    return mass, np.asarray(mesh.center_mass), np.asarray(mesh.moment_inertia)


def shell_element(mesh: trimesh.Trimesh, mass: float) -> tuple:
    """Mass spread evenly over the surface, by sampling it (seeded, so reproducible)."""
    points, _ = trimesh.sample.sample_surface(mesh, SHELL_SAMPLES, seed=0)
    com = points.mean(axis=0)
    r = points - com
    m = mass / len(points)
    inertia = m * (np.eye(3) * (r * r).sum() - r.T @ r)
    return mass, com, inertia


def combine(elements: list[tuple], labels: list[str]) -> MassProperties:
    """Sum (mass, com, inertia-about-com) elements with the parallel axis theorem."""
    total = sum(e[0] for e in elements)
    com = sum(e[0] * e[1] for e in elements) / total
    inertia = np.zeros((3, 3))
    for m, c, i in elements:
        d = c - com
        inertia += i + m * (np.eye(3) * (d @ d) - np.outer(d, d))
    check_inertia(inertia)
    return MassProperties(mass=float(total), center_of_mass=com, inertia=inertia, elements=labels)


def _volume_source(part: Part) -> trimesh.Trimesh:
    if part.solid.is_watertight and part.solid.volume > 0:
        return part.solid
    # no closed volume to integrate, so fall back to the convex parts;
    # overlapping parts get counted twice, which is acceptable for a fallback
    return trimesh.util.concatenate(part.hulls)


def assign_masses(parts: list[Part], prior: Prior, materials: MaterialTable) -> dict:
    """Fill in part.mass for every part. Returns a summary for the report."""
    kind = prior.mass_model.get("type", "solid")
    if kind in ("components", "liquid") and len(parts) != 1:
        raise ValueError(f"{kind} mass model needs a single-part asset")
    handler = {"solid": _uniform, "shell": _uniform, "components": _components, "liquid": _liquid}[kind]
    summary = handler(parts, prior, materials, kind)
    for part in parts:
        if part.sphere is not None:
            # a wheel's mass sits on its axle; off-center, gravity spins the free ball joint
            # and the sphere rolls the whole chair across a flat floor
            part.mass.center_of_mass = np.asarray(part.sphere[0], float)
    summary["total_mass_kg"] = sum(p.mass.mass for p in parts)
    summary["model"] = kind
    return summary


def _uniform(parts, prior, materials, kind):
    shell = kind == "shell"
    sources = [_volume_source(p) for p in parts]
    # relative weight of each part: density x volume, or density x area for shells
    weights = np.array([
        materials[p.material].density * (s.area if shell else s.volume) for p, s in zip(parts, sources)
    ])
    if prior.mass_kg is not None:
        masses = prior.mass_kg * weights / weights.sum()
    elif shell:
        raise ValueError("a shell mass model needs mass_kg; wall thickness is unknown")
    else:
        masses = weights  # density x volume
    for part, source, m in zip(parts, sources, masses):
        element = shell_element(source, m) if shell else solid_element(source, m)
        part.mass = combine([element], [f"{part.material} {'shell' if shell else 'solid'} {m:.3g} kg"])
    return {"parts": {p.name: round(p.mass.mass, 4) for p in parts}}


def _components(parts, prior, materials, kind):
    part = parts[0]
    comps = [c for c in part.solid.split(only_watertight=True) if c.volume > 0]
    names = prior.mass_model["materials"]
    if len(comps) != len(names):
        raise ValueError(f"expected {len(names)} closed components, found {len(comps)}")
    if prior.mass_model.get("order", "centroid_z") == "centroid_z":
        comps.sort(key=lambda c: c.centroid[2])
    weights = np.array([materials[n].density * c.volume for n, c in zip(names, comps)])
    masses = weights / weights.sum() * (prior.mass_kg if prior.mass_kg is not None else weights.sum())
    part.mass = combine(
        [solid_element(c, m) for c, m in zip(comps, masses)],
        [f"{n} solid {m:.3g} kg" for n, m in zip(names, masses)],
    )
    # friction: each collision hull takes the material of the component containing its
    # center, or the nearest one when the center falls outside both (e.g. across a gap)
    if part.hulls:
        centers = np.array([h.centroid for h in part.hulls])
        inside = np.column_stack([c.contains(centers) for c in comps])
        dists = np.column_stack([np.abs(trimesh.proximity.signed_distance(c, centers)) for c in comps])
        choice = np.where(inside.any(axis=1), np.argmax(inside, axis=1), np.argmin(dists, axis=1))
        part.hull_materials = [names[i] for i in choice]
    return {"components": {n: round(float(m), 4) for n, m in zip(names, masses)}}


def _liquid(parts, prior, materials, kind):
    """Container shell plus liquid that moves under gravity, as a damped pendulum.

    The standard 'equivalent mechanical model' for sloshing: the liquid's mass hangs from
    the cavity's centroid at the length that puts it at the real liquid COM when upright.
    Tip the bottle and the mass swings to the low side; in free fall it floats; on impact
    it surges. One mode only, and the swing radius is fixed, so it is a first-order model.
    """
    part = parts[0]
    spec = prior.mass_model
    lo, hi = part.solid.bounds
    level = lo[2] + spec["fill_height_frac"] * (hi[2] - lo[2])
    liquid = halfspace(part.solid, [0, 0, level], [0, 0, -1])
    liquid_mass = materials[spec["liquid"]].density * liquid.volume
    container = prior.mass_kg if prior.mass_kg is not None else 0.05 * liquid_mass
    part.mass = combine([shell_element(part.solid, container)], [f"{part.material} shell {container:.3g} kg"])

    pivot = np.asarray(part.solid.center_mass)
    rest = np.asarray(liquid.center_mass)
    length = float(np.linalg.norm(pivot - rest))
    # a small ball rather than a point, so MuJoCo gets a well-conditioned inertia
    radius = 0.25 * min(liquid.extents[:2]) / 2
    # water sloshing in a small bottle decays over a few cycles: damping ratio ~0.1
    zeta = spec.get("damping_ratio", 0.1)
    damping = 2 * zeta * liquid_mass * length * np.sqrt(GRAVITY * length)

    # the visual is the liquid's own shape, shrunk a little so it pokes through the walls
    # less when it swings (it moves rigidly, it doesn't reshape)
    marker = liquid.copy()
    marker.apply_transform(trimesh.transformations.scale_matrix(0.92, rest))
    slosh = Part(
        name=f"{part.name}_liquid", visual=marker, solid=marker, material=spec["liquid"],
        parent=part.name, liquid=True,
        joint=Joint("ball", pivot, damping=float(damping), tuned=True),
    )
    slosh.mass = combine(
        [(liquid_mass, rest, np.eye(3) * 0.4 * liquid_mass * radius ** 2)],
        [f"{spec['liquid']} {liquid_mass:.3g} kg on a {length * 100:.1f} cm pendulum from z={pivot[2]:.3f} m"],
    )
    slosh.collision_info = {"parts": 0, "note": "liquid: mass only, no contacts"}
    slosh.beads = _beads(part.solid, liquid, level)
    parts.append(slosh)

    # how this single pendulum compares with the first slosh mode of an upright cylinder
    # of the same radius and fill (Abramson 1966): w^2 = g (1.841 / R) tanh(1.841 h / R)
    r, h = min(liquid.extents[:2]) / 2, liquid.extents[2]
    theory = np.sqrt(GRAVITY * 1.841 / r * np.tanh(1.841 * h / r))
    return {
        "container_kg": container,
        "liquid_kg": round(float(liquid_mass), 4),
        "liquid_litres": round(float(liquid.volume * 1000), 3),
        "pendulum_length_m": round(length, 4),
        "pendulum_hz": round(float(np.sqrt(GRAVITY / length) / (2 * np.pi)), 2),
        "first_slosh_mode_hz": round(float(theory / (2 * np.pi)), 2),
        "damping_ratio": zeta,
    }


BEAD_COUNT = 120


def _beads(container: trimesh.Trimesh, liquid: trimesh.Trimesh, level: float) -> dict:
    """Visual liquid: beads tumbling in a box that approximates the container's body cavity.

    The box spans the liquid's footprint and runs up to where the container narrows
    (the shoulder), so beads can't flow into the neck. Beads only collide with the box
    walls and each other, so they look like liquid without touching anything else.
    """
    lo, hi = liquid.bounds
    inset = 0.08 * min(liquid.extents[:2])
    width = min(liquid.extents[:2])
    top = level
    for z in np.linspace(level, container.bounds[1][2], 60):
        section = container.section([0, 0, 1], [0, 0, z])
        if section is None or np.ptp(np.vstack(section.discrete)[:, :2], axis=0).min() < 0.7 * width:
            break
        top = z
    box_lo = np.array([lo[0] + inset, lo[1] + inset, lo[2] + inset])
    box_hi = np.array([hi[0] - inset, hi[1] - inset, top - inset])
    # bead size so BEAD_COUNT of them at ~55% packing fill the liquid volume
    radius = float((0.55 * liquid.volume / (BEAD_COUNT * 4 / 3 * np.pi)) ** (1 / 3))
    step = 2.05 * radius
    axes = [np.arange(a + radius, b - radius, step) for a, b in zip(box_lo, box_hi)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    grid = grid[np.argsort(grid[:, 2], kind="stable")][:BEAD_COUNT]
    return {"lo": box_lo, "hi": box_hi, "radius": radius, "positions": grid}


def check_inertia(inertia: np.ndarray):
    principal = np.linalg.eigvalsh(inertia)
    if (principal <= 0).any():
        raise ValueError(f"inertia not positive definite: {principal}")
    a, b, c = principal
    # any rigid body satisfies the triangle inequality; MuJoCo rejects ones that don't
    if a + b < c * (1 - 1e-6):
        raise ValueError(f"inertia violates triangle inequality: {principal}")
