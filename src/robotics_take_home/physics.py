from dataclasses import dataclass

import numpy as np
import trimesh

from .priors import Prior


@dataclass
class MassProperties:
    mass: float
    center_of_mass: np.ndarray  # (3,) in the body frame
    inertia: np.ndarray  # (3, 3) about the center of mass
    volume: float
    density: float  # effective, i.e. mass / reconstructed volume
    source: str  # which geometry the distribution came from


def scale_factor(mesh: trimesh.Trimesh, prior: Prior) -> float:
    # reconstructions have no metric scale; anchor the longest side to the prior
    return prior.longest_m / mesh.extents.max()


def mass_properties(
    mesh: trimesh.Trimesh, hulls: list[trimesh.Trimesh], prior: Prior
) -> MassProperties:
    """Mass comes from the prior; the mesh only says how that mass is distributed."""
    if mesh.is_watertight and mesh.volume > 0:
        solid, source = mesh, "mesh"
    else:
        # no closed volume to integrate, so fall back to the convex parts;
        # overlapping parts get counted twice, which is acceptable for a fallback
        solid, source = trimesh.util.concatenate(hulls), "collision hulls"

    volume = solid.volume
    mass = prior.mass_kg if prior.mass_kg is not None else prior.density_kg_m3 * volume

    solid = solid.copy()
    solid.density = mass / volume
    props = MassProperties(
        mass=float(mass),
        center_of_mass=np.asarray(solid.center_mass),
        inertia=np.asarray(solid.moment_inertia),
        volume=float(volume),
        density=float(mass / volume),
        source=source,
    )
    check_inertia(props.inertia)
    return props


def check_inertia(inertia: np.ndarray):
    principal = np.linalg.eigvalsh(inertia)
    if (principal <= 0).any():
        raise ValueError(f"inertia not positive definite: {principal}")
    a, b, c = principal
    # any rigid body satisfies the triangle inequality; MuJoCo rejects ones that don't
    if a + b < c * (1 - 1e-6):
        raise ValueError(f"inertia violates triangle inequality: {principal}")
