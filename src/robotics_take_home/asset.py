"""An asset is a tree of rigid parts. Single-body assets are just one root part.

All geometry stays in the asset frame (metres, Z up, floor at z=0); exporters work out
body-relative offsets, so nothing upstream has to think about parent frames.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh


@dataclass
class Joint:
    kind: str  # "hinge" or "ball"
    anchor: np.ndarray  # asset frame
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    range: Optional[tuple[float, float]] = None  # radians, hinge only
    damping: float = 0.0
    stiffness: float = 0.0  # spring back to the reconstructed pose (springref 0)
    # set stiffness from the part's mass so it holds its reconstructed pose (creased card)
    hold_pose: bool = False
    # damping/stiffness already set by whoever built the joint; finalize_joints leaves it
    tuned: bool = False


@dataclass
class Part:
    name: str
    visual: trimesh.Trimesh
    solid: trimesh.Trimesh
    material: str
    parent: Optional[str] = None
    joint: Optional[Joint] = None
    # (center, radius): a sphere instead of mesh hulls, e.g. caster wheels that must roll
    sphere: Optional[tuple[np.ndarray, float]] = None
    hulls: list[trimesh.Trimesh] = field(default_factory=list)
    hull_materials: list[str] = field(default_factory=list)
    collision_info: dict = field(default_factory=dict)
    mass: object = None  # physics.MassProperties, filled in by the pipeline
    # a sloshing-liquid pendulum: carries mass but never touches anything
    liquid: bool = False
    # visual-only liquid beads in a box cavity: {"lo", "hi", "radius", "positions"}
    beads: Optional[dict] = None

    @property
    def collides(self) -> bool:
        return not self.liquid

    def body_origin(self) -> np.ndarray:
        # bodies sit at their joint anchor so the joint is at the body origin
        return self.joint.anchor if self.joint is not None else np.zeros(3)

    def collision_materials(self) -> list[str]:
        if self.sphere is not None:
            return [self.material]
        return self.hull_materials or [self.material] * len(self.hulls)


def face_subset(mesh: trimesh.Trimesh, mask: np.ndarray) -> trimesh.Trimesh:
    """Faces where mask is true, keeping UVs, material and vertex normals intact."""
    faces = mesh.faces[mask]
    used, inverse = np.unique(faces, return_inverse=True)
    visual = None
    if isinstance(mesh.visual, trimesh.visual.TextureVisuals) and mesh.visual.uv is not None:
        visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv[used], material=mesh.visual.material)
    return trimesh.Trimesh(
        vertices=mesh.vertices[used],
        faces=inverse.reshape(-1, 3),
        vertex_normals=mesh.vertex_normals[used],
        visual=visual,
        process=False,
    )


def halfspace(solid: trimesh.Trimesh, origin, normal) -> trimesh.Trimesh:
    """The closed part of a watertight solid on the side `normal` points to."""
    return trimesh.intersections.slice_mesh_plane(
        solid, plane_normal=np.asarray(normal, float), plane_origin=np.asarray(origin, float), cap=True
    )


def tree_order(parts: list[Part]) -> list[Part]:
    """Parents before children."""
    by_parent: dict[Optional[str], list[Part]] = {}
    for p in parts:
        by_parent.setdefault(p.parent, []).append(p)
    ordered, stack = [], list(reversed(by_parent.get(None, [])))
    while stack:
        p = stack.pop()
        ordered.append(p)
        stack.extend(reversed(by_parent.get(p.name, [])))
    if len(ordered) != len(parts) or len(by_parent.get(None, [])) != 1:
        raise ValueError("parts must form a single tree")
    return ordered
