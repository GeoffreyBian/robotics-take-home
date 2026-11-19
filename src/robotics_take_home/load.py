import numpy as np
import trimesh

# components smaller than this share of the faces are treated as reconstruction debris
DEBRIS_FACE_FRACTION = 0.01


def load_mesh(filepath: str) -> trimesh.Trimesh:
    mesh = trimesh.load(filepath, force="mesh")
    # glbs split UV seams; merge to make watertight
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    # glbs have Y as up-axis; convert to Z as up-axis as Mujoco expects
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return mesh


def repair(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    """Return a geometry-only copy that is watertight if we can make it so.

    The visual mesh is left alone; this copy only feeds mass properties and collision.
    """
    notes = []
    fixed = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=True)
    if fixed.is_watertight:
        return fixed, notes

    trimesh.repair.fill_holes(fixed)
    trimesh.repair.fix_normals(fixed)
    if fixed.is_watertight:
        notes.append("filled holes")
        return fixed, notes

    parts = fixed.split(only_watertight=False)
    min_faces = DEBRIS_FACE_FRACTION * len(fixed.faces)
    kept = [p for p in parts if p.is_watertight]
    dropped = [p for p in parts if not p.is_watertight]
    if kept and all(len(p.faces) < min_faces for p in dropped):
        notes.append(f"dropped {len(dropped)} open debris component(s)")
        return trimesh.util.concatenate(kept), notes

    notes.append("could not make watertight")
    return fixed, notes
