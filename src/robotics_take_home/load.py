import numpy as np
import trimesh

# components smaller than this share of the faces are treated as reconstruction debris
DEBRIS_FACE_FRACTION = 0.01
# visual meshes above this get decimated; physics never sees the visual mesh
VISUAL_MAX_FACES = 40_000

# glbs have Y as up-axis; convert to Z as up-axis as Mujoco expects
Y_UP_TO_Z_UP = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])


def load_mesh(filepath: str) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Returns (visual, geometry).

    The visual keeps the glb's split UV seams and vertex normals so textures map cleanly.
    The geometry copy merges seam vertices so it can be watertight.
    """
    visual = trimesh.load(filepath, force="mesh")
    visual.apply_transform(Y_UP_TO_Z_UP)
    visual.vertex_normals  # computed and cached now, carried through scaling and export
    geometry = trimesh.Trimesh(visual.vertices.copy(), visual.faces.copy(), process=True)
    geometry.merge_vertices()
    return visual, geometry


def repair(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    """Return a geometry-only copy that is watertight if we can make it so."""
    notes = []
    fixed = mesh.copy()
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


def simplify_visual(mesh: trimesh.Trimesh, max_faces: int = VISUAL_MAX_FACES) -> trimesh.Trimesh:
    """Quadric decimation that respects UV seams, so the texture still lines up afterwards."""
    if len(mesh.faces) <= max_faces:
        return mesh
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=mesh.vertices, face_matrix=mesh.faces, v_tex_coords_matrix=mesh.visual.uv
    ))
    # per-corner UVs let the seam vertices merge without losing the seams themselves
    ms.compute_texcoord_transfer_vertex_to_wedge()
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.001))
    ms.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=max_faces, preserveboundary=True, planarquadric=True
    )
    out = ms.current_mesh()
    vertices, faces = out.vertex_matrix(), out.face_matrix()
    corner_uv = out.wedge_tex_coord_matrix().reshape(-1, 2)

    # split vertices again wherever a corner's UV differs (the seams)
    keys = np.column_stack([faces.reshape(-1), np.round(corner_uv * 1e6)])
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    smooth_normals = trimesh.Trimesh(vertices, faces, process=False).vertex_normals
    corner_vertex = faces.reshape(-1)[first]
    return trimesh.Trimesh(
        vertices=vertices[corner_vertex],
        faces=inverse.reshape(-1, 3),
        vertex_normals=smooth_normals[corner_vertex],
        visual=trimesh.visual.TextureVisuals(uv=corner_uv[first], material=mesh.visual.material),
        process=False,
    )
