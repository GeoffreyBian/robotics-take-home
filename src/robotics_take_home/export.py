"""Writes the asset package. Format-specific writers live in mjcf.py, urdf.py and usd.py."""

import json
import os

import numpy as np
import trimesh

from .asset import Part

# glTF texture slot -> our file suffix. glTF packs occlusion/roughness/metallic as R/G/B,
# the same layout MuJoCo's "orm" role and USD previews expect, so it is copied as-is
TEXTURE_SLOTS = {"baseColorTexture": "rgb", "normalTexture": "normal", "metallicRoughnessTexture": "orm"}


def write_textures(name: str, visual: trimesh.Trimesh, mesh_dir: str) -> dict[str, str]:
    """Returns {role: filename} for the textures the source asset has."""
    material = getattr(visual.visual, "material", None)
    files = {}
    for slot, role in TEXTURE_SLOTS.items():
        image = getattr(material, slot, None)
        if image is not None:
            filename = f"{name}_{role}.png"
            image.convert("RGB").save(os.path.join(mesh_dir, filename))
            files[role] = filename
    return files


def write_mtl(name: str, textures: dict[str, str], mesh_dir: str) -> str:
    """Wavefront material so the OBJs keep their texture in Blender, PyBullet, Isaac etc."""
    lines = [f"newmtl {name}_mat", "Kd 1 1 1", "Ka 0 0 0", "Ks 0 0 0"]
    if "rgb" in textures:
        lines.append(f"map_Kd {textures['rgb']}")
    if "normal" in textures:
        lines.append(f"map_Bump {textures['normal']}")
    filename = f"{name}.mtl"
    with open(os.path.join(mesh_dir, filename), "w") as f:
        f.write("\n".join(lines) + "\n")
    return filename


def write_obj(mesh: trimesh.Trimesh, path: str, mtl: str = None, material: str = None):
    textured = isinstance(mesh.visual, trimesh.visual.TextureVisuals) and mesh.visual.uv is not None
    text = trimesh.exchange.obj.export_obj(
        mesh, include_normals=True, include_texture=textured, return_texture=False, write_texture=False
    )
    lines = [l for l in text.splitlines(keepends=True) if not l.startswith(("mtllib", "usemtl"))]
    header = [f"mtllib {mtl}\n", f"usemtl {material}\n"] if mtl and textured else []
    with open(path, "w") as f:
        f.writelines(header + lines)


def write_meshes(name: str, parts: list[Part], textures: dict, mesh_dir: str):
    mtl = write_mtl(name, textures, mesh_dir) if textures else None
    for part in parts:
        write_obj(part.visual, os.path.join(mesh_dir, f"{part.name}_visual.obj"), mtl, f"{name}_mat")
        for i, hull in enumerate(part.hulls):
            write_obj(hull, os.path.join(mesh_dir, f"{part.name}_c{i:02d}.obj"))


def write_properties(out_dir: str, name: str, parts: list[Part], prior, report: dict):
    def part_entry(p: Part):
        entry = {
            "material": p.material,
            "parent": p.parent,
            "mass_kg": p.mass.mass,
            "center_of_mass": p.mass.center_of_mass.tolist(),
            "inertia": p.mass.inertia.tolist(),
            "mass_elements": p.mass.elements,
            "collision": p.collision_info,
            "visual_faces": len(p.visual.faces),
        }
        if p.joint is not None:
            j = p.joint
            entry["joint"] = {
                "type": j.kind, "anchor": j.anchor.tolist(), "axis": j.axis.tolist(),
                "range": list(j.range) if j.range else None,
                "stiffness": j.stiffness, "damping": j.damping,
            }
        return entry

    from dataclasses import asdict
    with open(os.path.join(out_dir, "properties.json"), "w") as f:
        json.dump(
            {"name": name, "prior": asdict(prior), **report, "parts": {p.name: part_entry(p) for p in parts}},
            f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o),
        )
