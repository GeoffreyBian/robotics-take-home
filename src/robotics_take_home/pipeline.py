import os
import time

from . import mjcf, render, urdf, usd, validate
from .articulate import build_parts, finalize_joints
from .collision import DEFAULT_BUDGET, build_collision
from .export import write_meshes, write_properties, write_textures
from .load import load_mesh, repair, simplify_visual
from .materials import MaterialTable
from .physics import assign_masses, scale_factor
from .priors import Prior


def make_sim_ready(
    filepath: str,
    prior: Prior,
    materials: MaterialTable,
    out_dir: str,
    collision_method: str = "coacd",
    hull_budget: int = DEFAULT_BUDGET,
    render_outputs: bool = True,
    run_validation: bool = True,
) -> dict:
    """glb in, MuJoCo / URDF / USD asset package out. Returns the summary in properties.json."""
    name, _ = os.path.splitext(os.path.basename(filepath))
    visual, geometry = load_mesh(filepath)
    solid, repair_notes = repair(geometry)

    scale = scale_factor(visual, prior)
    visual.apply_scale(scale)
    solid.apply_scale(scale)
    source_faces = len(visual.faces)
    visual = simplify_visual(visual)

    parts, articulation_notes = build_parts(name, visual, solid, prior)
    start = time.perf_counter()
    build_collision(parts, collision_method, hull_budget)
    collision_s = time.perf_counter() - start
    mass_summary = assign_masses(parts, prior, materials)
    finalize_joints(parts)

    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    textures = write_textures(name, visual, mesh_dir)
    write_meshes(name, parts, textures, mesh_dir)
    with open(os.path.join(out_dir, "model.xml"), "w") as f:
        f.write(mjcf.model_xml(name, parts, textures, materials))
    scene = os.path.join(out_dir, "scene.xml")
    with open(scene, "w") as f:
        f.write(mjcf.scene_xml(name, parts, materials))
    with open(os.path.join(out_dir, "model.urdf"), "w") as f:
        f.write(urdf.urdf(name, parts, materials))
    usd.write_usd(os.path.join(out_dir, "model.usda"), name, parts, textures, materials)

    report = {
        "source": filepath,
        "scale_factor": scale,
        "extents_m": (visual.bounds[1] - visual.bounds[0]).tolist(),
        "watertight": solid.is_watertight,
        "repairs": repair_notes,
        "visual_faces": {"source": source_faces, "exported": sum(len(p.visual.faces) for p in parts)},
        "articulation": articulation_notes,
        "mass": mass_summary,
        "collision": {
            "method": collision_method,
            "parts": sum(p.collision_info["parts"] for p in parts),
            "seconds": round(collision_s, 2),
        },
    }
    if run_validation:
        report["validation"] = validate.run(out_dir, parts, prior, materials)
    write_properties(out_dir, name, parts, prior, report)
    if render_outputs:
        render.preview(scene, os.path.join(out_dir, "preview.png"))
        render.render_drop(scene, os.path.join(out_dir, "drop.gif"))
    return {"name": name, **report}
