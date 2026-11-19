import os
import time

from . import collision, export, physics, render, validate
from .load import load_mesh, repair
from .priors import Prior


def make_sim_ready(
    filepath: str,
    prior: Prior,
    out_dir: str,
    collision_method: str = "coacd",
    render_outputs: bool = True,
    run_validation: bool = True,
) -> dict:
    """glb in, MuJoCo asset package out. Returns the summary written to properties.json."""
    name, _ = os.path.splitext(os.path.basename(filepath))
    visual = load_mesh(filepath)
    solid, repair_notes = repair(visual)

    scale = physics.scale_factor(visual, prior)
    visual.apply_scale(scale)
    solid.apply_scale(scale)

    start = time.perf_counter()
    hulls = collision.METHODS[collision_method](solid)
    decompose_s = time.perf_counter() - start

    props = physics.mass_properties(solid, hulls, prior)
    hull_volume = sum(h.volume for h in hulls)
    report = {
        "source": filepath,
        "scale_factor": scale,
        "extents_m": visual.extents.tolist(),
        "watertight": solid.is_watertight,
        "repairs": repair_notes,
        "collision": {
            "method": collision_method,
            "parts": len(hulls),
            # >1 means the collision shape is fatter than the object
            "volume_ratio": hull_volume / props.volume,
            "seconds": round(decompose_s, 2),
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    scene = export.write_package(out_dir, name, visual, hulls, props, prior)
    if run_validation:
        report["validation"] = validate.run(out_dir, visual, hulls, prior.longest_m)
    export.write_properties(out_dir, name, props, prior, report)
    if render_outputs:
        render.visualize(visual, hulls, os.path.join(out_dir, "preview.png"))
        render.render_drop(scene, os.path.join(out_dir, "drop.gif"))
    return {"name": name, "mass_kg": props.mass, **report}
