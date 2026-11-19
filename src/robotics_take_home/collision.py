import coacd
import trimesh

# hulls below this share of the total hull volume add contacts without adding shape
MIN_HULL_VOLUME_FRACTION = 1e-4


def convex_hull(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    return [mesh.convex_hull]


def convex_decomposition(
    mesh: trimesh.Trimesh, threshold: float = 0.05, max_hulls: int = 32, seed: int = 0
) -> list[trimesh.Trimesh]:
    """Approximate a concave mesh with convex parts so cavities and gaps survive.

    threshold is CoACD's concavity tolerance (lower = more faithful, more hulls).
    """
    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        threshold=threshold,
        max_convex_hull=max_hulls,
        seed=seed,
    )
    # re-hull each part so every piece is guaranteed convex and watertight
    hulls = [trimesh.Trimesh(v, f).convex_hull for v, f in parts]
    total = sum(h.volume for h in hulls)
    return [h for h in hulls if h.volume > MIN_HULL_VOLUME_FRACTION * total]


METHODS = {"coacd": convex_decomposition, "hull": convex_hull}
