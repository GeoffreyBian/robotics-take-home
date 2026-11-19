import hashlib
import os

import coacd
import numpy as np
import trimesh

from .asset import Part

# hulls below this share of the total hull volume add contacts without adding shape
MIN_HULL_VOLUME_FRACTION = 1e-4
# stop refining once the collision shape is at most this much bigger than the object
TARGET_VOLUME_RATIO = 1.3
# concavity thresholds tried in order, coarse to fine (lower = more faithful, slower)
THRESHOLDS = (0.05, 0.03, 0.02)
# total convex parts per asset: every part is a contact candidate on every step
DEFAULT_BUDGET = 64
CACHE_DIR = ".cache/coacd"


def union_volume(hulls: list[trimesh.Trimesh], n: int = 200_000) -> float:
    """Monte Carlo volume of the union, since overlapping hulls double count in a plain sum."""
    corners = np.vstack([h.bounds for h in hulls])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    points = np.random.default_rng(0).uniform(lo, hi, (n, 3))
    inside = np.zeros(n, dtype=bool)
    for hull in hulls:
        offsets = np.einsum("ij,ij->i", hull.face_normals, hull.vertices[hull.faces[:, 0]])
        inside[~inside] = ((points[~inside] @ hull.face_normals.T - offsets) <= 0).all(axis=1)
    return float(inside.mean() * np.prod(hi - lo))


def _cache_path(mesh: trimesh.Trimesh, threshold: float, max_hulls: int, seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    digest.update(f"{threshold}|{max_hulls}|{seed}|{coacd.__version__ if hasattr(coacd, '__version__') else ''}".encode())
    return os.path.join(CACHE_DIR, digest.hexdigest()[:24] + ".npz")


def decompose(mesh: trimesh.Trimesh, threshold: float, max_hulls: int, seed: int = 0) -> list[trimesh.Trimesh]:
    """CoACD, cached on disk by mesh content and parameters."""
    path = _cache_path(mesh, threshold, max_hulls, seed)
    if os.path.exists(path):
        data = np.load(path)
        return [trimesh.Trimesh(data[f"v{i}"], data[f"f{i}"]) for i in range(int(data["n"]))]

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold, max_convex_hull=max_hulls, seed=seed
    )
    # re-hull each part so every piece is guaranteed convex and watertight
    hulls = [trimesh.Trimesh(v, f).convex_hull for v, f in parts]
    total = sum(h.volume for h in hulls)
    hulls = [h for h in hulls if h.volume > MIN_HULL_VOLUME_FRACTION * total]

    os.makedirs(CACHE_DIR, exist_ok=True)
    arrays = {"n": len(hulls)}
    for i, h in enumerate(hulls):
        arrays[f"v{i}"], arrays[f"f{i}"] = h.vertices, h.faces
    np.savez_compressed(path, **arrays)
    return hulls


def adaptive_decomposition(mesh: trimesh.Trimesh, budget: int) -> tuple[list[trimesh.Trimesh], dict]:
    """Refine the concavity threshold until the hulls hug the object or the budget runs out."""
    tried, best = [], None
    for threshold in THRESHOLDS:
        hulls = decompose(mesh, threshold, budget)
        ratio = union_volume(hulls) / mesh.volume
        tried.append({"threshold": threshold, "parts": len(hulls), "volume_ratio": round(ratio, 3)})
        if best is None or ratio < best[1]:
            best = (hulls, ratio, threshold)
        if ratio <= TARGET_VOLUME_RATIO:
            stop = "target reached"
            break
    else:
        stop = "budget reached" if len(best[0]) >= budget else "finest threshold reached"
    hulls, ratio, threshold = best
    return hulls, {"threshold": threshold, "parts": len(hulls), "volume_ratio": round(ratio, 3),
                   "stop": stop, "tried": tried}


def convex_hull(mesh: trimesh.Trimesh, budget: int) -> tuple[list[trimesh.Trimesh], dict]:
    hulls = [mesh.convex_hull]
    return hulls, {"parts": 1, "volume_ratio": round(hulls[0].volume / mesh.volume, 3)}


METHODS = {"coacd": adaptive_decomposition, "hull": convex_hull}


def build_collision(parts: list[Part], method: str, budget: int = DEFAULT_BUDGET):
    """Fill part.hulls. The hull budget is shared across parts by volume (at least 4 each)."""
    meshed = [p for p in parts if p.sphere is None and p.collides]
    volumes = np.array([p.solid.volume for p in meshed])
    shares = np.maximum(4, np.floor(budget * volumes / volumes.sum())).astype(int)
    while shares.sum() > budget and shares.max() > 4:
        shares[np.argmax(shares)] -= 1
    for part, share in zip(meshed, shares):
        part.hulls, part.collision_info = METHODS[method](part.solid, int(share))
    for part in parts:
        if part.sphere is not None:
            part.collision_info = {"parts": 1, "primitive": "sphere", "radius": round(part.sphere[1], 4)}
