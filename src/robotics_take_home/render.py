import os

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import trimesh
from PIL import Image


def render_mesh(mesh: trimesh.Trimesh, color):
    size, supersample = 700, 2  # render at 2x, box-filter down to antialias
    res = size * supersample

    # orbit the camera around the mesh, which is Z-up by the time it gets here
    target = mesh.bounds.mean(axis=0)
    elev, azim = np.radians(25), np.radians(-125)
    eye = target + mesh.extents.max() * 2.2 * np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
    )
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    basis = np.stack([right, np.cross(right, forward), forward])

    cam = (mesh.vertices - eye) @ basis.T
    depth = np.clip(cam[:, 2], 1e-6, None)
    ndc = cam[:, :2] / depth[:, None]
    # focal length that just fits the projection, so the mesh fills the frame
    # regardless of how elongated it is or where we view it from
    focal = (res / 2) * 0.94 / np.abs(ndc).max()
    screen = np.stack(
        [res / 2 + focal * ndc[:, 0], res / 2 - focal * ndc[:, 1]], axis=1
    )

    # flat-shade off a headlight; abs() keeps back-facing triangles from going black
    shade = np.abs(mesh.face_normals @ forward) * 0.75 + 0.25
    shaded = np.clip(np.asarray(color, dtype=float) * shade[:, None], 0, 1)

    zbuf = np.full((res, res), np.inf)
    image = np.ones((res, res, 3))
    tri_screen, tri_depth = screen[mesh.faces], depth[mesh.faces]
    half_edge = 0.7 * supersample / 2  # 0.7px wireframe

    # far to near only to help the cache; the depth buffer is what decides visibility
    for i in np.argsort(-tri_depth.mean(axis=1)):
        p, z = tri_screen[i], tri_depth[i]
        x0 = max(int(np.floor(p[:, 0].min())), 0)
        y0 = max(int(np.floor(p[:, 1].min())), 0)
        x1 = min(int(np.ceil(p[:, 0].max())), res - 1)
        y1 = min(int(np.ceil(p[:, 1].max())), res - 1)
        if x1 < x0 or y1 < y0:
            continue

        # signed area, doubled; also the barycentric denominator
        area2 = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (
            p[1, 1] - p[0, 1]
        )
        if abs(area2) < 1e-9:
            continue

        xs = np.arange(x0, x1 + 1)[None, :]
        ys = np.arange(y0, y1 + 1)[:, None]
        w1 = ((xs - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (ys - p[0, 1])) / area2
        w2 = ((p[1, 0] - p[0, 0]) * (ys - p[0, 1]) - (xs - p[0, 0]) * (p[1, 1] - p[0, 1])) / area2
        w0 = 1 - w1 - w2
        bary = np.stack([w0, w1, w2])
        inside = (bary >= 0).all(axis=0)
        if not inside.any():
            continue

        pixel_z = (bary * z[:, None, None]).sum(axis=0)
        window = zbuf[y0:y1 + 1, x0:x1 + 1]
        visible = inside & (pixel_z < window)
        if not visible.any():
            continue
        window[visible] = pixel_z[visible]

        # barycentric i scales to pixel distance from the opposite edge by 2*area/len
        lengths = np.linalg.norm(p[[1, 2, 0]] - p[[2, 0, 1]], axis=1)
        edge_dist = (bary * (abs(area2) / np.maximum(lengths, 1e-9))[:, None, None]).min(axis=0)
        tile = image[y0:y1 + 1, x0:x1 + 1]
        tile[visible] = shaded[i]
        tile[visible & (edge_dist < half_edge)] = 0.0

    return image.reshape(size, supersample, size, supersample, 3).mean(axis=(1, 3))


def hull_colors(hulls: list[trimesh.Trimesh]) -> np.ndarray:
    # one colour per convex part so the decomposition is readable
    palette = plt.get_cmap("tab20")(np.arange(len(hulls)) % 20)[:, :3]
    return np.concatenate([np.tile(c, (len(h.faces), 1)) for c, h in zip(palette, hulls)])


def visualize(visual: trimesh.Trimesh, hulls: list[trimesh.Trimesh], output_path: str):
    collision = trimesh.util.concatenate(hulls)
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    for ax, title, mesh, color in (
        (axes[0], f"visual  ({len(visual.faces)} faces)", visual, (0.35, 0.55, 0.85)),
        (axes[1], f"collision  ({len(hulls)} convex parts)", collision, hull_colors(hulls)),
    ):
        ax.imshow(render_mesh(mesh, color))
        ax.set_title(title)
        ax.set_axis_off()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_drop(scene_path: str, output_path: str, duration: float = 3.0, fps: int = 30):
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("drop").id)
    drop_height = data.qpos[2]

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance, camera.elevation, camera.lookat[2] = (
        drop_height * 4,
        -20,
        drop_height / 2,
    )

    frames = []
    with mujoco.Renderer(model, 480, 640) as renderer:
        while data.time < duration:
            mujoco.mj_step(model, data)
            if len(frames) < data.time * fps:
                renderer.update_scene(data, camera)
                frames.append(Image.fromarray(renderer.render()))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
