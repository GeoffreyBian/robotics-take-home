"""Previews rendered by MuJoCo from the exported scene, so they show exactly what loads."""

import os

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VISUAL_GROUP, COLLISION_GROUP = 2, 3


def _camera(model, data, azimuth=-125.0, elevation=-22.0, fit=1.5) -> mujoco.MjvCamera:
    """Orbit camera framed on the asset's bodies (body 0 is the world)."""
    mujoco.mj_forward(model, data)
    visual = [i for i in range(model.ngeom) if model.geom_group[i] == VISUAL_GROUP]
    centers = data.geom_xpos[visual]
    radius = model.geom_rbound[visual]
    lo = (centers - radius[:, None]).min(axis=0)
    hi = (centers + radius[:, None]).max(axis=0)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (lo + hi) / 2
    camera.distance = fit * np.linalg.norm(hi - lo) / 2 / np.tan(np.radians(model.vis.global_.fovy / 2)) / 1.6
    camera.azimuth, camera.elevation = azimuth, elevation
    return camera


def _groups(*visible) -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1  # floor
    for g in visible:
        option.geomgroup[g] = 1
    return option


def preview(scene_path: str, output_path: str, size=(720, 720)):
    """Visual mesh next to the collision geometry, one colour per convex part."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    collision = [i for i in range(model.ngeom) if model.geom_group[i] == COLLISION_GROUP]
    palette = plt.get_cmap("tab20")(np.arange(len(collision)) % 20)
    for i, colour in zip(collision, palette):
        model.geom_rgba[i] = colour
        model.geom_matid[i] = -1
    camera = _camera(model, data)
    floor_reflectance = model.mat_reflectance.copy()
    try:
        font = ImageFont.load_default(size=26)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    tiles = []
    with mujoco.Renderer(model, size[1], size[0]) as renderer:
        for group, title in ((VISUAL_GROUP, "visual"), (COLLISION_GROUP, f"collision ({len(collision)} parts)")):
            # no floor reflection under the collision view, where it reads as ghost geometry
            model.mat_reflectance[:] = floor_reflectance if group == VISUAL_GROUP else 0
            renderer.update_scene(data, camera, scene_option=_groups(group))
            tile = Image.fromarray(renderer.render())
            ImageDraw.Draw(tile).text((18, 14), title, fill=(25, 25, 25), font=font)
            tiles.append(tile)
    sheet = Image.new("RGB", (size[0] * 2, size[1]))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * size[0], 0))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path)


def render_drop(scene_path: str, output_path: str, duration: float = 3.0, fps: int = 30, size=(640, 480)):
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    camera = _camera(model, data, elevation=-15.0, fit=2.6)
    mujoco.mj_resetDataKeyframe(model, data, model.key("drop").id)
    camera.lookat[2] = data.qpos[2] / 2

    frames = []
    with mujoco.Renderer(model, size[1], size[0]) as renderer:
        while data.time < duration:
            mujoco.mj_step(model, data)
            if len(frames) < data.time * fps:
                renderer.update_scene(data, camera, scene_option=_groups(VISUAL_GROUP))
                frames.append(Image.fromarray(renderer.render()))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
