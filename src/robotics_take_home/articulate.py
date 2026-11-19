"""Split a single reconstructed mesh into jointed parts using simple per-category templates.

Reconstructions arrive as one fused surface, so each template finds its cut planes from
horizontal cross-sections (where the shape gets thin, or splits into separate loops) and
slices both the textured visual and the closed solid with the same planes.
"""

import numpy as np
import trimesh

from .asset import Joint, Part, face_subset, halfspace

GRAVITY = 9.81
UP = np.array([0.0, 0.0, 1.0])


def single_part(name, visual, solid, material) -> list[Part]:
    return [Part(name=name, visual=visual, solid=solid, material=material)]


def build_parts(name, visual, solid, prior) -> tuple[list[Part], list[str]]:
    """Returns parts and notes. Falls back to one rigid part if a template can't find its cuts."""
    spec = prior.articulation
    if not spec:
        return single_part(name, visual, solid, prior.material), []
    builder = {"lid": _lid, "flaps": _flaps, "office_chair": _office_chair}[spec["type"]]
    try:
        return builder(name, visual, solid, prior.material, spec)
    except ArticulationError as e:
        return single_part(name, visual, solid, prior.material), [f"{spec['type']}: {e}, kept rigid"]


class ArticulationError(Exception):
    pass


def _sections(solid, lo, hi, steps=200):
    """(z, loops) for horizontal slices; each loop is an (N, 3) polyline."""
    out = []
    for z in np.linspace(lo, hi, steps):
        s = solid.section(UP, [0, 0, z])
        out.append((z, [] if s is None else list(s.discrete)))
    return out


def _extent(loops):
    pts = np.vstack(loops)[:, :2]
    return np.ptp(pts, axis=0), pts.min(axis=0), pts.max(axis=0)


def _faces_where(visual, keep) -> np.ndarray:
    return keep(visual.triangles_center)


def _lid(name, visual, solid, material, spec):
    """Box with one thin panel standing up from a top edge (an open mailer box)."""
    lo, hi = solid.bounds
    height = hi[2] - lo[2]
    body_min_width = np.min(solid.extents[:2])
    z_cut = None
    for z, loops in _sections(solid, lo[2] + 0.3 * height, hi[2] - 0.02 * height):
        if loops and _extent(loops)[0].min() < 0.25 * body_min_width:
            z_cut = z
            break
    if z_cut is None:
        raise ArticulationError("no thin panel above the box")

    lid_solid = halfspace(solid, [0, 0, z_cut], UP)
    body_solid = halfspace(solid, [0, 0, z_cut], -UP)
    if len(lid_solid.split(only_watertight=False)) != 1:
        raise ArticulationError("lid region is not one piece")

    # hinge runs along the lid's long horizontal side, on the cut plane
    lid_lo, lid_hi = lid_solid.bounds
    along = int(np.argmax(lid_solid.extents[:2]))
    anchor = (lid_lo + lid_hi) / 2
    # the hinge is where the lid meets the cut, not the middle of its (tilted) bounding box
    edge = lid_solid.vertices[lid_solid.vertices[:, 2] < z_cut + 0.005]
    anchor[1 - along] = np.median(edge[:, 1 - along])
    anchor[2] = z_cut
    toward = np.zeros(3)
    toward[1 - along] = body_solid.centroid[1 - along] - anchor[1 - along]
    axis = _hinge_axis(along, toward)
    # closed = lid lying flat over the box; how far it has to swing from the photo pose
    arm = lid_solid.centroid - anchor
    arm -= (arm @ axis) * axis
    closed = toward / np.linalg.norm(toward)
    to_close = np.arctan2(axis @ np.cross(arm, closed), arm @ closed) % (2 * np.pi)

    above = _faces_where(visual, lambda c: c[:, 2] >= z_cut)
    return [
        Part(name=name, visual=face_subset(visual, ~above), solid=body_solid, material=material),
        Part(
            name=f"{name}_lid", visual=face_subset(visual, above), solid=lid_solid, material=material,
            parent=name,
            # 0 = as photographed (open); to_close = lying shut over the box. Lid and box are
            # parent/child so MuJoCo skips their contacts; the upper limit stands in for them
            joint=Joint("hinge", anchor, axis, range=(-0.3, float(to_close)), hold_pose=True),
        ),
    ], [f"lid: hinge at z={z_cut:.3f} m, closes after {np.degrees(to_close):.0f} deg"]


def _hinge_axis(along: int, toward: np.ndarray) -> np.ndarray:
    """Axis along x or y whose positive rotation moves a point above the hinge toward `toward`."""
    axis = np.zeros(3)
    axis[along] = 1.0
    # rotation about axis by +theta moves the point (0,0,h) along axis x z
    if np.dot(np.cross(axis, UP), toward) < 0:
        axis = -axis
    return axis


def _flaps(name, visual, solid, material, spec):
    """Open box whose top flaps hang folded outward over the walls."""
    lo, hi = solid.bounds
    height = hi[2] - lo[2]
    # walls give two loops (outside and inside); a flap's free bottom edge adds more
    sections = _sections(solid, lo[2] + 0.5 * height, hi[2] - 0.02 * height)
    z_free = next((z for z, loops in sections if len(loops) > 2), None)
    if z_free is None:
        raise ArticulationError("no flap edges found")
    below = [loops for z, loops in sections if z < z_free - 0.01 * height and loops]
    if not below:
        raise ArticulationError("no wall section below the flaps")
    # outer wall face just below the flaps; anything further out above z_free is flap
    _, wall_lo, wall_hi = _extent(below[-1])
    z_cut = z_free - 0.01 * height
    margin = 0.002 * solid.extents.max()

    parts = []
    taken = np.zeros(len(visual.faces), bool)
    upper = halfspace(solid, [0, 0, z_cut], UP)
    body = halfspace(solid, [0, 0, z_cut], -UP)
    centers = visual.triangles_center
    remaining_upper = upper
    for axis_i, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        outward = np.zeros(3)
        outward[axis_i] = sign
        wall = (wall_hi if sign > 0 else wall_lo)[axis_i] + sign * margin
        origin = np.zeros(3)
        origin[axis_i] = wall
        try:
            flap = halfspace(remaining_upper, origin, outward)
            remaining_upper = halfspace(remaining_upper, origin, -outward)
        except Exception:
            continue
        if flap.is_empty or flap.volume <= 0 or flap.volume < 1e-3 * solid.volume:
            continue
        mask = ~taken & (centers[:, 2] >= z_cut) & (sign * (centers[:, axis_i] - wall) >= 0)
        if mask.sum() < 20:
            continue
        taken |= mask
        f_lo, f_hi = flap.bounds
        anchor = (f_lo + f_hi) / 2
        anchor[axis_i] = wall
        anchor[2] = f_hi[2]  # the fold is at the top of the hanging flap
        along = 1 - axis_i
        # positive angle lifts the flap's free edge outward and up, back over the box
        # (the free edge hangs below the hinge, hence -outward)
        axis = _hinge_axis(along, -outward)
        parts.append(Part(
            # pos/neg rather than +/-: USD prim names allow neither
            name=f"{name}_flap_{'xy'[axis_i]}{'pos' if sign > 0 else 'neg'}",
            visual=face_subset(visual, mask), solid=flap, material=material, parent=name,
            joint=Joint("hinge", anchor, axis, range=(-0.3, 3.2), hold_pose=True),
        ))
    if not parts:
        raise ArticulationError("flap cuts produced no usable pieces")

    body_solid = trimesh.util.concatenate([body, remaining_upper])
    root = Part(name=name, visual=face_subset(visual, ~taken), solid=body_solid, material=material)
    return [root] + parts, [f"flaps: {len(parts)} hinged at the top edge, free edges from z={z_free:.3f} m"]


def _office_chair(name, visual, solid, material, spec):
    """Seat swivels on the gas column; casters below the star base swivel and roll."""
    lo, hi = solid.bounds
    height = hi[2] - lo[2]

    # swivel: the narrowest horizontal section in the lower third is the gas column
    column = [(z, loops) for z, loops in _sections(solid, lo[2] + 0.12 * height, lo[2] + 0.35 * height, 120) if loops]
    z_swivel, loops = min(column, key=lambda zl: np.prod(_extent(zl[1])[0]))
    width, c_lo, c_hi = _extent(loops)
    if width.max() > 0.15 * solid.extents[:2].max():
        raise ArticulationError("no narrow column to swivel on")
    pivot = np.array([*(c_lo + c_hi) / 2, z_swivel])

    seat_solid = halfspace(solid, pivot, UP)
    base_full = halfspace(solid, pivot, -UP)

    # casters: going down from the star base, the first height where the section is only
    # separate small loops is just above the wheels
    z_caster = None
    for z, loops in reversed(_sections(solid, lo[2] + 0.005 * height, lo[2] + 0.12 * height, 120)):
        big = [l for l in loops if np.ptp(l[:, :2], axis=0).max() > 0.02 * height]
        if len(big) >= 3 and all(np.ptp(l[:, :2], axis=0).max() < 0.12 * solid.extents[:2].max() for l in big):
            z_caster = z
            break
    if z_caster is None:
        raise ArticulationError("no caster wheels found")
    base_solid = halfspace(base_full, [0, 0, z_caster], UP)
    wheels = [w for w in halfspace(base_full, [0, 0, z_caster], -UP).split(only_watertight=False)
              if w.volume > 0.01 * solid.volume / 100]
    if len(wheels) < 3:
        raise ArticulationError(f"found {len(wheels)} caster wheels")

    centers = visual.triangles_center
    seat_mask = centers[:, 2] >= z_swivel
    wheel_mask = centers[:, 2] < z_caster
    wheel_xy = np.array([w.bounds.mean(axis=0)[:2] for w in wheels])
    nearest = np.argmin(np.linalg.norm(centers[:, None, :2] - wheel_xy[None], axis=2), axis=1)

    base_material = spec.get("base_material", material)
    caster_material = spec.get("caster_material", material)
    parts = [
        Part(name=name, visual=face_subset(visual, ~seat_mask & ~wheel_mask), solid=base_solid,
             material=base_material),
        Part(name=f"{name}_seat", visual=face_subset(visual, seat_mask), solid=seat_solid,
             material=spec.get("seat_material", material), parent=name, joint=Joint("hinge", pivot, UP.copy(), damping=0.0)),
    ]
    for i, wheel in enumerate(wheels):
        w_lo, w_hi = wheel.bounds
        radius = min(w_hi[2] - lo[2], np.ptp(wheel.vertices[:, :2], axis=0).min()) / 2
        center = np.array([*(w_lo[:2] + w_hi[:2]) / 2, lo[2] + radius])
        parts.append(Part(
            name=f"{name}_caster{i}", visual=face_subset(visual, wheel_mask & (nearest == i)), solid=wheel,
            material=caster_material, parent=name,
            # a ball joint lets the wheel both swivel and roll; the sphere does the rolling
            joint=Joint("ball", center, damping=0.0), sphere=(center, radius),
        ))
    return parts, [f"office chair: swivel at z={z_swivel:.3f} m, {len(wheels)} casters below z={z_caster:.3f} m"]


def finalize_joints(parts: list[Part]):
    """Joint springs and damping that need the masses, which are known only after physics."""
    for part in parts:
        joint, mass = part.joint, part.mass
        if joint is None or joint.tuned:
            continue
        lever = np.linalg.norm(np.cross(mass.center_of_mass - joint.anchor, joint.axis))
        # inertia about the joint axis, by the parallel axis theorem
        r = mass.center_of_mass - joint.anchor
        inertia = joint.axis @ mass.inertia @ joint.axis + mass.mass * (r @ r - (r @ joint.axis) ** 2)
        if joint.hold_pose:
            # a creased flap holds its pose: gravity's torque sags it at most ~5 degrees
            joint.stiffness = mass.mass * GRAVITY * max(lever, 1e-3) / np.radians(5)
            joint.damping = 2 * np.sqrt(joint.stiffness * inertia)  # critically damped
        elif joint.kind == "ball":
            # rolling resistance so a nudged chair coasts to a stop
            joint.damping = 0.02 * mass.mass * GRAVITY * 0.01
        else:
            joint.damping = 0.05 * max(inertia, 1e-4)
