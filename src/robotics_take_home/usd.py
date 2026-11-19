"""USD export with UsdPhysics schemas (rigid bodies, mass, convex collision, joints).

Rigid bodies are siblings under the asset prim rather than nested, as UsdPhysics expects;
joints connect them. Visuals get a UsdPreviewSurface reading the same textures.
"""

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .asset import Part, tree_order
from .materials import MaterialTable


def _principal(inertia: np.ndarray):
    """USD stores inertia as diagonal moments plus the rotation to the principal axes."""
    moments, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(axes).as_quat()
    return Gf.Vec3f(*moments.tolist()), Gf.Quatf(float(w), float(x), float(y), float(z))


def _mesh(stage, path, mesh, offset, uv=False):
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((mesh.vertices - offset).astype(np.float32)))
    prim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
    prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.faces.reshape(-1).astype(np.int32)))
    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if uv and getattr(mesh.visual, "uv", None) is not None:
        prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertex_normals.astype(np.float32)))
        prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        st = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(mesh.visual.uv.astype(np.float32)))
    return prim


def _look(stage, path, textures, look):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{path}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def texture(role, colorspace):
        tex = UsdShade.Shader.Define(stage, f"{path}/{role}")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"meshes/{textures[role]}")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        return tex

    if "rgb" in textures:
        rgb = texture("rgb", "sRGB")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            rgb.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    if "normal" in textures:
        normal = texture("normal", "raw")
        normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    if "orm" in textures:
        orm = texture("orm", "raw")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
            orm.CreateOutput("g", Sdf.ValueTypeNames.Float))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
            orm.CreateOutput("b", Sdf.ValueTypeNames.Float))
    else:
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0 - look.shininess)
    if look.opacity < 1:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(look.opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _physics_material(stage, path, look):
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(look.default_friction)
    api.CreateDynamicFrictionAttr(look.default_friction)
    api.CreateRestitutionAttr(0.0)
    return material


def write_usd(path: str, name: str, parts: list[Part], textures: dict, materials: MaterialTable):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    parts = tree_order(parts)
    if len(parts) > 1:
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    used = sorted({m for p in parts for m in [p.material, *p.collision_materials()]})
    looks = {m: _look(stage, f"/{name}/Looks/{m}", textures, materials[m]) for m in used}
    physics = {m: _physics_material(stage, f"/{name}/PhysicsMaterials/{m}", materials[m]) for m in used}

    origins = {}
    for part in parts:
        origin = part.body_origin()
        origins[part.name] = origin
        body = UsdGeom.Xform.Define(stage, f"/{name}/{part.name}")
        body.AddTranslateOp().Set(Gf.Vec3d(*origin.tolist()))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass_api.CreateMassAttr(part.mass.mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*(part.mass.center_of_mass - origin).tolist()))
        moments, axes = _principal(part.mass.inertia)
        mass_api.CreateDiagonalInertiaAttr(moments)
        mass_api.CreatePrincipalAxesAttr(axes)

        visual = _mesh(stage, f"/{name}/{part.name}/visual", part.visual, origin, uv=True)
        UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(looks[part.material])

        for i, material in enumerate(part.collision_materials()):
            if part.sphere is not None:
                center, radius = part.sphere
                geom = UsdGeom.Sphere.Define(stage, f"/{name}/{part.name}/collision_{i:02d}")
                geom.CreateRadiusAttr(float(radius))
                geom.AddTranslateOp().Set(Gf.Vec3d(*(center - origin).tolist()))
            else:
                geom = _mesh(stage, f"/{name}/{part.name}/collision_{i:02d}", part.hulls[i], origin)
                UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            geom.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(
                physics[material], UsdShade.Tokens.weakerThanDescendants, "physics")

    for part in parts[1:]:
        j = part.joint
        path = f"/{name}/joints/{part.name}"
        local0 = Gf.Vec3f(*(origins[part.name] - origins[part.parent]).tolist())
        if j.kind == "hinge":
            joint = UsdPhysics.RevoluteJoint.Define(stage, path)
            # USD hinges are about X, Y or Z; a negative axis is the same hinge with q -> -q
            axis_i = int(np.argmax(np.abs(j.axis)))
            sign = float(np.sign(j.axis[axis_i]))
            joint.CreateAxisAttr("XYZ"[axis_i])
            if j.range:
                lo, hi = sorted(sign * np.degrees(np.array(j.range)))
                joint.CreateLowerLimitAttr(float(lo))
                joint.CreateUpperLimitAttr(float(hi))
            if j.stiffness or j.damping:
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
                # USD angular drives are per degree
                drive.CreateStiffnessAttr(float(j.stiffness * np.pi / 180))
                drive.CreateDampingAttr(float(j.damping * np.pi / 180))
                drive.CreateTargetPositionAttr(0.0)
        else:
            joint = UsdPhysics.SphericalJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/{name}/{part.parent}")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/{name}/{part.name}")])
        joint.CreateLocalPos0Attr(local0)
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    stage.GetRootLayer().Save()
