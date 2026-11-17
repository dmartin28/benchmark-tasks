"""Native PhysX surface-FEM cloth authoring. Import USD only after SimulationApp starts.

No runtime vertex targets, kinematic cloth or pose writes. Adapted from the accepted task
`peg-out-a-tshirt` (native_cloth.py, MIT licence, see LICENSE), trimmed to the cloth parts and given
a render material. Fabric parameters are the same provisional values that task used, at 0.6 mm.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Fabric:
    thickness: float = .0006
    density: float = 333.0  # 200 g/m2 at the nominal 0.6 mm thickness
    youngs_modulus: float = 2e6
    poissons_ratio: float = .3
    dynamic_friction: float = .55
    bend_modulus: float = 1.8e5
    elasticity_damping: float = .005
    bend_damping: float = .005
    contact_offset: float = .0008
    rest_offset: float = .0003
    solver_iterations: int = 32


def _set(prim, name, value):
    attr = prim.GetAttribute(name)
    if not attr:
        raise RuntimeError(f'Installed simulator lacks required attribute {name}')
    if not attr.Set(value):
        raise RuntimeError(f'Cannot author {name}')


def author_cloth(stage, path, mesh, material=Fabric(), points=None, color=(.62, .42, .22), texture=None):
    """One triangle mesh that is both simulated and rendered. Called only while building the scene."""
    import numpy as np
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt
    info = mesh.audit()
    p = mesh.rest if points is None else np.asarray(points)
    if p.shape != mesh.rest.shape or not np.isfinite(p).all():
        raise ValueError('initial cloth points do not match rest mesh')
    tri = UsdGeom.Mesh.Define(stage, path)
    tri.CreatePointsAttr(Vt.Vec3fArray(p.tolist()))
    tri.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    tri.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
    tri.CreateSubdivisionSchemeAttr('none')
    tri.CreateDoubleSidedAttr(True)
    tri.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim = tri.GetPrim()
    for schema in ('OmniPhysicsDeformableBodyAPI', 'OmniPhysicsSurfaceDeformableSimAPI',
                   'PhysxSurfaceDeformableBodyAPI'):
        if not prim.ApplyAPI(schema):
            raise RuntimeError(f'Native surface cloth API unavailable: {schema}')
    _set(prim, 'omniphysics:restShapePoints', Vt.Vec3fArray(mesh.rest.tolist()))
    _set(prim, 'omniphysics:restTriVtxIndices', [Gf.Vec3i(*map(int, t)) for t in mesh.faces])
    _set(prim, 'omniphysics:restBendAnglesDefault', 'restShapeDefault')
    _set(prim, 'omniphysics:mass', info['surface_area_m2'] * material.thickness * material.density)
    _set(prim, 'physxDeformableBody:selfCollision', True)
    _set(prim, 'physxDeformableBody:disableGravity', False)
    _set(prim, 'physxDeformableBody:solverPositionIterationCount', material.solver_iterations)
    UsdPhysics.CollisionAPI.Apply(prim)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(material.contact_offset)
    collision.CreateRestOffsetAttr(material.rest_offset)
    mat = UsdShade.Material.Define(stage, path + '/Fabric')
    m = mat.GetPrim()
    for api in ('OmniPhysicsBaseMaterialAPI', 'OmniPhysicsDeformableMaterialAPI',
                'OmniPhysicsSurfaceDeformableMaterialAPI', 'PhysxSurfaceDeformableMaterialAPI'):
        if not m.ApplyAPI(api):
            raise RuntimeError(f'Missing material schema {api}')
    for attr, val in {'omniphysics:density': material.density,
                      'omniphysics:dynamicFriction': material.dynamic_friction,
                      'omniphysics:youngsModulus': material.youngs_modulus,
                      'omniphysics:poissonsRatio': material.poissons_ratio,
                      'omniphysics:surfaceThickness': material.thickness,
                      'omniphysics:surfaceBendStiffness': material.bend_modulus,
                      'physxDeformableMaterial:elasticityDamping': material.elasticity_damping,
                      'physxDeformableMaterial:bendDamping': material.bend_damping}.items():
        _set(m, attr, val)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
    # Render material: RTX ignores displayColor alone, so bind a matte preview surface.
    look = UsdShade.Material.Define(stage, path + '/Look')
    shader = UsdShade.Shader.Define(stage, path + '/Look/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    if texture is None or mesh.uv is None:
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    else:   # printed look (stain blotches) mapped with the flat sheet's coordinates; visual only
        corners = mesh.uv[mesh.faces.ravel()]   # one coordinate per triangle corner
        pv = UsdGeom.PrimvarsAPI(tri).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                                    UsdGeom.Tokens.faceVarying)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(float(a), float(b)) for a, b in corners]))
        reader = UsdShade.Shader.Define(stage, path + '/Look/St')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        tex = UsdShade.Shader.Define(stage, path + '/Look/Tex')
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
        tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.9)
    look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI(prim).Bind(look, UsdShade.Tokens.weakerThanDescendants)
    return tri


def ensure_gpu_scene(stage):
    from pxr import PhysxSchema, UsdPhysics
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        raise RuntimeError('Create the simulation context before cloth')
    for prim in scenes:
        api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        api.CreateEnableGPUDynamicsAttr(True)
        api.CreateBroadphaseTypeAttr('GPU')
        api.CreateEnableCCDAttr(False)  # unsupported together with the GPU tensor pipeline
        api.CreateSolverTypeAttr('TGS')
