"""Fixed props with their own geometry and surface materials.

The YAM loader gives plain Boxes rubber-like friction (1.2/1.0); these props author their own:
- `WoodBoard`: the stained pine board, finished-wood friction (0.35/0.30).
- `TrashBin`: open-top round plastic wastebasket, 24 wall staves and a floor (really open, really hollow).
- `StainCan`: gallon can of oil-based stain, lid off, with a printed label band turned toward the overhead camera.
- `StainCanLid`, `FoamBrush`: the can's lid lying upside down, and the foam brush.
All are static colliders. Registration follows the accepted task `peg-out-a-tshirt`
(arena_props.py, MIT licence, see LICENSE).
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from yam_arena.task_file import Box

HERE = Path(__file__).resolve().parent
SPECS = {}


@dataclass(frozen=True)
class WoodBoard(Box):
    static_friction: float = 0.35
    dynamic_friction: float = 0.30


@dataclass(frozen=True)
class TrashBin(Box):
    """`pos` is the centre of the bin's floor on the bench."""
    radius: float = 0.115
    height: float = 0.26
    wall: float = 0.004


@dataclass(frozen=True)
class StainCan(Box):
    """`pos` is the centre of the can's base on the bench; the label is centred at `label_angle` (rad about z)."""
    radius: float = 0.084
    height: float = 0.190
    label_angle: float = math.pi


@dataclass(frozen=True)
class StainCanLid(Box):
    radius: float = 0.080


@dataclass(frozen=True)
class FoamBrush(Box):
    pass


# -- USD helpers ------------------------------------------------------------------------------------
def _collide(prim, offset=0.004):
    from pxr import PhysxSchema, UsdPhysics
    UsdPhysics.CollisionAPI.Apply(prim)
    cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    cp.CreateContactOffsetAttr(offset)
    cp.CreateRestOffsetAttr(0.0)


def _friction(stage, prim, path, static, dynamic):
    from pxr import UsdPhysics, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(static)
    pm.CreateDynamicFrictionAttr(dynamic)
    pm.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _look(stage, prim, path, color, roughness=0.6, metallic=0.0, texture=None):
    from pxr import Gf, Sdf, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if texture is None:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    else:
        reader = UsdShade.Shader.Define(stage, path + "/St")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, path + "/Tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _cube(stage, path, size, pos, color, rot_z=0.0, collide=True):
    from pxr import Gf, UsdGeom
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rot_z:
        c.AddRotateZOp().Set(math.degrees(rot_z))
    c.AddScaleOp().Set(Gf.Vec3d(*size))
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        _collide(c.GetPrim())
    return c


def _cylinder(stage, path, radius, height, pos, color, collide=True, axis="Z"):
    from pxr import Gf, UsdGeom
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(axis)
    c.AddTranslateOp().Set(Gf.Vec3d(*pos))
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        _collide(c.GetPrim())
    return c


def label_band(stage, path, radius, z0, z1, centre_angle, span_deg, segments=32):
    """Curved label mesh around a can, with texture coordinates; visual only."""
    from pxr import Gf, Sdf, UsdGeom, Vt
    m = UsdGeom.Mesh.Define(stage, path)
    a0 = centre_angle - math.radians(span_deg) / 2
    pts, st, idx, counts = [], [], [], []
    for i in range(segments + 1):
        u = i / segments
        a = a0 + u * math.radians(span_deg)
        for v, z in ((0.0, z0), (1.0, z1)):
            pts.append(Gf.Vec3f(radius * math.cos(a), radius * math.sin(a), z))
            st.append(Gf.Vec2f(u, v))  # u runs from the robot's left to its right as seen from -x
    for i in range(segments):
        a, b, c, d = 2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1
        idx += [a, b, c, d]
        counts.append(4)
    m.CreatePointsAttr(Vt.Vec3fArray(pts))
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateDoubleSidedAttr(True)
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray(st))
    return m


# -- per-prop authoring ------------------------------------------------------------------------------
def _author(stage, path, spec):
    from pxr import Gf, UsdGeom
    root = UsdGeom.Xform.Define(stage, path)
    x, y, z = spec.pos
    if isinstance(spec, WoodBoard):
        c = _cube(stage, path + "/Shape", spec.size, spec.pos, spec.color, collide=True)
        _friction(stage, c.GetPrim(), path + "/Friction", spec.static_friction, spec.dynamic_friction)
        _look(stage, c.GetPrim(), path + "/Look", spec.color, roughness=0.5)
    elif isinstance(spec, TrashBin):
        n, r_mid = 24, spec.radius - spec.wall / 2
        width = 2 * spec.radius * math.tan(math.pi / n) + 0.002
        g = UsdGeom.Xform.Define(stage, path + "/Body")
        g.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        for i in range(n):
            a = 2 * math.pi * i / n
            _cube(stage, f"{path}/Body/stave{i:02d}", (spec.wall, width, spec.height),
                  (r_mid * math.cos(a), r_mid * math.sin(a), spec.height / 2), spec.color, rot_z=a)
        _cylinder(stage, path + "/Body/floor", spec.radius, 0.004, (0, 0, 0.002), spec.color)
        _friction(stage, g.GetPrim(), path + "/Friction", 0.4, 0.35)
        _look(stage, g.GetPrim(), path + "/Look", spec.color, roughness=0.55)
    elif isinstance(spec, StainCan):
        g = UsdGeom.Xform.Define(stage, path + "/Body")
        g.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        body = _cylinder(stage, path + "/Body/shell", spec.radius, spec.height, (0, 0, spec.height / 2),
                         (0.70, 0.70, 0.72))
        _look(stage, body.GetPrim(), path + "/Body/shell/Look", (0.70, 0.70, 0.72), roughness=0.3, metallic=0.8)
        # The open can seen from above: a silver rim ring around the dark liquid surface (the shell's
        # collider is solid, so the surface is drawn just on top of it).
        stain = _cylinder(stage, path + "/Body/stain", spec.radius - 0.007, 0.001,
                          (0, 0, spec.height + 0.0006), (0.20, 0.10, 0.04), collide=False)
        _look(stage, stain.GetPrim(), path + "/Body/stain/Look", (0.06, 0.025, 0.01), roughness=0.08)
        band = label_band(stage, path + "/Body/label", spec.radius + 0.0006, 0.012, 0.178, spec.label_angle, 360, 72)
        _look(stage, band.GetPrim(), path + "/Body/label/Look", None, roughness=0.6,
              texture=HERE / "stain_label.png")
    elif isinstance(spec, StainCanLid):
        lid = _cylinder(stage, path + "/Shape", spec.radius, 0.010, (x, y, z + 0.005), (0.70, 0.70, 0.72))
        _look(stage, lid.GetPrim(), path + "/Look", (0.70, 0.70, 0.72), roughness=0.3, metallic=0.8)
    elif isinstance(spec, FoamBrush):
        # 50 mm foam brush: wooden handle 150 x 20 x 12 mm along y, grey foam head 50 x 60 x 12 mm.
        h = _cube(stage, path + "/handle", (0.020, 0.150, 0.012), (x, y - 0.055, z), (0.80, 0.62, 0.40))
        _look(stage, h.GetPrim(), path + "/handle/Look", (0.80, 0.62, 0.40))
        f = _cube(stage, path + "/foam", (0.050, 0.060, 0.012), (x, y + 0.050, z), (0.35, 0.22, 0.12))
        _look(stage, f.GetPrim(), path + "/foam/Look", (0.35, 0.22, 0.12), roughness=0.95)
    else:
        raise TypeError(type(spec))
    return root.GetPrim()


def _spawn(path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.utils.stage import get_current_stage
    if orientation is not None and not np.allclose(orientation, (0., 0., 0., 1.)):
        raise ValueError("props use the authored frame")
    return _author(get_current_stage(), str(path), SPECS[cfg.prop_name])


def prop_asset(spec):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    SPECS[spec.name] = spec

    @configclass
    class SpawnCfg(SpawnerCfg):
        func: object = clone(_spawn)
        prop_name: str = ""

    return Object(name=spec.name, object_type=ObjectType.BASE, spawner_cfg=SpawnCfg(prop_name=spec.name),
                  initial_pose=Pose(position_xyz=(0., 0., 0.), rotation_xyzw=(0., 0., 0., 1.)))


def register():
    import cloth_bridge
    for cls in (WoodBoard, TrashBin, StainCan, StainCanLid, FoamBrush):
        if not any(c is cls for c, _ in cloth_bridge.EXTRA_FACTORIES):
            cloth_bridge.EXTRA_FACTORIES.append((cls, prop_asset))
