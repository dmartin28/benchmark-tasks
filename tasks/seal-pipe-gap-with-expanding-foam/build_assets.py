"""Write the scene's USD assets from geometry.py.

    python build_assets.py          # needs pxr (in Isaac Sim's Python), not the simulator

- assets/can.usda: the foam can, a floating articulation. The root body is the can with its valve cup, actuator (a
  small plastic cylinder) and straw (one rigid body, centre of mass low); the trigger is a second body on a revolute
  joint ``lever`` on the actuator's side under the straw (0 at rest, LEVER_REST_DOWN below level; LEVER_TRAVEL pressed further down toward the can).
  Its return spring is set by the task file (scenekit ``drives``). Origin: the centre of the can's base.
- assets/wall.usda: a fixed (kinematic) wall section: the rim board with a round hole (box colliders around a bore of
  box staves, a visual-only plate hiding the square opening's corners), the drywall backer closing the back of the
  hole, and the copper pipe through the hole's centre.
  Origin: the hole's centre on the board's front face; x runs into the wall.

Plain, unbranded colours; everything is built from primitives here, so no outside assets or licences are involved.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import geometry as g

OUT = Path(__file__).resolve().parent / "assets"

CAN_BLUE = (0.20, 0.36, 0.62)
CAN_SILVER = (0.78, 0.79, 0.80)
ADAPTER_BLACK = (0.08, 0.08, 0.09)
STRAW_WHITE = (0.92, 0.92, 0.90)
LEVER_ORANGE = (0.95, 0.45, 0.10)
SPRUCE = (0.82, 0.68, 0.48)
DRYWALL = (0.80, 0.80, 0.78)
COPPER = (0.78, 0.45, 0.25)


# -- USD helpers (after the accepted child-resistant-cap and rack-blanking-panel tasks) -------------------------------
def stage_for(name: str):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.usda"
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def material(stage, path: str, color, roughness=0.5, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(mat)


def quat_about(axis, angle: float) -> Gf.Quatf:
    s = math.sin(angle / 2)
    return Gf.Quatf(math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s)


def cube(stage, path, center, size, mat, orient: Gf.Quatf | None = None, collide=True):
    """A box of ``size`` centred at ``center`` (parent frame), turned by ``orient``."""
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*center))
    if orient is not None:
        c.AddOrientOp().Set(orient)
    c.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def box(stage, path, lo, hi, mat, collide=True):
    """An axis-aligned box from corner ``lo`` to corner ``hi``."""
    return cube(stage, path, [(a + b) / 2 for a, b in zip(lo, hi)], [b - a for a, b in zip(lo, hi)], mat,
                collide=collide)


def cylinder(stage, path, radius, height, center, mat, axis="Z", orient: Gf.Quatf | None = None, collide=True):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(axis)
    c.AddTranslateOp().Set(Gf.Vec3d(*center))
    if orient is not None:
        c.AddOrientOp().Set(orient)
    bind(c, mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def contact_patch(prim, radius: float) -> None:
    """Give a collider a torsional contact patch, so a pinch on it resists twisting about the line between the
    fingers (PhysX's default patch is a point, which lets a pinched can pivot freely)."""
    prim = prim.GetPrim()
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute("physxCollision:torsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(radius)
    prim.CreateAttribute("physxCollision:minTorsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(radius / 2)


def rigid_body(stage, path: str, pos=(0.0, 0.0, 0.0), orient: Gf.Quatf | None = None):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if orient is not None:
        xf.AddOrientOp().Set(orient)
    UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    return xf


# -- the can ------------------------------------------------------------------------------------------------------------
def build_can():
    stage, _ = stage_for("can")
    r = "/can"
    blue = material(stage, f"{r}/Looks/body", CAN_BLUE, roughness=0.35, metallic=0.3)
    silver = material(stage, f"{r}/Looks/silver", CAN_SILVER, roughness=0.3, metallic=0.8)
    black = material(stage, f"{r}/Looks/actuator", ADAPTER_BLACK, roughness=0.5)
    white = material(stage, f"{r}/Looks/straw", STRAW_WHITE, roughness=0.4)
    orange = material(stage, f"{r}/Looks/lever", LEVER_ORANGE, roughness=0.5)

    # the can's body, valve cup, actuator and straw: one rigid body, the articulation's floating root
    body = rigid_body(stage, f"{r}/body")
    prim = body.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    prim.AddAppliedSchema("PhysxArticulationAPI")
    prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("physxArticulation:stabilizationThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(g.CAN_MASS - g.LEVER_MASS)
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, g.CAN_COM_Z))
    ixx = g.CAN_MASS * (3 * g.CAN_R**2 + g.CAN_H**2) / 12            # a solid cylinder about its centre of mass
    izz = g.CAN_MASS * g.CAN_R**2 / 2
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(ixx, ixx, izz))
    contact_patch(cylinder(stage, f"{r}/body/can", g.CAN_R, g.CAN_H, (0, 0, g.CAN_H / 2), blue), g.GRIP_PATCH)
    cylinder(stage, f"{r}/body/rim", g.CAN_R + 0.0008, 0.006, (0, 0, g.CAN_H - 0.003), silver, collide=False)
    cylinder(stage, f"{r}/body/valve_cup", g.CUP_R, g.CUP_H, (0, 0, g.CAN_H + g.CUP_H / 2), silver)
    cylinder(stage, f"{r}/body/actuator", g.ACT_R, g.ACT_H, tuple(g.ADAPTER_C), black)
    straw_mid = g.STRAW_BASE + g.STRAW_DIR * g.STRAW_LEN / 2
    tilt = math.pi / 2 - g.STRAW_UP                                  # turn the cylinder's z onto the straw's line
    cylinder(stage, f"{r}/body/straw", g.STRAW_R, g.STRAW_LEN, tuple(straw_mid), white,
             orient=quat_about((0.0, 1.0, 0.0), tilt))

    # the trigger: a narrow lever under the straw, hinged on the actuator's side, LEVER_REST_DOWN below level at rest.
    # Its own frame: x out along the lever, z up from its top face; the joint turns it about +y, so a positive angle
    # tips its end further down toward the can.
    rest = quat_about((0.0, 1.0, 0.0), g.LEVER_REST_DOWN)           # about +y: the end tips down
    lever = rigid_body(stage, f"{r}/lever", tuple(g.LEVER_HINGE), rest)
    UsdPhysics.MassAPI.Apply(lever.GetPrim()).CreateMassAttr(g.LEVER_MASS)
    cube(stage, f"{r}/lever/strip", (g.LEVER_LEN / 2, 0.0, -g.LEVER_T / 2), (g.LEVER_LEN, g.LEVER_W, g.LEVER_T), orange)
    cube(stage, f"{r}/lever/plunger", (0.0025, 0.0, -0.004), (0.005, 0.006, 0.006), orange, collide=False)

    j = UsdPhysics.RevoluteJoint.Define(stage, f"{r}/joints/lever")
    j.CreateAxisAttr("Y")
    j.CreateBody0Rel().SetTargets([Sdf.Path(f"{r}/body")])
    j.CreateBody1Rel().SetTargets([Sdf.Path(f"{r}/lever")])
    j.CreateLocalPos0Attr(Gf.Vec3f(*g.LEVER_HINGE))
    j.CreateLocalRot0Attr(rest)
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(math.degrees(g.LEVER_TRAVEL))             # USD joint limits are in degrees
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)                                   # the task file's drive sets the spring
    drive.CreateDampingAttr(0.0)
    drive.CreateTargetPositionAttr(0.0)
    stage.GetRootLayer().Save()


# -- the wall -----------------------------------------------------------------------------------------------------------
def corner_fill(stage, path, half, hole_r, x, mat, n=48):
    """Visual only (no collider): a flat plate at depth ``x`` filling the square opening of half-width ``half``
    outside a round hole of radius ``hole_r``, so the face reads as a board with a round hole (behind it, the staves'
    corners would show as dark pockets)."""
    pts, faces = [], []
    for k in range(n):
        a = 2 * math.pi * k / n
        c, s = math.cos(a), math.sin(a)
        edge = half / max(abs(c), abs(s))                 # the ray at this angle meets the square here
        pts += [Gf.Vec3f(x, hole_r * c, hole_r * s), Gf.Vec3f(x, edge * c, edge * s)]
    for k in range(n):
        i, j = 2 * k, 2 * ((k + 1) % n)
        faces.append((i, i + 1, j + 1, j))
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([4] * n)
    m.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    m.CreateDoubleSidedAttr(True)
    m.CreateNormalsAttr([Gf.Vec3f(-1, 0, 0)] * len(pts))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    bind(m, mat)
    return m


def build_wall():
    stage, root = stage_for("wall")
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    r = "/wall"
    spruce = material(stage, f"{r}/Looks/spruce", SPRUCE, roughness=0.8)
    drywall = material(stage, f"{r}/Looks/drywall", DRYWALL, roughness=0.9)
    copper = material(stage, f"{r}/Looks/copper", COPPER, roughness=0.3, metallic=0.9)
    hw, z0 = g.WALL_W / 2, g.TOP - g.HOLE_Z                           # z0: the table, below the hole's centre
    z1 = z0 + g.WALL_H
    h, t = g.HOLE_R, g.RIM_T
    # the rim board around a square opening as wide as the hole; the staves round it off
    box(stage, f"{r}/board_left", (0, h, z0), (t, hw, z1), spruce)
    box(stage, f"{r}/board_right", (0, -hw, z0), (t, -h, z1), spruce)
    box(stage, f"{r}/board_top", (0, -h, h), (t, h, z1), spruce)
    box(stage, f"{r}/board_bottom", (0, -h, z0), (t, h, -h), spruce)
    n = g.BORE_STAVES
    width = 2 * h * math.tan(math.pi / n) * 1.05
    for k in range(n):
        a = 2 * math.pi * k / n
        rc = h + g.STAVE_T / 2
        cube(stage, f"{r}/bore_{k:02d}", (t / 2, rc * math.cos(a), rc * math.sin(a)), (t, g.STAVE_T, width), spruce,
             orient=quat_about((1.0, 0.0, 0.0), a))
    corner_fill(stage, f"{r}/corner_fill", h, h + g.STAVE_T / 2, 0.001, spruce)
    box(stage, f"{r}/backer", (t, -hw, z0), (t + g.BACKER_T, hw, z1), drywall)
    x0, x1 = -g.PIPE_STUB, t + g.BACKER_T + g.PIPE_BEHIND
    cylinder(stage, f"{r}/pipe", g.PIPE_R, x1 - x0, ((x0 + x1) / 2, 0, 0), copper, axis="X")
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    build_can()
    build_wall()
    for name in ("can", "wall"):
        print(f"wrote {OUT / (name + '.usda')}")
