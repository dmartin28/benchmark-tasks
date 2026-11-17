"""Countertop oily-waste can (Justrite 09200) with a gravity-closed hinged lid.

Two rigid bodies in one collection: the can shell (kinematic, never moves) and
the lid (dynamic). A revolute joint at the back rim joins them; its limits are
0 (shut) to 60 degrees (the maker's stated maximum). No spring: the lid's own
weight shuts it, as on the real can. Nothing in software opens, holds or shuts
the lid; only contact moves it.

Pattern adapted from the accepted task `peg-out-a-tshirt` (arena_props.py, MIT
licence, see LICENSE in this folder). Geometry is original.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState

# Justrite 09200: 232 mm tall overall, 244 mm outer diameter (maker's datasheet).
BODY_R = 0.117          # shell outer radius
WALL = 0.005            # collider thickness (real steel ~0.6 mm; thicker for stable contact)
BODY_H = 0.218          # shell height to the rim
LID_R = 0.122           # lid overhangs the rim: 244 mm overall
LID_T = 0.010
TAB = (0.030, 0.050, 0.004)   # front lift tab, past the lid edge (x out, y wide, z thick)
TAB_LIP = 0.012               # the tab's turned-up front lip height
LID_MASS = 0.35
OPEN_LIMIT_DEG = 60.0
STAVES = 24
RED = (0.72, 0.08, 0.06)
LABEL_Z = (0.075, 0.175)      # the warning label's bottom and top edges above the can's floor
LABEL_SPAN_DEG = 112          # about 230 mm of the can's circumference


@dataclass(frozen=True)
class OilyWasteCan(Box):
    """`pos` is the centre of the can's floor on the table; `size` is only the bounding box.

    `label_angle` (rad about z) is where the yellow warning label is centred on the can's side."""
    label_angle: float = math.pi

    @property
    def free(self):
        return False

    def _check(self):
        super()._check()
        if self.yaw != 0.0:
            raise ValueError("can yaw is fixed: hinge at +x (far side), tab at -x (robot side)")


@dataclass(frozen=True)
class CanState(ObjectState):
    lid_angle_deg: float = 0.0
    lid_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))


SPECS = {}


def hinge_local():
    """Hinge point in the can frame (origin at the floor centre): top of the back rim."""
    return np.array([LID_R, 0.0, BODY_H + LID_T / 2])


def _box(stage, path, size, pos, color, rot_z=0.0):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rot_z:
        cube.AddRotateZOp().Set(math.degrees(rot_z))
    cube.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cp = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
    cp.CreateContactOffsetAttr(0.002)
    cp.CreateRestOffsetAttr(0.0)
    return cube


def _cyl(stage, path, radius, height, pos, color):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr("Z")
    c.AddTranslateOp().Set(Gf.Vec3d(*pos))
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    cp = PhysxSchema.PhysxCollisionAPI.Apply(c.GetPrim())
    cp.CreateContactOffsetAttr(0.002)
    cp.CreateRestOffsetAttr(0.0)
    return c


def _look(stage, prim, path, color, roughness=0.45, metallic=0.2):
    from pxr import Gf, Sdf, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _friction(stage, prim, path, mu=0.5):
    from pxr import UsdPhysics, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(mu)
    pm.CreateDynamicFrictionAttr(mu)
    pm.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def author_body(stage, path, pos, label_angle=math.pi):
    """Open steel shell: 24 wall staves and a floor, as one kinematic body."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*pos))
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(2.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    r_mid = BODY_R - WALL / 2
    width = 2 * BODY_R * math.tan(math.pi / STAVES) + 0.002
    for i in range(STAVES):
        a = 2 * math.pi * i / STAVES
        _box(stage, f"{path}/stave{i:02d}", (WALL, width, BODY_H),
             (r_mid * math.cos(a), r_mid * math.sin(a), BODY_H / 2), RED, rot_z=a)
    _cyl(stage, f"{path}/floor", BODY_R, 0.006, (0, 0, 0.003), RED)
    _look(stage, prim, f"{path}/Look", RED)
    # Printed warning label on the side (visual only), just outside the staves' corners.
    from props import HERE, _look as look, label_band
    band = label_band(stage, f"{path}/label", BODY_R + 0.0015, LABEL_Z[0], LABEL_Z[1], label_angle, LABEL_SPAN_DEG)
    look(stage, band.GetPrim(), f"{path}/label/Look", None, roughness=0.5, texture=HERE / "oily_waste_label.png")
    _friction(stage, prim, f"{path}/Friction")
    return prim


def author_lid(stage, path, pos):
    """Lid disc with a front lift tab and turned-up lip; dynamic, gravity-closed."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*pos))
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(LID_MASS)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateSolverPositionIterationCountAttr(32)
    rb.CreateSolverVelocityIterationCountAttr(8)
    # Lid frame origin: centre of the lid disc when shut.
    _cyl(stage, f"{path}/disc", LID_R, LID_T, (0, 0, 0), RED)
    tx = -(LID_R + TAB[0] / 2 - 0.004)
    _box(stage, f"{path}/tab", TAB, (tx, 0, -LID_T / 2 + TAB[2] / 2), (0.62, 0.62, 0.60))
    _box(stage, f"{path}/tab_lip", (0.004, TAB[1], TAB_LIP),
         (tx - TAB[0] / 2 + 0.002, 0, -LID_T / 2 + TAB_LIP / 2), (0.62, 0.62, 0.60))
    _look(stage, prim, f"{path}/Look", RED)
    _friction(stage, prim, f"{path}/Friction")
    return prim


def author_hinge(stage, path, body_path, lid_path):
    from pxr import Gf, Sdf, UsdPhysics
    h = hinge_local()
    j = UsdPhysics.RevoluteJoint.Define(stage, path + "/hinge")
    j.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(lid_path)])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(*h))
    j.CreateLocalPos1Attr(Gf.Vec3f(LID_R, 0.0, 0.0))
    # +Y rotation lifts the front (-x) edge.
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(OPEN_LIMIT_DEG)
    j.CreateCollisionEnabledAttr(False)
    return j


def _spawn(path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.utils.stage import get_current_stage
    stage = get_current_stage()
    spec = SPECS[cfg.can_name]
    root = str(path).rsplit("/", 1)[0]
    floor = np.asarray(translation if translation is not None else spec.pos, float)
    if cfg.part == "body":
        return author_body(stage, str(path), floor, spec.label_angle)
    prim = author_lid(stage, str(path), floor + [0, 0, BODY_H + LID_T / 2])
    author_hinge(stage, root, root + "/body", str(path))
    return prim


def reset_can(env, env_ids, can_name):
    import torch
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    obj = base.scene[can_name]
    origins = getattr(base.scene.env_origins, "torch", base.scene.env_origins)
    ids = torch.arange(len(origins), device=origins.device, dtype=torch.long) if env_ids is None else env_ids
    spec = SPECS[can_name]
    floor = torch.as_tensor(spec.pos, device=origins.device, dtype=origins.dtype)
    pose = torch.zeros((len(ids), 2, 7), device=origins.device, dtype=origins.dtype)
    pose[:, 0, :3] = floor + origins[ids]
    pose[:, 1, :3] = floor + origins[ids]
    pose[:, 1, 2] += BODY_H + LID_T / 2
    pose[:, :, 6] = 1.0   # identity, xyzw
    obj.write_body_link_pose_to_sim_index(body_poses=pose, env_ids=ids)
    obj.write_body_link_velocity_to_sim_index(
        body_velocities=torch.zeros((len(ids), 2, 6), device=origins.device, dtype=origins.dtype), env_ids=ids)


def can_asset(spec):
    from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.managers import EventTermCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.asset import Asset
    spec._check()
    SPECS[spec.name] = spec

    @configclass
    class SpawnCfg(SpawnerCfg):
        func: object = clone(_spawn)
        part: str = "body"
        can_name: str = ""

    class CanAsset(Asset):
        def __init__(self):
            super().__init__(spec.name, tags=["oily-waste-can", "hinged-lid", "physx"])
            bodies = {}
            for part, dz in (("body", 0.0), ("lid", BODY_H + LID_T / 2)):
                bodies[part] = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/" + spec.name + "/" + part,
                    spawn=SpawnCfg(part=part, can_name=spec.name),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(spec.pos[0], spec.pos[1], spec.pos[2] + dz),
                                                              rot=(0.0, 0.0, 0.0, 1.0)))
            self.cfg = RigidObjectCollectionCfg(rigid_objects=bodies)

        def get_object_cfg(self):
            return self.name, self.cfg

        def get_event_cfg(self):
            return self.name, EventTermCfg(func=reset_can, mode="reset", params={"can_name": self.name})

        def validate_simulation_cfg(self, sim_cfg):
            from isaaclab_physx.physics import PhysxCfg
            if sim_cfg.physics is not None and not isinstance(sim_cfg.physics, PhysxCfg):
                raise ValueError("The hinged can requires PhysX")

    return CanAsset()


def lid_angle_deg(quat_xyzw):
    """Lid opening angle: rotation of the lid about the hinge's +Y axis."""
    x, y, z, w = quat_xyzw
    # Pitch about Y from the lid's rotation matrix: R[0,2] = sin(theta).
    r02 = 2 * (x * z + y * w)
    r22 = 1 - 2 * (x * x + y * y)
    return math.degrees(math.atan2(r02, r22))


def can_state(env, name):
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    obj = base.scene[name]
    names = list(getattr(obj, "body_names", getattr(obj, "object_names", ())))
    i_body, i_lid = names.index("body"), names.index("lid")

    def read(*attrs):
        for attr in attrs:
            if hasattr(obj.data, attr):
                v = getattr(obj.data, attr)
                return getattr(v, "torch", v)[0].detach().cpu().numpy().astype(float)
        raise AttributeError(attrs)

    pos = read("body_link_pos_w", "object_link_pos_w")
    quat = read("body_link_quat_w", "object_link_quat_w")
    origin = getattr(base.scene.env_origins, "torch", base.scene.env_origins)[0].detach().cpu().numpy()
    pos = pos - origin
    floor = pos[i_body]
    lo = floor + [-LID_R - TAB[0], -LID_R, 0.0]
    hi = floor + [LID_R, LID_R, BODY_H + LID_T]
    # Isaac Lab 3.0 pose quaternions are already xyzw (isaaclab.utils.math docstrings).
    q = np.asarray(quat[i_lid], float)
    return CanState(pos=(lo + hi) / 2, quat_xyzw=np.array((0., 0., 0., 1.)), lin_vel=np.zeros(3),
                    ang_vel=np.zeros(3), low=lo, high=hi, lid_angle_deg=lid_angle_deg(q),
                    lid_pose=np.r_[pos[i_lid], q])


def register():
    import cloth_bridge
    if not any(cls is OilyWasteCan for cls, _ in cloth_bridge.EXTRA_FACTORIES):
        cloth_bridge.EXTRA_FACTORIES.append((OilyWasteCan, _factory))


def _factory(spec):
    import cloth_bridge
    cloth_bridge.EXTRA_STATES[spec.name] = can_state
    return can_asset(spec)
