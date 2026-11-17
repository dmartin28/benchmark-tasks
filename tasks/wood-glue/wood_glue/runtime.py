"""Task-local USD construction and YAM hooks. No installed simulator files are modified."""
from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys

import numpy as np

from . import _toolkit
from .geometry import Pose, rotation
from .observable import ObservableAssemblyModel, evaluate_hold
from .scene import (BOTTLE_BASE, HEAD, HOLDER_XY, LEG, Miter, OBJECTS, RISER,
                    THICK, TOP, all_initial_poses, surfaces)

_SPECS = {}
_MODEL = None
_ACCELERATED_BOND = False


def configure(accelerated_bond=False):
    global _ACCELERATED_BOND
    _ACCELERATED_BOND = bool(accelerated_bond)


def current():
    return _MODEL


def arr(value):
    value = getattr(value, "torch", value)
    return value.detach().cpu().numpy().astype(float)


def _spawn_miter(path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.miter_name]
    stage = sim.get_current_stage()
    root = UsdGeom.Xform.Define(stage,path)
    if translation is not None:
        root.AddTranslateOp().Set(Gf.Vec3d(*map(float,translation)))
    if orientation is not None:
        x,y,z,w = map(float,orientation)
        root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w,Gf.Vec3d(x,y,z)))
    vertices, faces, _ = spec.mesh()
    mesh = UsdGeom.Mesh.Define(stage,path+"/wood")
    mesh.CreatePointsAttr(vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*linear_rgb(spec.color))])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    collision.CreateContactOffsetAttr(.001)
    collision.CreateRestOffsetAttr(0.)
    material = sim.RigidBodyMaterialCfg(static_friction=.5,dynamic_friction=.4)
    material.func(path+"/friction",material)
    sim.bind_physics_material(path+"/wood",path+"/friction",stage=stage)
    if spec.free:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(spec.mass)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
        rb.CreateSolverPositionIterationCountAttr(32)
        rb.CreateMaxDepenetrationVelocityAttr(1.)
    return root.GetPrim()


def _miter_object(spec):
    from isaaclab.sim import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose as ArenaPose

    @configclass
    class MiterCfg(SpawnerCfg):
        func: object = clone(_spawn_miter)
        miter_name: str = ""

    _SPECS[spec.name] = spec
    return Object(name=spec.name,object_type=ObjectType.RIGID if spec.free else ObjectType.BASE,
                  spawner_cfg=MiterCfg(miter_name=spec.name),
                  initial_pose=ArenaPose(position_xyz=spec.pos,
                                        rotation_xyzw=(0.,0.,math.sin(spec.yaw/2),math.cos(spec.yaw/2))))


def _read_miter(env,spec):
    from yam_arena.ik import matrix_to_quat_xyzw
    if spec.free:
        data = env.unwrapped.scene[spec.name].data
        p = arr(data.root_link_pose_w)[0]
        origin = arr(env.unwrapped.scene.env_origins)[0]
        pose = Pose(p[:3]-origin,rotation(p[3:7]),arr(data.root_lin_vel_w)[0],arr(data.root_ang_vel_w)[0])
    else:
        pose = all_initial_poses()[spec.name]
    verts = pose.world(spec.mesh()[0])
    return _toolkit.ToolState(pose.pos,matrix_to_quat_xyzw(pose.rot),pose.linear,pose.angular,verts.min(0),verts.max(0),
                              sliders={},named={"origin":pose.pos,"rotation":pose.rot})


def _patch_scene(module):
    if getattr(module,"_wood_glue_meshes",False):
        return
    original_object,original_states = module.task_object,module.object_states

    def task_object(spec):
        return _miter_object(spec) if isinstance(spec,Miter) else original_object(spec)

    def object_states(env,names):
        normal = [n for n in names if n not in _SPECS]
        out = original_states(env,normal) if normal else {}
        out.update({n:_read_miter(env,_SPECS[n]) for n in names if n in _SPECS})
        return {n:out[n] for n in names}

    module.task_object,module.object_states = task_object,object_states
    module._wood_glue_meshes = True


def measured_poses(emb,refresh=False):
    # All consumers within one control interval use the same measured snapshot. Refresh only
    # immediately after native physics/reset; repeated camera/controller/scorer reads cannot advance it.
    if not refresh and hasattr(emb,"_wood_glue_snapshot"):
        return emb._wood_glue_snapshot
    from yam_arena.scene import object_states
    states = object_states(emb._env,emb.task.names)
    poses = all_initial_poses()
    for name in (HEAD.name,LEG.name):
        state = states[name]
        poses[name] = Pose(state.named["origin"],state.named["rotation"],state.lin_vel,state.ang_vel)
    bottle = states["glue_bottle"]
    poses["glue_bottle"] = Pose(bottle.pos,bottle.rotation(),bottle.lin_vel,bottle.ang_vel)
    emb._wood_glue_snapshot = poses,states
    return emb._wood_glue_snapshot


def hand_bounds(emb):
    """Conservative bounds of actual finger/gripper collision hulls, from measured joints."""
    from yam_arena import station,YAM_URDF
    from yam_arena.analysis import world_hulls
    from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS
    from yam_arena.urdf import forward_kinematics
    out = []
    for side in ("left","right"):
        robot = emb._base.scene[side+"_robot"]
        q = arr(robot.data.joint_pos)[0]
        lookup = dict(zip(robot.joint_names,q))
        transforms = forward_kinematics(YAM_URDF,{n:lookup[n] for n in ARM_JOINTS+FINGER_JOINTS})
        for transform in transforms.values():
            transform[:3,3] += station.BASES[side]
        hulls = world_hulls(transforms,["gripper","tip_left","tip_right"])
        vertices = np.concatenate([h.vertices for _,h in hulls])
        out.append((vertices.min(0),vertices.max(0)))
    return out


def aabb_distance(a,b):
    return float(np.linalg.norm(np.maximum(np.maximum(a[0]-b[1],b[0]-a[1]),0.)))


def endpoint(poses,states,hands):
    bottle = poses["glue_bottle"]
    parked = (np.max(np.abs(bottle.pos[:2]-HOLDER_XY)) <= .004 and
              abs(bottle.pos[2]-BOTTLE_BASE[2]) <= .004 and
              bottle.rot[2,2] >= math.cos(math.radians(10)) and bottle.still())
    targets = [(states[n].low,states[n].high) for n in (HEAD.name,LEG.name,"glue_bottle")]
    clear = all(aabb_distance(h,t) >= .030 for h in hands for t in targets)
    wood_clearance = min(aabb_distance(h,t) for h in hands for t in targets[:2])
    return bool(clear),bool(parked),float(wood_clearance*1000)


class FilmRenderer:
    """Non-colliding wet-film meshes parented to the owning rigid body."""
    def __init__(self,emb,model):
        from pxr import UsdGeom
        self.stage = emb._base.sim.stage
        self.roots = {}
        for spec in OBJECTS:
            found = [p for p in self.stage.Traverse() if p.GetName()==spec.name and "env_0" in str(p.GetPath())]
            if not found:
                raise RuntimeError(f"missing USD root for {spec.name}")
            self.roots[spec.name] = str(found[0].GetPath())
        self.roots["bench"] = "/World/WoodGlueBenchMarks"
        UsdGeom.Xform.Define(self.stage,self.roots["bench"])
        self.meshes = {}
        for name,film in model.films.items():
            mesh = UsdGeom.Mesh.Define(self.stage,self.roots[film.surface.owner]+"/wet_glue")
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            self.meshes[name] = mesh
        self.spills = {}
        self._last_signature = None

    @staticmethod
    def _quads(mesh,centres,u,v,du,dv,colors):
        from pxr import Gf
        centres = np.asarray(centres).reshape(-1,3)
        corners = np.array([-.5*np.asarray(u)*du-.5*np.asarray(v)*dv,
                            .5*np.asarray(u)*du-.5*np.asarray(v)*dv,
                            .5*np.asarray(u)*du+.5*np.asarray(v)*dv,
                           -.5*np.asarray(u)*du+.5*np.asarray(v)*dv])
        points = (centres[:,None,:]+corners).reshape(-1,3)
        mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
        mesh.GetFaceVertexCountsAttr().Set([4]*len(centres))
        mesh.GetFaceVertexIndicesAttr().Set(list(range(len(points))))
        attr = mesh.CreateDisplayColorAttr([Gf.Vec3f(*c) for c in colors])
        from pxr import UsdGeom
        UsdGeom.PrimvarsAPI(mesh).GetPrimvar("displayColor").SetInterpolation("uniform")
        return attr

    def update(self,model):
        signature = (tuple(f.volume.tobytes() for f in model.films.values()),len(model.marks))
        if signature == self._last_signature:
            return False
        self._last_signature = signature
        from pxr import UsdGeom
        from yam_arena.station import linear_rgb
        for name,film in model.films.items():
            wet = film.volume > 1e-16
            heights = film.thickness[wet]
            centres = film.points[wet]+film.surface.normal*(heights[:,None]+.00003)
            colors = [linear_rgb((.98,.91,.55) if h <= model.settings.max_film else (.98,.72,.20)) for h in heights]
            self._quads(self.meshes[name],centres,film.surface.u,film.surface.v,*film.pitch,colors)
        surfaces_by_name = {s.name:s for s in model.surfaces}
        for key,(owner,p) in model.marks.items():
            if key in self.spills:
                continue
            surface = surfaces_by_name[key[0]]
            root = self.roots.get(owner,self.roots["bench"])
            mesh = UsdGeom.Mesh.Define(self.stage,root+f"/spill_{len(self.spills)}")
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            self._quads(mesh,[p+surface.normal*.0001],surface.u,surface.v,.001,.001,[linear_rgb((.98,.83,.3))])
            self.spills[key] = mesh
        return True

    def clear(self):
        for mesh in self.spills.values():
            self.stage.RemovePrim(mesh.GetPath())
        self.spills.clear()


def _is_task(emb):
    return bool(emb.task and getattr(emb.task.module,"WOOD_GLUE_TASK",False))


def _patch_embodiment(module):
    cls = module.YamArenaEmbodiment
    if getattr(cls,"_wood_glue_runtime",False):
        return
    original_step,original_reset = cls._task_step,cls.reset

    def reset(self,*args,**kwargs):
        global _MODEL
        if hasattr(self,'_wood_glue_bond'):
            self._wood_glue_bond.clear()
            del self._wood_glue_bond
        result = original_reset(self,*args,**kwargs)
        if _is_task(self):
            if hasattr(self,"_wood_glue_renderer"):
                self._wood_glue_renderer.clear()
            self._wood_glue_model = ObservableAssemblyModel(surfaces())
            _MODEL = self._wood_glue_model
            poses,states = measured_poses(self,refresh=True)
            bottle = states["glue_bottle"]
            compression = -bottle.sliders["panel_left"]+bottle.sliders["panel_right"]
            _MODEL.previous = (bottle.point("tip"),bottle.rotation(),compression)
            self._wood_glue_tick = 0
            self._wood_glue_renderer = FilmRenderer(self,_MODEL)
            # A painted parking outline has no physical support/holder geometry.
            from pxr import UsdGeom,UsdShade,Gf,Sdf
            stage=self._base.sim.stage
            mesh = UsdGeom.Mesh.Define(stage,'/World/WoodGlueBottleHome')
            points,counts,indices = [],[],[]
            for axis in (0,1):
                for sign in (-1,1):
                    middle = BOTTLE_BASE.copy()
                    middle[2] = TOP+.00003
                    middle[axis] += sign*.05
                    u,v = np.zeros(3),np.zeros(3)
                    u[axis] = .001
                    v[1-axis] = .05
                    points.extend([Gf.Vec3f(*p) for p in (middle-u-v,middle+u-v,middle+u+v,middle-u+v)])
                    indices.extend(range(len(points)-4,len(points)))
                    counts.append(4)
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr(counts)
            mesh.CreateFaceVertexIndicesAttr(indices)
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(.9,.9,.9)])
            # Give the mark its own material instead of inheriting the blue
            # cover material; points are in World coordinates, not a Box root.
            material=UsdShade.Material.Define(stage,'/World/WoodGlueHomeMaterial')
            shader=UsdShade.Shader.Define(stage,'/World/WoodGlueHomeMaterial/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.9,.9,.9))
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.8)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
            if _ACCELERATED_BOND:
                from .bond import AdhesiveBond
                self._wood_glue_bond = AdhesiveBond(self)
            if self._wood_glue_renderer.update(_MODEL):
                self._base.sim.render()
            result = self._observation()
        return result

    def task_step(self,*args,**kwargs):
        global _MODEL
        if _is_task(self):
            _MODEL = self._wood_glue_model
            poses,states = measured_poses(self,refresh=True)
            bottle = states["glue_bottle"]
            compression = -bottle.sliders["panel_left"]+bottle.sliders["panel_right"]
            if poses['glue_bottle'].pos[2] < TOP-.05:
                _MODEL.invalidate("bottle_off_table")
            _MODEL.advance(self._wood_glue_tick,self._base.step_dt,bottle.point("tip"),bottle.rotation(),
                           compression,poses)
            held_status = evaluate_hold(self,poses,states,_MODEL)
            _MODEL.held_corner = held_status
            _MODEL.hand_wood_clearance_mm = 0. if any(h['holding'] for h in held_status['hands'].values()) else None
            _MODEL.joint_step(self._wood_glue_tick,self._base.step_dt,poses,False,
                              held_status['bottle_parked'],
                              bonded=hasattr(self,'_wood_glue_bond') and self._wood_glue_bond.alive,
                              held=held_status['qualified'])
            if hasattr(self,'_wood_glue_bond'):
                self._wood_glue_bond.update(_MODEL,poses,self._base.step_dt)
                _MODEL.bonded = self._wood_glue_bond.alive
            self._wood_glue_tick += 1
            if self._wood_glue_renderer.update(_MODEL):
                self._base.sim.render()
        result = original_step(self,*args,**kwargs)
        if _is_task(self):
            result.info.update(_MODEL.metrics())
            if hasattr(self,'_wood_glue_bond'):
                result.info['adhesive_bond'] = self._wood_glue_bond.metrics()
        return result

    cls.reset,cls._task_step = reset,task_step
    cls._wood_glue_runtime = True


class _Finder(importlib.abc.MetaPathFinder):
    targets = {"yam_arena.scene":_patch_scene,"yam_arena.inspect_embodiment":_patch_embodiment}

    def find_spec(self,fullname,path,target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0,self)
        if spec is None or spec.loader is None:
            return None
        original_exec,patch = spec.loader.exec_module,self.targets[fullname]
        def execute(module):
            original_exec(module)
            patch(module)
        spec.loader.exec_module = execute
        return spec


def install():
    # Toolkit first, then our mesh and runtime wrappers. Both are confined to this process.
    _toolkit.install()
    for name,patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f,_Finder) for f in sys.meta_path):
        sys.meta_path.insert(0,_Finder())
