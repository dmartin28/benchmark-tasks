"""Load original scene assets and inspection cameras into the standard station.

Process-local adapters only; the installed SDK and robot actions are unchanged.
"""
import copy
import importlib.abc
import importlib.util
import sys
import numpy as np
from work_layout import BOX_INITIAL_ROT_XYZW,CHANNEL_ROT_XYZW
import work_layout as layout
import iberville_assets as a

FOLDER=None
BOX=None
FRAME=None
FRAME_ASSET="stud_frame.usda"
BUILDING=False
PLACEMENT_RUN=False
FORMING_YIELD_SCALE=None
FORMING_DAMPING=.004
BOX_ASSET_OVERRIDE=None
PHYSICS_DEVICE_OVERRIDE=None


def asset(spec):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose
    import isaaclab.sim as su
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    rotation=BOX_INITIAL_ROT_XYZW if isinstance(spec,BOX) else CHANNEL_ROT_XYZW
    pose=Pose(position_xyz=tuple(spec.pos),rotation_xyzw=rotation)
    if isinstance(spec,FRAME):
        return Object(name=spec.name,object_type=ObjectType.BASE,
                      usd_path=str(FOLDER/'assets'/FRAME_ASSET),initial_pose=pose)
    class Device(Object):
        def _generate_articulation_cfg(self):
            cfg=ArticulationCfg(prim_path=self.prim_path,
                spawn=self._get_spawn_cfg(activate_contact_sensors=True),
                init_state=ArticulationCfg.InitialStateCfg(joint_pos={'.*_fold_.*':0.}),
                actuators={'folds':ImplicitActuatorCfg(joint_names_expr=['.*_fold_.*'],
                    stiffness={f'{side}_fold_{k}':a.fold_stiffness(k)
                        for side in a.STRAP_ROOTS for k in range(len(a.FOLD_STATIONS)-1)},
                    damping=.004,armature=.000005,joint_effort_limit=100)})
            if FORMING_YIELD_SCALE is not None:
                from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg
                cfg.spawn.articulation_props=PhysxArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,solver_position_iteration_count=32,solver_velocity_iteration_count=4)
                from flap_material_native import PlasticFlapActuatorCfg
                cfg.actuators['folds']=PlasticFlapActuatorCfg(
                    joint_names_expr=['.*_fold_.*'],
                    stiffness={f'{side}_fold_{k}':a.fold_stiffness(k)
                        for side in a.STRAP_ROOTS for k in range(len(a.FOLD_STATIONS)-1)},
                    damping=FORMING_DAMPING,armature=.000005,joint_effort_limit=100,
                    yield_scale=FORMING_YIELD_SCALE)
            return self._add_initial_pose_to_cfg(cfg)
    return Device(name=spec.name,object_type=ObjectType.ARTICULATION,
        usd_path=str(BOX_ASSET_OVERRIDE or FOLDER/'assets/device_box.usda'),initial_pose=pose,
        spawn_cfg_addon={'physics_material':su.RigidBodyMaterialCfg(static_friction=.5,dynamic_friction=.5)})


def patch(scene):
    if getattr(scene,'_iberville_scene_preview',False):return
    scene._iberville_scene_preview=True
    old_object=scene.task_object
    scene.task_object=lambda spec: asset(spec) if isinstance(spec,(BOX,FRAME)) else old_object(spec)
    old_config=scene.env_cfg_callback
    def config_factory(hz=None):
        configure=old_config(hz)
        if not BUILDING:return configure
        def config(cfg):
            from isaaclab.sensors import CameraCfg
            import isaaclab.sim as su
            from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg
            from yam_arena.ik import matrix_to_quat_xyzw
            cfg=configure(cfg);cfg.sim.dt=.001;cfg.decimation=round(1/(.001*(hz or 20)))
            if PHYSICS_DEVICE_OVERRIDE is not None:cfg.sim.device=PHYSICS_DEVICE_OVERRIDE
            cfg.sim.render_interval=cfg.decimation
            if FORMING_YIELD_SCALE is not None:
                cfg.sim.use_newton_actuators=False
                cfg.episode_length_s=240.
            for side in ['left','right']:
                robot=getattr(cfg.scene,side+'_robot')
                robot.spawn.collision_props=PhysxCollisionPropertiesCfg(contact_offset=.0002,rest_offset=0)
                robot.actuators['gripper'].joint_effort_limit=40.
                if PLACEMENT_RUN:
                    from isaaclab.sensors import ContactSensorCfg
                    setattr(cfg.scene,side+'_contacts',ContactSensorCfg(
                        prim_path='{ENV_REGEX_NS}/'+('LeftArm' if side=='left' else 'RightArm')+'/.*',
                        history_length=50,update_period=.001))
            bp=np.asarray(layout.BOX_INITIAL_POSITION);cr=np.asarray(layout.CHANNEL_POSITION)
            box_center=bp+layout.BOX_INITIAL_ROTATION@np.array([a.DEPTH/2,0,0])
            views={
                'channel_detail':(cr+layout.CHANNEL_ROTATION@np.array([-.09,.16,a.STUD_HEIGHT+.20]),
                                  cr+layout.CHANNEL_ROTATION@np.array([a.STUD_DEPTH/2,a.STUD_FLANGE/2,a.STUD_HEIGHT-.015]),42.),
                'overview':((-.10,0,1.80),(.29,0,.98),20.),
                'work_area':((-.08,0,1.40),(.29,0,.94),26.),
                'box_detail':(box_center+layout.BOX_INITIAL_ROTATION@np.array([-.16,-.12,.18]),box_center,42.),
                'mount_side_detail':(box_center+layout.BOX_INITIAL_ROTATION@np.array([-.09,.17,.13]),box_center+layout.BOX_INITIAL_ROTATION@np.array([0,.02,0]),42.),
                'support_side_detail':(box_center+layout.BOX_INITIAL_ROTATION@np.array([-.08,-.17,.13]),box_center,42.),
            }
            if PLACEMENT_RUN:
                views={key:value for key,value in views.items() if key in ['work_area']}
            for name,(eye,target,focal) in views.items():
                eye=np.array(eye);forward=np.array(target)-eye;forward/=np.linalg.norm(forward)
                right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right)
                quat=matrix_to_quat_xyzw(np.column_stack([right,down,forward]))
                setattr(cfg.scene,name,CameraCfg(prim_path='{ENV_REGEX_NS}/'+name,
                    height=800,width=1200,data_types=['rgb'],
                    spawn=su.PinholeCameraCfg(focal_length=focal,horizontal_aperture=30.,clipping_range=(.01,10.)),
                    offset=CameraCfg.OffsetCfg(pos=tuple(eye),rot=tuple(quat),convention='ros'),
                    renderer_cfg=copy.deepcopy(cfg.scene.top_cam.renderer_cfg)))
            return cfg
        return config
    scene.env_cfg_callback=config_factory
    old_make=scene.make_env
    def make_env(*a,**kw):
        global BUILDING
        task=kw.get('task');previous=BUILDING
        BUILDING=task is not None and task.path.parent.resolve()==FOLDER.resolve()
        try:return old_make(*a,**kw)
        finally:BUILDING=previous
    scene.make_env=make_env


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if fullname!='yam_arena.scene':return None
        sys.meta_path.remove(self);spec=importlib.util.find_spec(fullname)
        execute=spec.loader.exec_module
        def load(module):execute(module);patch(module)
        spec.loader.exec_module=load
        return spec


def install(folder,box,frame,frame_asset='stud_frame.usda'):
    global FOLDER,BOX,FRAME,FRAME_ASSET
    FOLDER=folder;BOX=box;FRAME=frame;FRAME_ASSET=frame_asset
    if 'yam_arena.scene' in sys.modules:patch(sys.modules['yam_arena.scene'])
    elif not any(isinstance(x,Finder) for x in sys.meta_path):sys.meta_path.insert(0,Finder())
