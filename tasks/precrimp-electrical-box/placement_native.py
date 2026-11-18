"""Execute the placement controller through ordinary SDK actions in Isaac.

MuJoCo data here are a read-only kinematic mirror used for IK/Jacobians and
logging. Production trials send only normal 14-D actions. An explicitly marked
development fixture may restore its initial state before the first policy step;
that path cannot count as a continuous pickup or installation demonstration.
"""
import atexit
from pathlib import Path
import mujoco
import numpy as np
from yam_arena import ik
from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS,FINGER_OPEN
import install_physics as p


class NativePhysics(p.Physics):
    def __init__(self,model,moments,args):
        super().__init__(model,moments)
        import work_layout as w
        from place_box import INITIAL_POSITION,INITIAL_ROTATION
        # Initial scene configuration, before native simulation exists.
        w.BOX_INITIAL_POSITION=tuple(float(v) for v in INITIAL_POSITION)
        w.BOX_INITIAL_ROTATION=INITIAL_ROTATION.copy()
        w.BOX_INITIAL_ROT_XYZW=tuple(float(v) for v in ik.matrix_to_quat_xyzw(INITIAL_ROTATION))
        from yam_arena.inspect_embodiment import YamArenaEmbodiment
        from yam_arena import inspect_embodiment as ie
        from inspect_robots import Scene
        from isaaclab.app import AppLauncher
        folder=Path(__file__).parent
        native_device=getattr(args,'physics_device',None) or 'cpu'
        self.emb=YamArenaEmbodiment(task=str(folder/'scene_u.py'),device=native_device,control_hz=20,
            object_jitter=0,cam_width=320,cam_height=180)
        import iberville_runtime
        iberville_runtime.PLACEMENT_RUN=True
        iberville_runtime.PHYSICS_DEVICE_OVERRIDE=getattr(args,'physics_device',None)
        if getattr(args,'forming_material',False):
            iberville_runtime.FORMING_YIELD_SCALE=args.yield_scale
            iberville_runtime.FORMING_DAMPING=getattr(args,'flap_damping',.004)
            iberville_runtime.BOX_ASSET_OVERRIDE=getattr(args,'forming_box_asset',None)
        ie._APP=AppLauncher(headless=True,device=native_device,enable_cameras=True,
            kit_args=f'--/renderer/activeGpu={args.renderer_gpu} --/renderer/multiGpu/enabled=false --/carb/tasking/plugin/threadCount=4 --/omni/tbb/globalcontrol/maxThreadCount=4').app
        atexit.register(ie._close_app_at_exit)
        instruction=('Hold the electrical box against the stud and precrimp both mounting flaps around its flanges.' if getattr(args,'precrimp_mode',False) else
            'Hold the box on the stud and fold both mounting flaps around the channel.' if getattr(args,'forming_material',False) else
            'Pick up the box and hold it seated on the stud; leave the flaps flat.')
        self.emb.reset(Scene(id='box-precrimp' if getattr(args,'precrimp_mode',False) else 'box-placement',instruction=instruction))
        # The SDK's generic rigid-object scorer cannot query this articulated box.
        # The scripted controller reads native state directly. Unrecorded
        # development runs need no camera observations; fetching those pixels
        # otherwise renders the cameras even when recording is disabled.
        # Ordinary SDK action/physics stepping remains unchanged.
        # Retain the task's live, geometry-based articulation scorer.
        self.base=self.emb._base;self.native_time=0.;self.writers={}
        self.force_history=[];self.capture_force_history=getattr(args,'forming_material',False)
        self.all_force_history=getattr(args,'diagnostic_force_warning',False)
        self.output=Path(args.output)
        self.native_ids={side:self.base.scene[side+'_robot'].find_joints(ARM_JOINTS+FINGER_JOINTS,preserve_order=True)[0] for side in ['left','right']}
        self.box=self.base.scene['device_box'];self.fold_ids=self.box.find_joints(list(self.folds),preserve_order=True)[0]
        self.free=int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'box_free')])
        if getattr(args,'checkpoint',None) and getattr(args,'native_development_reset',False):self.restore_development_initial_state(args.checkpoint)
        if args.record:
            import imageio.v2 as imageio
            for name,filename in [('work_area','work_area'),('top_cam','top_cam')]:
                self.writers[name]=imageio.get_writer(str(Path(args.output)/(filename+'.mp4')),fps=20,
                    codec='libx264',quality=8,macro_block_size=2,
                    ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart'])
        else:self.base.cfg.sim.render_interval=10000000
        self.sync()

    def restore_development_initial_state(self,path):
        """Initialization only, before the first policy step of an explicit test.

        Production trials never enter this method. Restores measured plastic
        memory as well as poses; no object/joint state is written during a trial.
        """
        import json,torch
        record=json.loads(Path(path).read_text());snap=record['final']
        if not record.get('plastic_state'):raise ValueError('Native development reset requires complete plastic memory')
        for side in ['left','right']:
            robot=self.base.scene[side+'_robot'];ids=self.native_ids[side]
            values=torch.tensor([snap['robot_joint_positions'][side]],device=robot.device,dtype=torch.float32)
            robot.write_joint_position_to_sim_index(position=values,joint_ids=ids)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(values),joint_ids=ids)
            self.targets[side][:]=snap['targets'][side]
        quat=np.asarray(snap['box_quat_wxyz'])[[1,2,3,0]]
        pose=torch.tensor([np.r_[snap['box_position'],quat].tolist()],device=self.box.device,dtype=torch.float32)
        self.box.write_root_link_pose_to_sim_index(root_pose=pose)
        self.box.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6),device=self.box.device))
        angles=torch.tensor([[np.deg2rad(snap['angles_deg'][name]) for name in self.folds]],device=self.box.device,dtype=torch.float32)
        self.box.write_joint_position_to_sim_index(position=angles,joint_ids=self.fold_ids)
        self.box.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(angles),joint_ids=self.fold_ids)
        group=self.box.actuators['folds']
        for attr,key in [('plastic_rest','rest'),('accumulated','accumulated'),('dissipation','dissipation')]:
            setattr(group,attr,torch.tensor([[record['plastic_state'][name][key] for name in group.joint_names]],device=group._device,dtype=torch.float32))
        print('DEVELOPMENT_INITIALIZATION_ONLY',path,'No continuous pickup or task-score claim',flush=True)

    @staticmethod
    def array(value):return (value.torch if hasattr(value,'torch') else value).cpu().numpy()

    def sync(self):
        for side,(qa,da) in self.addresses.items():
            robot=self.base.scene[side+'_robot'];ids=self.native_ids[side]
            self.data.qpos[qa]=self.array(robot.data.joint_pos)[0,ids]
            self.data.qvel[da]=self.array(robot.data.joint_vel)[0,ids]
        pose=self.array(self.box.data.root_link_pose_w)[0]
        self.data.qpos[self.free:self.free+7]=np.r_[pose[:3],pose[[6,3,4,5]]]
        values=self.array(self.box.data.joint_pos)[0,self.fold_ids]
        for (q,law),v in zip(self.folds.values(),values):self.data.qpos[q]=v
        group=self.box.actuators['folds']
        if getattr(group,'plastic_rest',None) is not None:
            for index,name in enumerate(group.joint_names):
                law=self.folds[name][1]
                law.rest=float(group.plastic_rest[0,index].item())
                law.accumulated=float(group.accumulated[0,index].item())
                law.dissipation=float(group.dissipation[0,index].item())
        self.data.time=self.native_time;mujoco.mj_forward(self.model,self.data)

    def tick(self,loads=None):
        from inspect_robots import Action
        loads=loads or {};m=self.model;d=self.data
        self.last_finger_peaks[:]=0.
        jacp=np.zeros((3,m.nv));jacr=np.zeros((3,m.nv));effective={}
        for side,(qa,da) in self.addresses.items():
            torque=np.zeros(8)
            if side in loads:
                site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,side+'_tcp')
                mujoco.mj_jacSite(m,d,jacp,jacr,site);load=loads[side]
                torque=jacp[:,da].T@np.asarray(load['force'])+jacr[:,da].T@np.asarray(load.get('moment',[0,0,0]))
            target=self.targets[side]+torque/self.kp
            target[:6]=np.clip(target[:6],ik.LOWER,ik.UPPER);target[6:]=np.clip(target[6:],FINGER_OPEN,0)
            effective[side]=target
        self.last_action=np.concatenate([np.r_[effective[s][:6],effective[s][6:].mean()/FINGER_OPEN] for s in ['left','right']]).tolist()
        result=self.emb.step(Action(data=np.array(self.last_action)))
        self.last_task_info=dict(result.info)
        self.native_time+=p.CONTROL_DT;self.sync()
        history_step={'t_end':self.native_time,'tip_vectors_w_n':{}} if self.capture_force_history else None
        force_samples=np.zeros((round(p.CONTROL_DT/.001),4))
        for offset,side in [(0,'left'),(8,'right')]:
            robot=self.base.scene[side+'_robot']
            if hasattr(robot.data,'applied_torque'):
                values=self.array(robot.data.applied_torque)[0,self.native_ids[side]]
                self.peak_effort[offset:offset+8]=np.maximum(self.peak_effort[offset:offset+8],np.abs(values))
            sensor=self.base.scene[side+'_contacts'];forces=self.array(sensor.data.net_forces_w_history)[0]
            for k,name in enumerate(self.tip_names):
                if name.startswith(side+'_'):
                    local=name[len(side)+1:]
                    idx=sensor.body_names.index(local)
                    self.last_finger_peaks[k]=np.linalg.norm(forces[:,idx,:],axis=-1).max()
                    self.peak_finger_forces[k]=max(self.peak_finger_forces[k],self.last_finger_peaks[k])
                    self.finger_forces[k]=forces[0,idx,:]
                    if forces.shape[0]!=force_samples.shape[0]:raise RuntimeError('Contact history does not cover one control period')
                    force_samples[:,k]=np.linalg.norm(forces[:,idx,:],axis=-1)
                    if history_step is not None:history_step['tip_vectors_w_n'][name]=forces[:,idx,:].tolist()
        # Sensor histories are newest first. Process every 1 ms sample in time
        # order, retaining overload duration across 20 Hz control boundaries.
        self.force_monitor.observe(force_samples[::-1],.001)
        if history_step is not None and (self.all_force_history or max(self.last_finger_peaks)>40.):self.force_history.append(history_step)
        if self.writers:
            for name,writer in self.writers.items():
                camera=self.base.scene[name];camera.update(self.base.step_dt,force_recompute=True)
                rgb=self.array(camera.data.output['rgb'])[0,:,:,:3].astype('uint8')
                writer.append_data(rgb)
        snapshot=self.snapshot();snapshot['task_score']=float(self.last_task_info.get('score',0.))
        return snapshot

    def close(self):
        if self.capture_force_history:
            import json
            (self.output/'force-history.json').write_text(json.dumps({'configured_sample_period_s':.001,
                'history_samples_per_step':50,'control_period_s':p.CONTROL_DT,
                'capture_scope':'all steps' if self.all_force_history else 'steps containing a sample over 40 N',
                'scope':'Native net normal force vectors per fingertip; opposing contact objects are not identified',
                'steps':self.force_history})+'\n')
        for writer in self.writers.values():writer.close()
        self.emb.close()
