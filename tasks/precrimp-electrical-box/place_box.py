"""Pick up and seat the box using bounded YAM motor targets and contact.

This is a placement subtask: the right hand continues holding the box and the
flaps stay unbent. There are no object pose writes, attachments or tab commands
following Physics initialization. Offline dynamics use the task's MuJoCo model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from yam_arena import ik
from yam_arena.constants import FINGER_OPEN
import iberville_assets as a
import install_physics as p
import work_layout as w

# Box closed back on the table. The same grasp is maintained through seating.
INITIAL_ROTATION=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
_vertices=np.concatenate([s.mesh.vertices for s in a.body_solids() if s.physical])
INITIAL_POSITION=np.array([.25,-.18,a.TOP+.0002-(_vertices@INITIAL_ROTATION.T)[:,2].min()])
GRASP_OFFSET=np.array([.012,-.0274,.001])
_angle=np.deg2rad(75.)
GRASP_ROTATION=np.array([[0.,-np.sin(_angle),np.cos(_angle)],
                         [0.,np.cos(_angle),np.sin(_angle)],[-1.,0.,0.]])
OPENING=-.0399


def make_model():
    return p.make_model(stud_front=w.CHANNEL_POSITION[0],stud_y=w.CHANNEL_POSITION[1],
        stud_yaw=np.deg2rad(w.CHANNEL_YAW_DEG),channel_profile='u',
        box_position=INITIAL_POSITION,box_rotation=ik.matrix_to_quat_xyzw(INITIAL_ROTATION))


class Placement:
    def __init__(self,args):
        self.args=args;self.out=Path(args.output);self.out.mkdir(parents=True,exist_ok=True)
        model,xml,self.moments=make_model();(self.out/'model.xml').write_text(xml)
        if args.engine=='isaac':
            from placement_native import NativePhysics
            self.sim=NativePhysics(model,self.moments,args)
        else:self.sim=p.Physics(model,self.moments)
        self.trace=[];self.stage='settle'
        self.started=time.perf_counter();self.payload=False
        self.integral=np.zeros(6);self.nominal=None

    def pose(self):
        d=self.sim.data;m=self.sim.model;i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'right_tcp')
        return d.site_xpos[i].copy(),d.site_xmat[i].reshape(3,3).copy()

    def tick(self):
        loads={};nominal=self.sim.targets['right'][:6].copy()
        if self.payload:
            hand,_=self.pose();m=self.sim.model;d=self.sim.data
            force=np.array([0.,0.,m.body_subtreemass[self.sim.housing]*9.81])
            loads={'right':{'force':force,'moment':np.cross(d.subtree_com[self.sim.housing]-hand,force)}}
            if self.args.integral:
                qa,_=self.sim.addresses['right']
                err=nominal-d.qpos[qa[:6]]
                self.integral=np.clip(self.integral+self.args.integral*p.CONTROL_DT*err,-.035,.035)
                self.sim.targets['right'][:6]+=self.integral
        snap=self.sim.tick(loads);self.sim.targets['right'][:6]=nominal
        snap['stage']=self.stage;self.trace.append(snap)
        if not np.all(np.isfinite(snap['box_position']+snap['box_quat_wxyz']+list(snap['angles_deg'].values()))):
            raise RuntimeError('Non-finite simulated state')
        if self.sim.peak_penetration>.0015:
            raise RuntimeError(f'Contact penetration {1000*self.sim.peak_penetration:.2f} mm: {self.sim.worst_pair}')
        if self.sim.robot_collision:raise RuntimeError('Inter-arm contact')
        if max(self.sim.peak_finger_forces)>40:
            raise RuntimeError(f'Fingertip force {max(self.sim.peak_finger_forces):.2f} N exceeds 40 N')
        if max(abs(v) for v in snap['angles_deg'].values())>3:
            raise RuntimeError('Unintended flap bend exceeds 3 degrees')

    def hold(self,seconds):
        for _ in range(round(seconds/p.CONTROL_DT)):self.tick()

    def move(self,target,rotation,seconds,grip=None,joint=False):
        pos,r=self.pose();g=self.sim.targets['right'][6]
        if joint:
            start=self.sim.targets['right'][:6].copy()
            goal,pe,re=self.sim.planner.solve('right',target,rotation,start)
            if pe>.002 or re>.025:raise RuntimeError(f'Transit IK {pe*1000:.2f} mm {re:.3f} rad')
        for t in np.linspace(0,1,round(seconds/p.CONTROL_DT)+1)[1:]:
            u=ik.smoothstep(t)
            if joint:self.sim.targets['right'][:6]=start+u*(goal-start)
            else:self.sim.command('right',pos+u*(np.asarray(target)-pos),ik.slerp_matrix(r,rotation,u))
            if grip is not None:self.sim.targets['right'][6:]=g+u*(grip-g)
            self.tick()
        self.report()

    def move_box(self,target,rotation,seconds,feedback=False):
        root,r=self.sim.box_pose();hand,hr=self.pose()
        offset=r.T@(hand-root);relative=r.T@hr
        for t in np.linspace(0,1,round(seconds/p.CONTROL_DT)+1)[1:]:
            u=ik.smoothstep(t);rr=ik.slerp_matrix(r,rotation,u)
            desired=root+u*(np.asarray(target)-root)
            goal=desired+rr@offset;hand_rotation=rr@relative
            if feedback:
                # Correct transverse tracking and orientation continuously;
                # the insertion direction retains its slow, prescribed path.
                actual,ar=self.sim.box_pose()
                correction=np.clip((desired-actual)*[2.,0.,.5],-.003,.003)
                error=ik._rot_error(rr,ar)*2.
                magnitude=np.linalg.norm(error)
                if magnitude>np.deg2rad(1.5):error*=np.deg2rad(1.5)/magnitude
                magnitude=np.linalg.norm(error)
                if magnitude>1e-12:
                    axis=error/magnitude
                    cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
                    delta=np.eye(3)+np.sin(magnitude)*cross+(1-np.cos(magnitude))*(cross@cross)
                    hand_rotation=delta@hand_rotation
                goal+=correction
            self.sim.command('right',goal,hand_rotation)
            self.tick()
            if feedback:
                actual,ar=self.sim.box_pose()
                angle=np.arccos(np.clip((np.trace(rotation.T@ar)-1)/2,-1,1))
                if actual[1]+a.WIDTH/2+a.STRAP_LENGTH>w.CHANNEL_POSITION[1]-.003 and (abs(actual[0]-target[0])>.001 or angle>np.deg2rad(.75)):
                    raise RuntimeError('Insertion stopped: transverse alignment exceeded 1 mm / 0.75 degree')
        self.hold(.4);self.report()

    def report(self):
        pos,r=self.sim.box_pose()
        print(self.stage,round(self.sim.data.time,2),'box',np.round(pos,5).tolist(),
              'peak fingers',np.round(self.sim.peak_finger_forces,2).tolist(),flush=True)

    def run(self):
        self.hold(.5)
        root,r=self.sim.box_pose();self.pickup_height=root[2]
        hand=root+r@GRASP_OFFSET;rotation=r@GRASP_ROTATION
        self.stage='approach';self.move(hand-rotation[:,2]*.030+[0,0,.060],rotation,3.,OPENING,joint=True)
        self.stage='lower';self.move(hand-rotation[:,2]*.030,rotation,2.)
        self.stage='enter_grasp';self.move(hand,rotation,2.)
        self.stage='grip';self.move(hand,rotation,1.5,self.args.closed);self.hold(.5)
        self.payload=True
        self.stage='lift';current,rotation=self.pose();self.move(current+[0,0,.08],rotation,3.);self.hold(.5)
        if self.sim.box_pose()[0][2]<self.pickup_height+.05:raise RuntimeError('Box was not lifted by the grasp')
        if self.args.stage=='pickup':return
        self.stage='align';pre=w.BOX_TARGET_POSITION+np.array([0,-.080,0])
        self.move_box(pre,w.BOX_TARGET_ROTATION,6.)
        # Correct small tracking errors before the narrow-clearance insertion.
        self.stage='fine_align';self.move_box(pre,w.BOX_TARGET_ROTATION,2.)
        if self.args.stage=='align':return
        self.stage='insert';self.move_box(w.BOX_TARGET_POSITION,w.BOX_TARGET_ROTATION,10.,feedback=True)
        self.stage='seat';self.move_box(w.BOX_TARGET_POSITION,w.BOX_TARGET_ROTATION,2.,feedback=True)
        self.stage='held_in_place';self.hold(2.);self.report()
        pos,r=self.sim.box_pose();error=pos-w.BOX_TARGET_POSITION
        angle=np.arccos(np.clip((np.trace(w.BOX_TARGET_ROTATION.T@r)-1)/2,-1,1))
        if abs(error[0])>.0005 or abs(error[1])>.001 or abs(error[2])>.003 or angle>np.deg2rad(.6):
            raise RuntimeError(f'Placement tolerance: {error*1000} mm, {np.rad2deg(angle):.3f} degrees')

    def save(self,error):
        pos,r=self.sim.box_pose()
        result={'engine':('Native Isaac/PhysX contact dynamics' if self.args.engine=='isaac' else 'MuJoCo contact dynamics; native Isaac execution not yet validated'),
            'requested_stage':self.args.stage,'requested_stage_completed':error is None,
            'placement_demonstrated':error is None and self.args.stage=='place',
            'full_installation_demonstrated':False,'box_still_held':self.payload,
            'error':error,'last_stage':self.stage,'simulated_seconds':self.sim.data.time,
            'wall_seconds':time.perf_counter()-self.started,'control_steps':len(self.trace),
            'initial_position':INITIAL_POSITION.tolist(),'initial_rotation':INITIAL_ROTATION.tolist(),
            'grasp_offset':GRASP_OFFSET.tolist(),'grasp_rotation':GRASP_ROTATION.tolist(),
            'final_position_error_mm':((pos-w.BOX_TARGET_POSITION)*1000).tolist(),
            'final_rotation_error_deg':float(np.rad2deg(np.arccos(np.clip((np.trace(w.BOX_TARGET_ROTATION.T@r)-1)/2,-1,1)))),
            'peak_motor_effort':self.sim.peak_effort.tolist(),
            'peak_penetration_mm':self.sim.peak_penetration*1000 if self.args.engine=='mujoco' else None,
            'worst_contact_pair':self.sim.worst_pair if self.args.engine=='mujoco' else None,
            'peak_finger_contact_forces_n':dict(zip(self.sim.tip_names,self.sim.peak_finger_forces.tolist())),
            'inter_arm_contact':self.sim.robot_collision if self.args.engine=='mujoco' else None,
            'contact_monitoring':('Every 50 microsecond step; penetration, inter-arm contact, and summed fingertip forces'
                if self.args.engine=='mujoco' else 'Native fingertip net-contact force history at 1 kHz; penetration and inter-arm pairs not instrumented'),
            'final':self.sim.snapshot(),'arguments':vars(self.args),
            'source_sha256':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
                for n in ['place_box.py','placement_native.py','iberville_runtime.py','install_physics.py','iberville_assets.py','work_layout.py']}}
        for name,value in [('result',result),('trace',self.trace),('actions',[s['action_14d'] for s in self.trace])]:
            (self.out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['final','source_sha256']},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['pickup','align','place'],default='place')
    parser.add_argument('--engine',choices=['mujoco','isaac'],default='mujoco')
    parser.add_argument('--renderer-gpu',type=int,default=0)
    parser.add_argument('--record',action='store_true')
    parser.add_argument('--closed',type=float,default=-.034)
    parser.add_argument('--integral',type=float,default=0.)
    args=parser.parse_args();trial=Placement(args);error=None
    try:trial.run()
    except Exception as exc:error=str(exc)
    trial.save(error)
    if hasattr(trial.sim,'close'):trial.sim.close()
    if error:raise SystemExit(1)

if __name__=='__main__':main()
