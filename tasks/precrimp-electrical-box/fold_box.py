"""Contact-driven flap forming while the right arm holds the seated box.

Checkpoint mode is a reset-only development test, not a pickup/install demo.
Only robot motor targets are commanded during the trial. Passive plastic rest
angles follow measured deformation; they are never set to a desired wrap angle.
"""
import argparse,hashlib,json,time
from pathlib import Path
import mujoco
import numpy as np
from yam_arena import ik
from yam_arena.constants import READY,FINGER_OPEN
import iberville_assets as a
import install_physics as p
import work_layout as w
import place_box as placement


def rotation_vector(v):
    angle=np.linalg.norm(v)
    if angle<1e-12:return np.eye(3)
    n=v/angle;cross=np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
    return np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)


def build_model(args):
    a.configure_flap_sheet(getattr(args,'flap_sheet_mm',.8)/1000.)
    return p.make_model(stud_front=w.CHANNEL_POSITION[0],stud_y=w.CHANNEL_POSITION[1],
        stud_yaw=np.deg2rad(w.CHANNEL_YAW_DEG),channel_profile='u',
        box_position=placement.INITIAL_POSITION,box_rotation=ik.matrix_to_quat_xyzw(placement.INITIAL_ROTATION),
        stress=250e6*args.yield_scale,timestep=args.dt,flap_damping=getattr(args,'flap_damping',.004),merge_flap_colliders=not getattr(args,'triangular_colliders',False))


def restore_checkpoint(sim,path):
    # Initialization only, before a development trial begins. A full trial uses
    # the contact pickup sequence instead and never calls this function.
    record=json.loads(Path(path).read_text());snap=record['final'];m=sim.model;d=sim.data
    free=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'box_free')]
    d.qpos[free:free+7]=snap['box_position']+snap['box_quat_wxyz'];d.qvel[:]=0.
    for side,(qa,_) in sim.addresses.items():
        d.qpos[qa]=snap['robot_joint_positions'][side]
        sim.targets[side][:]=snap['targets'][side]
    for name,(q,law) in sim.folds.items():
        d.qpos[q]=np.deg2rad(snap['angles_deg'][name])
        state=record.get('plastic_state',{}).get(name)
        if state is not None:
            law.rest=state['rest'];law.accumulated=state['accumulated'];law.dissipation=state['dissipation']
        elif name in record.get('plastic_rest_angles_deg',{}):
            # Older development traces lack accumulated strain. This defines a
            # pre-bent development reset, not an exact continuation or full demo.
            law.rest=np.deg2rad(record['plastic_rest_angles_deg'][name])
            law.accumulated=abs(law.rest);law.dissipation=law.yield_moment*law.accumulated
        m.qpos_spring[q]=law.rest
    mujoco.mj_forward(m,d)


def fingertip_vertices(model):
    d=mujoco.MjData(model);qa,_=p.addresses(model,'left');d.qpos[qa]=np.r_[READY,-.001,-.001]
    mujoco.mj_forward(model,d);site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'left_tcp')
    origin=d.site_xpos[site];R=d.site_xmat[site].reshape(3,3);points=[]
    for g in range(model.ngeom):
        body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
        if body not in ['left_tip_left','left_tip_right'] or model.geom_dataid[g]<0:continue
        mesh=model.geom_dataid[g];v=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
        local=(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-origin)@R
        points.extend(local[local[:,2]>=.010])
    return np.array(points)


def push_pose(side,degrees,root,rotation,tip_vertices,clearance=0.,closing_axis="x",station=.047,flip=True):
    curve=a.strap_centerline(side,degrees);tangent=curve[-1]-curve[-2];tangent/=np.linalg.norm(tangent)
    point=curve[-1]-(a.STRAP_LENGTH-station)*tangent
    contact=root+rotation@np.r_[point,0.]
    angle=np.deg2rad(degrees);normal=rotation@np.array([(1 if side=='front' else -1)*np.cos(angle),-np.sin(angle),0.])
    normal[2]=0.;normal/=np.linalg.norm(normal)
    direction=-w.BOX_TARGET_ROTATION[:,1];axis=w.BOX_TARGET_ROTATION[:,0 if closing_axis=='x' else 2]
    tool=np.column_stack([axis,np.cross(direction,axis),direction])
    if flip:tool=tool@np.diag([-1.,-1.,1.])
    offsets=tip_vertices@tool.T;projection=offsets@normal;top=projection.max()
    selected=np.flatnonzero(projection>=top-1e-7);j=min(selected,key=lambda j:abs(offsets[j,2]))
    target=contact-offsets[j]-normal*(a.STRAP_SHEET/2+clearance)
    return target,tool,normal,contact


class Forming(placement.Placement):
    def __init__(self,args):
        if getattr(args,'diagnostic_force_warning',False) and not (args.engine=='isaac' and args.checkpoint and getattr(args,'finish_only',False) and getattr(args,'native_development_reset',False)):
            raise ValueError('Warning-only contact monitoring is restricted to explicit native development diagnostics')
        self.args=args;self.out=Path(args.output);self.out.mkdir(parents=True,exist_ok=True)
        if args.checkpoint:(self.out/'development-initial-state.json').write_bytes(Path(args.checkpoint).read_bytes())
        source_names=sorted(path.name for path in Path(__file__).parent.glob('*.py'))
        self.source_hashes={};(self.out/'source').mkdir(exist_ok=True)
        for name in source_names:
            raw=(Path(__file__).parent/name).read_bytes()
            self.source_hashes[name]=hashlib.sha256(raw).hexdigest();(self.out/'source'/name).write_bytes(raw)
        m,xml,self.moments=build_model(args);(self.out/'model.xml').write_text(xml)
        if args.engine=='isaac':
            args.forming_box_asset=str(self.out/'device_box.usda')
            (self.out/'device_box.usda').write_text(a.box_usd(merge_flap_colliders=not args.triangular_colliders)[0])
            if args.checkpoint and not (getattr(args,'native_development_reset',False) and getattr(args,'finish_only',False)):
                raise ValueError('Native trials must start with the full pickup; only explicit finish-only development resets may use a checkpoint')
            from placement_native import NativePhysics
            self.sim=NativePhysics(m,self.moments,args)
        else:self.sim=p.Physics(m,self.moments)
        alarm_ms=getattr(args,'contact_alarm_ms',20.)
        if not np.isfinite(alarm_ms) or alarm_ms<0:raise ValueError('Contact alarm duration must be finite and nonnegative')
        self.sim.force_monitor=p.ContactForceMonitor(duration_s=alarm_ms/1000.)
        self.trace=[];self.force_warnings=[];self.stage='initial';self.started=time.perf_counter()
        self.integral=np.zeros(6);self.payload=False;self.forming=False;self.push_load=None;self.prebent_development_start=False
        self.tip=fingertip_vertices(m);self.hold_origin=None;self.hold_rotation=None
        if args.checkpoint:
            if args.engine!='isaac':restore_checkpoint(self.sim,args.checkpoint)
            self.payload=True
            self.prebent_development_start=max(abs(v) for v in self.sim.snapshot()['angles_deg'].values())>3.
        self.max_box_error=0.;self.max_box_angle=0.

    def pose_side(self,side):
        i=mujoco.mj_name2id(self.sim.model,mujoco.mjtObj.mjOBJ_SITE,side+'_tcp')
        return self.sim.data.site_xpos[i].copy(),self.sim.data.site_xmat[i].reshape(3,3).copy()

    def tick(self):
        loads={};nominal=self.sim.targets['right'][:6].copy()
        if self.forming:
            root,R=self.sim.box_pose();err=self.hold_origin-root;er=ik._rot_error(self.hold_rotation,R)
            self.max_box_error=max(self.max_box_error,float(np.linalg.norm(err)))
            self.max_box_angle=max(self.max_box_angle,float(np.linalg.norm(er)))
            goal=self.hold_origin+self.hold_rotation@self.hold_offset+np.clip(err*1.5,-.003,.003)
            correction=er*1.5
            if np.linalg.norm(correction)>np.deg2rad(2):correction*=np.deg2rad(2)/np.linalg.norm(correction)
            rr=rotation_vector(correction)@self.hold_rotation@self.hold_relative
            self.sim.command('right',goal,rr)
            nominal=self.sim.targets['right'][:6].copy()
        if self.payload:
            hand,_=self.pose_side('right');m=self.sim.model;d=self.sim.data
            force=np.array([0.,0.,m.body_subtreemass[self.sim.housing]*9.81])
            moment=np.cross(d.subtree_com[self.sim.housing]-hand,force)
            if self.forming:
                # Brace the sidewall against the web. Counter the pusher's
                # withdrawal load through the ordinary right-arm motor target.
                press=self.args.brace_n
                if self.args.brace_n>0 and self.push_load is not None:press+=max(0.,-float(self.push_load['force'][1]))
                force+=np.array([0.,press,0.])
                if self.args.brace_moment_at_com:moment=np.cross(d.subtree_com[self.sim.housing]-hand,force)
            loads['right']={'force':force,'moment':moment}
            qa,_=self.sim.addresses['right'];err=nominal-d.qpos[qa[:6]]
            self.integral=np.clip(self.integral+self.args.integral*p.CONTROL_DT*err,-.035,.035)
            self.sim.targets['right'][:6]+=self.integral
        if self.push_load is not None:loads['left']=self.push_load
        snap=self.sim.tick(loads);self.sim.targets['right'][:6]=nominal
        snap['stage']=self.stage;self.trace.append(snap)
        snap['finger_peak_forces_n']=dict(zip(self.sim.tip_names,self.sim.last_finger_peaks.tolist()))
        if max(self.sim.last_finger_peaks)>40.:
            event={'t':snap['t'],'stage':self.stage,'peaks_n':snap['finger_peak_forces_n'],
                'longest_overload_ms':(self.sim.force_monitor.longest_s*1000.).tolist()}
            self.force_warnings.append(event);print('CONTACT_FORCE_WARNING',event,flush=True)
        if np.any(self.sim.force_monitor.triggered) and not getattr(self.args,'diagnostic_force_warning',False):
            raise RuntimeError(f'Sustained fingertip contact exceeds 40 N for {self.sim.force_monitor.duration_s*1000.:g} ms')
        if self.sim.peak_penetration>.0015:raise RuntimeError(f'Penetration {self.sim.peak_penetration*1000:.2f} mm: {self.sim.worst_pair}')
        if self.sim.robot_collision:raise RuntimeError(f'Inter-arm collision: {self.sim.robot_collision_pair}')
        if not np.all(np.isfinite(self.sim.data.qpos)):raise RuntimeError('Non-finite state')
        if self.forming and (self.max_box_error>.005 or self.max_box_angle>np.deg2rad(5)):
            raise RuntimeError(f'Box moved during forming: {self.max_box_error*1000:.2f} mm / {np.rad2deg(self.max_box_angle):.2f} degrees')
        if not self.forming and not self.prebent_development_start and max(abs(v) for v in snap['angles_deg'].values())>3:raise RuntimeError('Unexpected bend during placement')

    def move_left(self,target,R,seconds,grip=-.001,joint=False):
        start,sr=self.pose_side('left');begin=self.sim.targets['left'][:6].copy();g=self.sim.targets['left'][6]
        if joint:
            goal,pe,re=self.sim.planner.solve('left',target,R,begin)
            if pe>.002 or re>.025:raise RuntimeError(f'Left transit unreachable: {pe*1000:.2f} mm / {re:.3f} rad')
        for t in np.linspace(0,1,round(seconds/p.CONTROL_DT)+1)[1:]:
            u=ik.smoothstep(t)
            if joint:self.sim.targets['left'][:6]=begin+u*(goal-begin)
            else:self.sim.command('left',start+u*(np.asarray(target)-start),ik.slerp_matrix(sr,R,u))
            self.sim.targets['left'][6:]=g+u*(grip-g);self.tick()
        self.report()

    def material_contact(self,side,clearance=0.):
        sign=-1 if side=='front' else 1
        bend=sign*sum(v for n,v in self.sim.snapshot()['angles_deg'].items() if n.startswith(side+'_'))
        station=self.args.contact_station
        if self.args.walk_contact:
            station+=self.args.walk_distance_mm/1000.*float(np.interp(bend,[0,60,90,120,160,200],[0,0,1,1,0,0]))
        body=mujoco.mj_name2id(self.sim.model,mujoco.mjtObj.mjOBJ_BODY,side+'_strap_14')
        data=self.sim.data;R=data.xmat[body].reshape(3,3)
        contact=data.xpos[body]+R@np.array([0.,station-a.FOLD_STATIONS[-2],0.])
        normal=R[:,0]*(1 if side=='front' else -1);normal[2]=0.;normal/=np.linalg.norm(normal)
        basis=w.BOX_TARGET_ROTATION;direction=-basis[:,1];axis=basis[:,0 if self.args.closing_axis=='x' else 2]
        tool=np.column_stack([axis,np.cross(direction,axis),direction])
        if not self.args.no_flip:tool=tool@np.diag([-1.,-1.,1.])
        offsets=self.tip@tool.T;projection=offsets@normal
        choices=np.flatnonzero(projection>=projection.max()-1e-7)
        j=min(choices,key=lambda k:abs(offsets[k,2]))
        offset=offsets[j].copy()
        if self.args.support_blend_mm>0:
            weights=np.exp((projection-projection.max())/(self.args.support_blend_mm/1000.))
            offset=(weights[:,None]*offsets).sum(axis=0)/weights.sum()
            # Blend tangential support across the two jaws, including height.
            # Preserve the exact outer support plane; this adds no penetration.
            offset+=normal*(projection.max()-offset@normal)
        goal=contact-offset-normal*(a.STRAP_SHEET/2+clearance)
        return goal,tool,normal,contact

    def form(self,side):
        root=self.hold_origin;R=self.hold_rotation
        position,tool,normal,contact=(self.material_contact(side,.003) if self.args.follow_live else push_pose(side,0.,root,R,self.tip,.003,self.args.closing_axis,self.args.contact_station,not self.args.no_flip))
        self.stage=side+'_approach';self.move_left(position-normal*.012+[0,.055,.030],tool,3.,joint=True)
        self.move_left(position,tool,2.)
        self.stage=side+'_contact';position,tool,normal,contact=(self.material_contact(side,-self.args.overtravel) if self.args.follow_live else push_pose(side,0.,root,R,self.tip,-self.args.overtravel,self.args.closing_axis,self.args.contact_station,not self.args.no_flip))
        self.move_left(position,tool,1.)
        if self.args.follow_live:
            self.live_fold(side,tool)
            return
        self.stage=side+'_fold'
        for step,degree in enumerate(np.linspace(0,self.args.degrees,round(self.args.fold_seconds/p.CONTROL_DT)+1)[1:],1):
            position,tool,normal,contact=push_pose(side,float(degree),root,R,self.tip,-self.args.overtravel,self.args.closing_axis,self.args.contact_station,not self.args.no_flip)
            self.sim.command('left',position,tool)
            hand,_=self.pose_side('left');force=normal*self.args.push_n*min(1.,step*p.CONTROL_DT/2.)
            self.push_load={'force':force,'moment':np.cross(contact-hand,force)}
            self.tick()
            if len(self.trace)%20==0:
                measured=sum(v for n,v in self.sim.snapshot()['angles_deg'].items() if n.startswith(side+'_'))
                print('FOLD_PROGRESS',side,round(float(degree),1),round(measured,1),np.round(self.sim.peak_finger_forces,2).tolist(),flush=True)
        self.report();self.push_load=None
        self.stage=side+'_retract';self.move_left(position-normal*.012+[0,.045,.015],tool,2.)
        self.hold(.5);self.report()
        angles=self.sim.snapshot()['angles_deg'];bend=sum(v for n,v in angles.items() if n.startswith(side+'_'))
        print('MEASURED_BEND',side,bend,flush=True)
        if abs(bend)<self.args.degrees*.7:raise RuntimeError(f'{side} wrap only {bend:.1f} degrees')

    def live_fold(self,side,tool):
        self.stage=side+f'_fold_{self.active_goal:g}'
        sign=-1 if side=='front' else 1
        point_target,_=self.pose_side('left')
        best=0.;last_progress=0.;finished=False
        limit=min(self.args.initial_push_n,self.args.push_n) if self.active_goal<=90 else self.args.push_n
        for step in range(round(self.args.fold_seconds/p.CONTROL_DT)):
            goal,tool,normal,contact=self.material_contact(side,-self.args.overtravel)
            change=goal-point_target;distance=np.linalg.norm(change)
            max_change=self.args.contact_speed_mm_s/1000.*p.CONTROL_DT
            if distance>max_change:change*=max_change/distance
            point_target+=change
            hand,hand_R=self.pose_side('left')
            tangent_error=point_target-hand
            tangent_error-=normal*np.dot(tangent_error,normal)
            correction=np.clip(tangent_error*1.5,-.003,.003)
            er=ik._rot_error(tool,hand_R)*1.5
            if np.linalg.norm(er)>np.deg2rad(2):er*=np.deg2rad(2)/np.linalg.norm(er)
            self.sim.command('left',point_target+correction,rotation_vector(er)@tool)
            hand,_=self.pose_side('left')
            # Back off the applied motor-target wrench before the hard guard.
            # This uses measured contact, never edits the sheet or its state.
            recent=max(getattr(self.sim,'last_finger_peaks',np.linalg.norm(self.sim.finger_forces,axis=1))[:2])
            if recent>self.args.soft_force_n:
                limit=max(0.,limit-(recent-self.args.soft_force_n)*.3)
                point_target-=normal*min(.0002,(recent-self.args.soft_force_n)*.00002)
            force=normal*limit*min(1.,(step+1)*p.CONTROL_DT/2.)
            self.push_load={'force':force,'moment':np.cross(contact-hand,force)}
            self.tick()
            bend=sign*sum(v for n,v in self.sim.snapshot()['angles_deg'].items() if n.startswith(side+'_'))
            if bend>best+.5:best=bend;last_progress=step*p.CONTROL_DT
            if step*p.CONTROL_DT-last_progress>1. and recent<min(32.,self.args.soft_force_n-2.):
                limit=min(self.args.max_push_n,limit+p.CONTROL_DT*self.args.force_increase_n_per_second)
            if step%20==19:print('LIVE_PROGRESS',side,round((step+1)*p.CONTROL_DT,1),round(bend,1),np.round(self.sim.peak_finger_forces,2).tolist(),flush=True)
            if bend>=self.active_goal:finished=True;break
            if step*p.CONTROL_DT-last_progress>5.:break
        self.push_load=None;self.report()
        current,_=self.pose_side('left')
        self.stage=side+'_retract';self.move_left(current-normal*.012+[0,.045,.015],tool,2.)
        self.hold(.5);self.report()
        bend=sign*sum(v for n,v in self.sim.snapshot()['angles_deg'].items() if n.startswith(side+'_'))
        print('MEASURED_BEND',side,bend,flush=True)
        minimum=getattr(self.args,'retained_min_degrees',150.) if self.active_goal>=150 else 0.
        if bend<max(self.active_goal*.7,minimum):raise RuntimeError(f'{side} retained wrap only {bend:.1f} degrees; peak progress {best:.1f}')

    def run(self):
        if not self.args.checkpoint:
            self.stage='placement';placement.Placement.run(self)
        self.prepare_holder()
        for side in self.args.flaps.split(','):
            goals=[80.,self.args.degrees] if self.args.staged_bends and self.args.degrees>100 else [self.args.degrees]
            for goal in goals:
                self.active_goal=goal;self.form(side)
        self.stage=('precrimp_complete_supported' if getattr(self.args,'precrimp_mode',False) else 'both_flaps_folded_held')
        self.hold(2.);self.report()

    def prepare_holder(self):
        self.stage='holder_settle';self.hold(.5)
        root,R=self.sim.box_pose();hand,hr=self.pose_side('right')
        if abs(self.args.height-root[2])>.002:
            self.stage='adjust_mounting_height';goal=root.copy();goal[2]=self.args.height
            self.move_box(goal,R,3.,feedback=True);root,R=self.sim.box_pose();hand,hr=self.pose_side('right')
        self.hold_origin=root.copy();self.hold_rotation=R.copy()
        self.hold_offset=R.T@(hand-root);self.hold_relative=R.T@hr;self.forming=True
        self.stage='brace_box';self.move(hand,hr,1.,grip=self.args.holding_grip)
        self.hold(.5)
        root,R=self.sim.box_pose();hand,hr=self.pose_side('right')
        self.hold_origin=root.copy();self.hold_rotation=R.copy()
        self.hold_offset=R.T@(hand-root);self.hold_relative=R.T@hr

    def save(self,error):
        super().save(error)
        path=self.out/'result.json';r=json.loads(path.read_text())
        if self.args.engine=='mujoco':r['contact_monitoring']=f'Every {self.args.dt*1e6:g} microsecond step; summed fingertip force, penetration, and inter-arm contact'
        tips={}
        root,R=self.sim.box_pose()
        for side in ['front','rear']:
            b=mujoco.mj_name2id(self.sim.model,mujoco.mjtObj.mjOBJ_BODY,side+'_strap_14')
            point=self.sim.data.xpos[b]+self.sim.data.xmat[b].reshape(3,3)@np.array([0.,a.STRAP_LENGTH-a.FOLD_STATIONS[-2],0.])
            tips[side]={'endpoint_in_box_m':(R.T@(point-root)).tolist(),
                'cumulative_angle_deg':sum(v for n,v in self.sim.snapshot()['angles_deg'].items() if n.startswith(side+'_'))}
        r.update(flap_geometry=tips,prebent_development_start=self.prebent_development_start,development_checkpoint=bool(self.args.checkpoint),checkpoint_path=self.args.checkpoint,
            forming_demonstrated=error is None,placement_demonstrated=not self.args.checkpoint and error is None,
            full_installation_demonstrated=False,held_during_forming=True,
            max_box_displacement_mm=self.max_box_error*1000,max_box_rotation_degrees=float(np.rad2deg(self.max_box_angle)),
            material={'yield_stress_mpa_assumed':250*self.args.yield_scale,'yield_scale':self.args.yield_scale,
                'elastic_modulus_gpa_assumed':200.,'flap_sheet_mm':a.STRAP_SHEET*1000,
                'hinge_damping_nm_s_per_rad':self.args.flap_damping,
                'calibrated_to_real_bracket':False},
            plastic_state={n:{'rest':law.rest,'accumulated':law.accumulated,'dissipation':law.dissipation} for n,(_,law) in self.sim.folds.items()},
            plastic_rest_angles_deg={n:float(np.rad2deg(law.rest)) for n,(_,law) in self.sim.folds.items()})
        r['source_sha256']=self.source_hashes
        r['task_score']=float(getattr(self.sim,'last_task_info',{}).get('score',0.))
        r['contact_alarm']={'threshold_n':40.,'sustained_duration_ms':self.sim.force_monitor.duration_s*1000.,
            'longest_overload_ms':dict(zip(self.sim.tip_names,(self.sim.force_monitor.longest_s*1000.).tolist())),
            'sustained_alarm_triggered':bool(np.any(self.sim.force_monitor.triggered)),
            'warning_only_diagnostic':getattr(self.args,'diagnostic_force_warning',False),
            'warnings':self.force_warnings}
        path.write_text(json.dumps(r,indent=2)+'\n')


def argument_parser():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--engine',choices=['mujoco','isaac'],default='mujoco')
    parser.add_argument('--record',action='store_true');parser.add_argument('--renderer-gpu',type=int,default=0)
    parser.add_argument('--physics-device',choices=['cpu','cuda:0'],default=None,
        help='Optional native PhysX device override; renderer GPU is independent')
    parser.add_argument('--checkpoint');parser.add_argument('--height',type=float,default=.94)
    parser.add_argument('--native-development-reset',action='store_true',help='Explicit initialization-only native finish test; never a continuous demo')
    parser.add_argument('--contact-alarm-ms',type=float,default=20.,help='Continuous time over 40 N required for a contact stop; 0 restores the instantaneous stop')
    parser.add_argument('--flap-sheet-mm',type=float,default=.8)
    parser.add_argument('--flap-damping',type=float,default=.02)
    parser.add_argument('--triangular-colliders',action='store_true')
    parser.add_argument('--yield-scale',type=float,default=1.);parser.add_argument('--dt',type=float,default=.0001)
    parser.add_argument('--degrees',type=float,default=165.);parser.add_argument('--fold-seconds',type=float,default=10.)
    parser.add_argument('--retained-min-degrees',type=float,default=150.)
    parser.add_argument('--initial-push-n',type=float,default=22.)
    parser.add_argument('--max-push-n',type=float,default=28.);parser.add_argument('--force-increase-n-per-second',type=float,default=1.)
    parser.add_argument('--push-n',type=float,default=12.);parser.add_argument('--overtravel',type=float,default=.00015)
    parser.add_argument('--closing-axis',choices=['x','z'],default='x');parser.add_argument('--contact-station',type=float,default=.047)
    parser.add_argument('--staged-bends',action='store_true');parser.add_argument('--brace-moment-at-com',action='store_true')
    parser.add_argument('--brace-n',type=float,default=5.);parser.add_argument('--holding-grip',type=float,default=-.0325)
    parser.add_argument('--walk-distance-mm',type=float,default=1.5)
    parser.add_argument('--support-blend-mm',type=float,default=.75)
    parser.add_argument('--contact-speed-mm-s',type=float,default=10.)
    parser.add_argument('--soft-force-n',type=float,default=28.)
    parser.add_argument('--walk-contact',action='store_true')
    parser.add_argument('--no-flip',action='store_true');parser.add_argument('--follow-live',action='store_true')
    parser.add_argument('--flaps',default='front,rear');parser.add_argument('--integral',type=float,default=1.)
    return parser


def main():
    args=argument_parser().parse_args();args.stage='place';args.closed=-.034;args.forming_material=True
    trial=Forming(args);error=None
    try:trial.run()
    except Exception as exc:error=str(exc)
    trial.save(error)
    if hasattr(trial.sim,'close'):trial.sim.close()
    if error:raise SystemExit(1)

if __name__=='__main__':main()
