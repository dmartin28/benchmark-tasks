"""Contact-driven installation experiments with the pinned YAM robot model.

No object attachments, pose commands or commanded tab angles during a trial.
The box's passive folds use uncalibrated elastic-plastic return mapping.
"""
from dataclasses import dataclass
import copy
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from shapely.geometry import LineString
from yam_arena import YAM_URDF,ik,station
from yam_arena.constants import READY,TCP_OFFSET,FINGER_OPEN,SHOULDER,WRIST,GRIPPER
import iberville_assets as a

DT=.00005
CONTROL_DT=.05


def numbers(values):return ' '.join(f'{float(v):.10g}' for v in values)


@dataclass
class PlasticFold:
    yield_moment:float
    stiffness:float=80.
    hardening:float=.05
    rest:float=0.
    accumulated:float=0.
    dissipation:float=0.

    def update(self,angle):
        trial=self.stiffness*(angle-self.rest)
        radius=self.yield_moment+self.hardening*self.accumulated
        excess=abs(trial)-radius
        if excess>0:
            increment=excess/(self.stiffness+self.hardening)
            self.rest+=math.copysign(increment,trial)
            self.accumulated+=increment
            self.dissipation+=self.yield_moment*increment
        return self.stiffness*(angle-self.rest)


def fold_moments(stress=250e6, stations=None):
    poly=a.strap_polygon();result={}
    for side in a.STRAP_ROOTS:
        for k,s in enumerate((stations or {}).get(side,a.FOLD_STATIONS)[:-1]):
            sample=max(.000001,s)
            width=poly.intersection(LineString([(sample,-a.STRAP_HEIGHT),(sample,a.STRAP_HEIGHT)])).length
            result[f'{side}_fold_{k}']=stress*width*a.STRAP_SHEET**2/4
    return result


def make_model(*,stud_front=a.STUD_FRONT,stud_y=a.STUD_Y,stud_depth=a.STUD_DEPTH,
               box_position=a.BOX_POSITION,box_rotation=a.BOX_ROT_XYZW,
               stress=250e6,grip_friction=.7,steel_friction=.3,
               timestep=DT,include_robot=True,box_fixed=False,stud_open_towards_box=False,fold_stations=None,channel_profile='c',stud_yaw=0.,flap_damping=.004,merge_flap_colliders=False):
    if channel_profile not in ['c','u']:raise ValueError('Channel profile must be c or u')
    root=ET.Element('mujoco',model='iberville_install_'+channel_profile)
    ET.SubElement(root,'compiler',angle='radian',balanceinertia='true')
    ET.SubElement(root,'size',memory='256M')
    ET.SubElement(root,'option',timestep=str(timestep),integrator='implicitfast',
                  solver='Newton',iterations='80',noslip_iterations='5',tolerance='1e-10',gravity='0 0 -9.81')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='1080',offheight='720')
    ET.SubElement(visual,'headlight',ambient='.45 .45 .45',diffuse='.7 .7 .7')
    default=ET.SubElement(root,'default')
    ET.SubElement(default,'geom',friction=f'{steel_friction} .001 .0001',
                  solref='.003 1',solimp='.95 .99 .0002',margin='.00005',condim='3')
    asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='.2 -.4 3',dir='0 0 -1',diffuse='.8 .8 .8')
    ET.SubElement(world,'geom',name='table',type='box',size='.4 .6 .02',
                  pos=numbers([.35,0,a.TOP-.02]),rgba='.55 .43 .3 1',contype='16',conaffinity='7')
    actuators=ET.SubElement(root,'actuator')
    if include_robot:
        urdf=ET.parse(YAM_URDF).getroot()
        for mesh in urdf.findall('.//mesh'):mesh.set('filename',str(YAM_URDF.parent/mesh.get('filename')))
        ET.SubElement(ET.SubElement(urdf,'mujoco'),'compiler',fusestatic='false',discardvisual='true',balanceinertia='true')
        original=ET.fromstring(mujoco.MjSpec.from_string(ET.tostring(urdf,encoding='unicode')).to_xml())
        for item in original.find('asset'):asset.append(copy.deepcopy(item))
        for side,mask in [('left',1),('right',2)]:
            robot=copy.deepcopy(original.find('worldbody/body'));robot.set('pos',numbers(station.BASES[side]))
            index=0
            for e in robot.iter():
                if 'name' in e.attrib:e.set('name',side+'_'+e.get('name'))
                if e.tag=='geom':
                    e.set('contype',str(mask));e.set('conaffinity',str((3^mask)|4|8|16))
                    e.set('friction',f'{grip_friction} .003 .0001');e.set('condim','4')
                    if 'name' not in e.attrib:e.set('name',f'{side}_geom_{index}')
                    index+=1
                if e.tag=='joint':
                    name=e.get('name');i=int(name[-1]);finger=i>6
                    drive=GRIPPER if finger else (SHOULDER if i<=3 else WRIST)
                    cap=40 if finger else drive['effort']
                    e.set('actuatorfrcrange',f'{-cap} {cap}');e.set('damping','0')
                    e.set('frictionloss','0' if finger else ('.3' if i<=3 else '.06'))
                    kp,kd=drive['stiffness'],drive['damping']
                    ET.SubElement(actuators,'general',name=name+'_motor',joint=name,gainprm=str(kp),
                        biastype='affine',biasprm=f'0 {-kp} {-kd}',forcelimited='true',forcerange=f'{-cap} {cap}')
            hand=next(b for b in robot.iter('body') if b.get('name')==side+'_gripper')
            ET.SubElement(hand,'site',name=side+'_tcp',pos=numbers(TCP_OFFSET),size='.002')
            world.append(robot)

    def add_solids(parent,prefix,solids,mask,affinity):
        for i,s in enumerate(solids):
            for suffix,mesh,physical in [('visual',s.mesh,False)]+[(f'cell{j}',c,True) for j,c in enumerate(s.cells)]:
                name=f'{prefix}_{i}_{suffix}'
                ET.SubElement(asset,'mesh',name=name,vertex=numbers(mesh.vertices.ravel()),face=' '.join(map(str,mesh.faces.ravel())))
                ET.SubElement(parent,'geom',name=name,type='mesh',mesh=name,density='0',
                    contype=str(mask if physical else 0),conaffinity=str(affinity if physical else 0),
                    group='3' if physical else '0',rgba='.62 .67 .71 1')
    def inertial(parent,solids):
        mass,center,inertia,quat=a.aggregate_mass(solids)
        ET.SubElement(parent,'inertial',mass=str(mass),pos=numbers(center),diaginertia=numbers(inertia),quat=numbers(quat))

    box=ET.SubElement(world,'body',name='housing',pos=numbers(box_position),quat=numbers(np.asarray(box_rotation)[[3,0,1,2]]))
    if not box_fixed:ET.SubElement(box,'freejoint',name='box_free')
    inertial(box,a.body_solids());add_solids(box,'housing',a.body_solids(),4,31)
    moments=fold_moments(stress,fold_stations);polygon=a.strap_polygon()
    for side,x in a.STRAP_ROOTS.items():
        parent=box;stations=(fold_stations or {}).get(side,a.FOLD_STATIONS)
        for k,(start,end) in enumerate(zip(stations,stations[1:])):
            name=f'{side}_fold_{k}'
            pivot=(x,a.WIDTH/2,0) if k==0 else (0,start-stations[k-1],0)
            link=ET.SubElement(parent,'body',name=f'{side}_strap_{k}',pos=numbers(pivot))
            ET.SubElement(link,'joint',name=name,type='hinge',axis='0 0 1',range='-3.054 3.054',
                limited='true',stiffness=str(a.fold_stiffness(k,stations)),damping=str(flap_damping),armature='.000005')
            poly=polygon.intersection(a.rectangle(start,-a.STRAP_HEIGHT,end,a.STRAP_HEIGHT))
            solids=[a.plate_solid(poly,0,s_origin=start,merge_cells=merge_flap_colliders)]
            inertial(link,solids);add_solids(link,f'{side}_{k}',solids,4,31);parent=link
    # The U option removes the return lips for a simplified access prototype.
    thickness=a.STUD_SHEET;flange=a.STUD_FLANGE;lip=a.STUD_LIP if channel_profile=='c' else 0.
    stud=ET.SubElement(world,'body',name='stud',pos=numbers([stud_front,stud_y,a.TOP+.0008]),
                       quat=numbers([np.cos(stud_yaw/2),0.,0.,np.sin(stud_yaw/2)]))
    def block(name,center,size):
        if stud_open_towards_box:center=(center[0],flange-center[1],center[2])
        ET.SubElement(stud,'geom',name=name,type='box',pos=numbers(center),size=numbers(np.array(size)/2),
                      contype='8',conaffinity='7',rgba='.45 .52 .59 1')
    block('stud_web',(stud_depth/2,thickness/2,a.STUD_HEIGHT/2),(stud_depth,thickness,a.STUD_HEIGHT))
    for side,x in [('front',thickness/2),('rear',stud_depth-thickness/2)]:
        block('stud_'+side,(x,flange/2,a.STUD_HEIGHT/2),(thickness,flange,a.STUD_HEIGHT))
        if lip>0:
            lip_x=lip/2 if side=='front' else stud_depth-lip/2
            block('stud_'+side+'_lip',(lip_x,flange-thickness/2,a.STUD_HEIGHT/2),(lip,thickness,a.STUD_HEIGHT))
    xml=ET.tostring(root,encoding='unicode')
    return mujoco.MjModel.from_xml_string(xml),xml,moments


def addresses(model,side):
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'{side}_joint{i}') for i in range(1,9)]
    return model.jnt_qposadr[ids],model.jnt_dofadr[ids]


class Planner:
    def __init__(self,model):
        self.model=model;self.data=mujoco.MjData(model)
        self.addresses={s:addresses(model,s) for s in ['left','right']}
        self.jacp=np.zeros((3,model.nv));self.jacr=np.zeros((3,model.nv))

    def solve(self,side,position,rotation,seed):
        qa,da=self.addresses[side];qa=qa[:6];da=da[:6]
        site=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,side+'_tcp');best=None
        for initial in [seed,np.asarray(READY),np.array([0,1.8,1.9,-1,0,0])]:
            q=np.asarray(initial).copy()
            for _ in range(100):
                self.data.qpos[qa]=q;mujoco.mj_kinematics(self.model,self.data);mujoco.mj_comPos(self.model,self.data)
                ep=np.asarray(position)-self.data.site_xpos[site]
                er=ik._rot_error(rotation,self.data.site_xmat[site].reshape(3,3))
                pe,re=float(np.linalg.norm(ep)),float(np.linalg.norm(er));score=pe+.15*re
                if best is None or score<best[0]:best=(score,q.copy(),pe,re)
                if pe<.00015 and re<.002:return q,pe,re
                mujoco.mj_jacSite(self.model,self.data,self.jacp,self.jacr,site)
                jac=np.vstack([self.jacp[:,da],.15*self.jacr[:,da]])
                delta=jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),np.r_[ep,.15*er])
                q=np.clip(q+np.clip(delta,-.15,.15),ik.LOWER,ik.UPPER)
        return best[1:]


class ContactForceMonitor:
    """Track uninterrupted overload per fingertip across control periods."""
    def __init__(self,threshold_n=40.,duration_s=.020):
        self.threshold_n=threshold_n;self.duration_s=duration_s
        self.current_s=np.zeros(4);self.longest_s=np.zeros(4)
        self.triggered=np.zeros(4,dtype=bool)

    def observe(self,samples,dt):
        for forces in np.asarray(samples).reshape(-1,4):
            if not np.all(np.isfinite(forces)):raise RuntimeError('Non-finite fingertip contact forces')
            above=forces>self.threshold_n
            self.current_s=np.where(above,self.current_s+dt,0.)
            self.longest_s=np.maximum(self.longest_s,self.current_s)
            self.triggered|=above & (self.current_s+1e-12>=self.duration_s)


class Physics:
    def __init__(self,model,moments):
        self.model=model;self.data=mujoco.MjData(model);self.planner=Planner(model)
        self.addresses={s:addresses(model,s) for s in ['left','right']}
        self.targets={s:np.r_[READY,FINGER_OPEN,FINGER_OPEN] for s in self.addresses}
        for s,(qa,_) in self.addresses.items():self.data.qpos[qa]=self.targets[s]
        self.folds={}
        for name,moment in moments.items():
            i=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
            self.folds[name]=(int(model.jnt_qposadr[i]),PlasticFold(moment,stiffness=float(model.jnt_stiffness[i])))
        self.kp=np.array([80]*3+[10]*3+[5000]*2,float)
        self.housing=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'housing')
        self.peak_effort=np.zeros(16);self.peak_penetration=0.;self.worst_pair=None
        self.robot_collision=False;self.trace=[]
        self.tip_names=[side+'_tip_'+finger for side in ['left','right'] for finger in ['left','right']]
        self.geom_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(model.ngeom)]
        self.body_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or '' for g in range(model.ngeom)]
        self.geom_tip=np.array([self.tip_names.index(n) if n in self.tip_names else -1 for n in self.body_names])
        self.finger_forces=np.zeros((4,3));self.peak_finger_forces=np.zeros(4);self.last_finger_peaks=np.zeros(4)
        self.force_monitor=ContactForceMonitor()
        self.robot_collision_pair=None
        mujoco.mj_forward(model,self.data)

    def box_pose(self):
        return self.data.xpos[self.housing].copy(),self.data.xmat[self.housing].reshape(3,3).copy()

    def command(self,side,position,rotation,grip=None):
        qa,_=self.addresses[side]
        q,pe,re=self.planner.solve(side,position,rotation,self.data.qpos[qa[:6]])
        if pe>.002 or re>.025:raise RuntimeError(f'{side} IK unreachable: {pe*1000:.2f} mm, {re:.3f} rad')
        self.targets[side][:6]=q
        if grip is not None:self.targets[side][6:]=grip

    def tick(self,loads=None):
        loads=loads or {};m=self.model;d=self.data
        self.last_finger_peaks[:]=0.
        jacp=np.zeros((3,m.nv));jacr=np.zeros((3,m.nv));effective={}
        # One ordinary bounded joint target per control period. Gravity
        # compensation remains a low-level motor operation, as in the SDK.
        for side,(qa,da) in self.addresses.items():
            torque=np.zeros(8)
            if side in loads:
                site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,side+'_tcp')
                mujoco.mj_jacSite(m,d,jacp,jacr,site);load=loads[side]
                if isinstance(load,dict):
                    torque=jacp[:,da].T@np.asarray(load['force'])+jacr[:,da].T@np.asarray(load.get('moment',[0,0,0]))
                else:torque=jacp[:,da].T@np.asarray(load)
            target=self.targets[side]+torque/self.kp
            target[:6]=np.clip(target[:6],ik.LOWER,ik.UPPER)
            target[6:]=np.clip(target[6:],FINGER_OPEN,0.)
            effective[side]=target
        self.last_action=np.concatenate([np.r_[effective[side][:6],effective[side][6:].mean()/FINGER_OPEN]
                                         for side in ['left','right']]).tolist()
        for _ in range(round(CONTROL_DT/m.opt.timestep)):
            for name,(q,law) in self.folds.items():law.update(float(d.qpos[q]));m.qpos_spring[q]=law.rest
            for side,(qa,da) in self.addresses.items():
                offset=0 if side=='left' else 8
                d.ctrl[offset:offset+8]=effective[side]+d.qfrc_bias[da]/self.kp
            mujoco.mj_step(m,d)
            self.peak_effort=np.maximum(self.peak_effort,np.abs(d.actuator_force))
            self.finger_forces[:]=0;cf=np.zeros(6)
            for ci,ct in enumerate(d.contact):
                if -ct.dist>self.peak_penetration:
                    self.peak_penetration=-float(ct.dist)
                    self.worst_pair=[self.geom_names[g] for g in [ct.geom1,ct.geom2]]
                names=[self.body_names[g] for g in [ct.geom1,ct.geom2]]
                tips=[self.geom_tip[g] for g in [ct.geom1,ct.geom2]]
                if max(tips)>=0:
                    mujoco.mj_contactForce(m,d,ci,cf);force=ct.frame.reshape(3,3).T@cf[:3]
                    for tip,sign in zip(tips,[-1,1]):
                        if tip>=0:self.finger_forces[tip]+=sign*force
                if ct.dist<-.0001 and any(n.startswith('left_') for n in names) and any(n.startswith('right_') for n in names):
                    self.robot_collision=True;self.robot_collision_pair=names
            self.peak_finger_forces=np.maximum(self.peak_finger_forces,np.linalg.norm(self.finger_forces,axis=1))
            self.last_finger_peaks=np.maximum(self.last_finger_peaks,np.linalg.norm(self.finger_forces,axis=1))
            self.force_monitor.observe(np.linalg.norm(self.finger_forces,axis=1),m.opt.timestep)
        return self.snapshot()

    def snapshot(self):
        p,r=self.box_pose()
        return {'t':float(self.data.time),'box_position':p.tolist(),'box_quat_wxyz':self.data.xquat[self.housing].tolist(),
                'angles_deg':{n:float(np.rad2deg(self.data.qpos[q])) for n,(q,_) in self.folds.items()},
                'robot_joint_positions':{s:self.data.qpos[qa].tolist() for s,(qa,_) in self.addresses.items()},
                'targets':{s:self.targets[s].tolist() for s in self.targets},
                'finger_contact_forces_n':dict(zip(self.tip_names,np.linalg.norm(self.finger_forces,axis=1).tolist())),
                'action_14d':getattr(self,'last_action',None)}
