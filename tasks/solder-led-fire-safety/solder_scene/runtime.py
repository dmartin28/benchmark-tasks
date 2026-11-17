"""Native pose extraction, declared heat surrogate and released-endpoint scoring."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
import math
import os
import numpy as np

from yam_arena import station
from yam_arena.analysis import box_signed_distance,world_hulls,hulls_to_box
from yam_arena.constants import LINKS
from . import mechkit as m
from .scene import (TOP,WORK_TOP,PANEL_CENTER,JOINT_LOCAL,IRON_TIP,CARDBOARD_SIZE,
                    MAT_SIZE,cord_points)
from .scoring import Outcome,score,cardboard_distance_xy,mat_supports_joint,mat_is_unobstructed
from .thermal import ThermalState

SIGNS=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
LIGHT_BOUNDS=np.array([[x,y,z] for x in (-.024,.009) for y in (-.171,.113) for z in (-.001,.008)])
IRON_BOUNDS=np.array([[x,y,z] for x in (-.091,.160) for y in (-.0145,.0145) for z in (-.0145,.016)])
MOVABLE=('led_light','cardboard','soldering_mat','soldering_iron')


def corners(state,size):return SIGNS*np.asarray(size)/2@state.rotation().T+state.pos


def _box(points,c,size):return box_signed_distance(np.asarray(points),np.asarray(c),np.eye(3),np.asarray(size))


def _panels():
    yield PANEL_CENTER,np.array([.350,.450,.018])
    for y in (-.216,.216):yield PANEL_CENTER+[0.,y,-.141],np.array([.350,.018,.264])
    yield PANEL_CENTER+[.166,0.,-.141],np.array([.018,.414,.264])
    yield PANEL_CENTER+[0.,0.,-.282],np.array([.350,.450,.018])


def _carton_panels(box):
    l,w,h=CARDBOARD_SIZE;t=.003
    yield box.pos+box.rotation()@[0.,0.,-h/2+t/2],np.array([l,w,t])
    for x in (-(l-t)/2,(l-t)/2):yield box.pos+box.rotation()@[x,0.,t/2],np.array([t,w,h-t])
    for y in (-(w-t)/2,(w-t)/2):yield box.pos+box.rotation()@[0.,y,t/2],np.array([l-2*t,t,h-t])


def _hot_surface_points():
    pts=[]
    # Heater shroud and conical tip surface, sampled at 2-3 mm intervals.
    for x in np.linspace(.055,.160,44):
        r=.0055 if x<.122 else (.003 if x<.140 else max(0.,.003*(.160-x)/.020))
        for a in np.linspace(0.,2*np.pi,12,endpoint=False):pts.append([x,r*np.cos(a),r*np.sin(a)])
    return np.array(pts)


HOT_POINTS=_hot_surface_points()


def tip_surface_distance(points):
    """Distance to the actual final 20 mm of the conical iron tip, in its frame."""
    p=np.asarray(points,float)
    q=np.column_stack((p[:,0],np.linalg.norm(p[:,1:],axis=1)))
    profile=np.array([[.140,.003],[.157,.0012],[.160,0.]])
    distances=[]
    for a,b in zip(profile[:-1],profile[1:]):
        v=b-a;t=np.clip((q-a)@v/(v@v),0.,1.)
        distances.append(np.linalg.norm(q-a-t[:,None]*v,axis=1))
    distances.append(np.linalg.norm(q-np.column_stack((np.full(len(q),.140),np.clip(q[:,1],0.,.003))),axis=1))
    d=np.min(distances,axis=0)
    inside=(q[:,0]>=.140)&(q[:,0]<=.160)&(q[:,1]<=np.interp(q[:,0],profile[:,0],profile[:,1]))
    return np.where(inside,0.,d)


def arm_hulls(env,side,links):
    """Actual native link poses with the installed robot's collision meshes."""
    robot=env.unwrapped.scene[side+'_robot']
    poses=m._np(robot.data.body_link_pose_w)[0]
    origin=m._np(env.unwrapped.scene.env_origins)[0]
    transforms={}
    for name in links:
        if name not in robot.body_names:continue
        p=poses[robot.body_names.index(name)]
        T=np.eye(4);T[:3,:3]=m.quat_matrix(p[3:7]);T[:3,3]=p[:3]-origin
        transforms[name]=T
    return world_hulls(transforms,links)


def actual_hands(env):
    return np.vstack([h.vertices for side in ('left','right')
                      for name,h in arm_hulls(env,side,('gripper','tip_left','tip_right'))])


def protected_at(mat,point,surface_z):
    p=(np.asarray(point)-mat.pos)@mat.rotation()
    return (mat.rotation()[2,2]>=math.cos(math.radians(10)) and
            abs(mat.pos[2]-MAT_SIZE[2]/2-surface_z)<=.002 and
            bool(np.all(np.abs(p[:2])<=MAT_SIZE[:2]/2)) and
            point[2]>=mat.pos[2]+MAT_SIZE[2]/2-.0008)


def unsafe_geometry(iron,box,mat,joint,contact):
    if contact and cardboard_distance_xy(corners(box,CARDBOARD_SIZE),joint)<.100:return True
    points=iron.to_world(HOT_POINTS)
    # The box is hollow: test its five actual panels, not the empty interior.
    for c,size in _carton_panels(box):
        if np.any(box_signed_distance(points,c,box.rotation(),size)<=.0006):return True
    center=np.asarray(station.TABLE_CENTER);size=np.asarray(station.TABLE_SIZE)
    over=(np.abs(points[:,0]-center[0])<=size[0]/2)&(np.abs(points[:,1]-center[1])<=size[1]/2)
    for p in points[over&(points[:,2]<=TOP+.0006)]:
        if not protected_at(mat,p,TOP):return True
    for c,size in _panels():
        for p in points[_box(points,c,size)<=.0006]:
            if not protected_at(mat,p,WORK_TOP):return True
    return False


def cardboard_supported(box):
    if box.tilt_deg()>10.:return False
    floor=box.pos[2]-CARDBOARD_SIZE[2]/2
    footprint=corners(box,CARDBOARD_SIZE)[:,:2]
    for height,center,size in ((WORK_TOP,[.525,0.],[.350,.450]),
                               (TOP,np.asarray(station.TABLE_CENTER)[:2],np.asarray(station.TABLE_SIZE)[:2]),
                               (TOP+.003,np.asarray(station.TABLE_CENTER)[:2],np.asarray(station.TABLE_SIZE)[:2])):
        if abs(floor-height)>.003:continue
        if np.all(np.abs(footprint-np.asarray(center))<=np.asarray(size)/2+.002):return True
    return False


def parked(iron):
    axis=iron.rotation()[:,0]
    return (abs(iron.pos[0]-.285)<=.020 and abs(iron.pos[1]+.320)<=.010 and
            abs(iron.pos[2]-.8093)<=.008 and axis[0]>=math.cos(math.radians(8)))


def workpiece_supported(light,mat):
    if light.tilt_deg()>10.:return False
    # Endpoints must retain the work on a reproducible work surface. Throwing
    # the light away cannot earn the clearance-only partial credit.
    from .scene import CHANNEL_WIDTH,TAPE_LENGTH,CHANNEL_OFFSET
    points=light.to_world([[x,y,0.] for x in (-CHANNEL_WIDTH/2,CHANNEL_WIDTH/2)
                          for y in (CHANNEL_OFFSET-TAPE_LENGTH/2,CHANNEL_OFFSET+TAPE_LENGTH/2)])
    top=np.asarray(station.TABLE_CENTER)[:2];size=np.asarray(station.TABLE_SIZE)[:2]
    for z,c,s in ((TOP,top,size),(WORK_TOP,np.array([.525,0.]),np.array([.35,.45]))):
        if abs(light.pos[2]-z)<=.004 and np.all(np.abs(points[:,:2]-c)<=s/2+.001):return True
    local=(points-mat.pos)@mat.rotation()
    return (abs(light.pos[2]-(mat.pos[2]+MAT_SIZE[2]/2))<=.004 and
            bool(np.all(np.abs(local[:,:2])<=MAT_SIZE[:2]/2+.001)))


def hand_clearance(states,tips):
    distances=[]
    for name,s in states.items():
        if name=='cardboard':d=box_signed_distance(tips,s.pos,s.rotation(),CARDBOARD_SIZE)
        elif name=='soldering_mat':d=box_signed_distance(tips,s.pos,s.rotation(),MAT_SIZE)
        elif name=='led_light':
            lo,hi=LIGHT_BOUNDS.min(0),LIGHT_BOUNDS.max(0)
            d=box_signed_distance(tips,s.pos+s.rotation()@((lo+hi)/2),s.rotation(),hi-lo)
        else:
            lo,hi=IRON_BOUNDS.min(0),IRON_BOUNDS.max(0)
            d=box_signed_distance(tips,s.pos+s.rotation()@((lo+hi)/2),s.rotation(),hi-lo)
        distances.append(d.min())
    return max(0.,float(min(distances))*1000)


class Trial:
    def __init__(self):self.reset()

    def reset(self,env=None):
        self.heat=ThermalState(tip_c=350.)
        self.history=deque(maxlen=21)
        self.outcome=Outcome()
        self.t=0.;self.contact=False;self.tip_distance=float('inf')
        self.contact_distance=float('inf')
        self.last_bond=False
        self.switch_on=True
        self.events=[]
        self.cord_mesh=None
        self.geometry=dict(min_arm_cabinet_m=1.,min_arm_box_m=1.,min_arm_arm_aabb_m=1.)
        if env is not None:
            m.set_visible(f'{m.ENV_PATH}/led_light/visual/red_wire_joint',False)
            m.set_target(env,'iron_power_switch','power',0.)

    def step(self,env):
        dt=float(env.unwrapped.step_dt);self.t+=dt
        states={n:m.state(env,n) for n in MOVABLE}
        light,iron,box,mat=(states[n] for n in ('led_light','soldering_iron','cardboard','soldering_mat'))
        joint=light.pos+light.rotation()@JOINT_LOCAL
        touch=joint+light.rotation()@[0.,0.,.0005]
        tip=iron.pos+iron.rotation()@IRON_TIP
        if self.cord_mesh is None:
            import isaaclab.sim as sim_utils
            from pxr import UsdGeom
            self.cord_mesh=UsdGeom.Mesh(sim_utils.get_current_stage().GetPrimAtPath(f'{m.ENV_PATH}/iron_cord/visual/slack_cord'))
        from pxr import Vt
        vertices=m.tube(cord_points(iron.pos,iron.rotation()),.0025,10).v
        self.cord_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        self.tip_distance=float(np.linalg.norm(tip-touch))
        wire=JOINT_LOCAL+np.array([[0.,y,.0001] for y in (-.001,0.,.001)])
        self.contact_distance=float(tip_surface_distance(iron.to_local(light.to_world(wire))).min())
        previous_contact=self.contact
        self.contact=(self.contact_distance<=.00065 and
                      np.dot(iron.rotation()[:,0],light.rotation()[:,2])<-.2)
        if self.contact and not previous_contact:
            self.events.append(dict(t=self.t,event='tip contact',distance_mm=self.contact_distance*1000))
        angle=m.state(env,'iron_power_switch').deg('power')+24.
        new=self.switch_on
        if angle>=13.:new=True
        elif angle<=11.:new=False
        if new!=self.switch_on:
            self.switch_on=new
            # A passive over-centre spring supplies the snap, through the normal
            # joint drive. No joint-position or object-pose writes are used.
            m.set_target(env,'iron_power_switch','power',0. if new else math.radians(-24.))
        power=self.switch_on
        unsafe=unsafe_geometry(iron,box,mat,joint,self.contact) if self.heat.tip_c>=179. else False
        was_wet,was_bonded,was_unsafe=self.heat.wetted,self.heat.bonded,self.heat.unsafe
        self.heat.advance(dt,power,self.contact,unsafe)
        for changed,name in ((self.heat.wetted and not was_wet,'reflow'),
                             (self.heat.bonded and not was_bonded,'cooled bond'),
                             (self.heat.unsafe and not was_unsafe,'unsafe heat')):
            if changed:self.events.append(dict(t=self.t,event=name,tip_c=self.heat.tip_c,joint_c=self.heat.joint_c))
        if self.heat.bonded!=self.last_bond:
            m.set_visible(f'{m.ENV_PATH}/led_light/visual/red_wire_joint',self.heat.bonded)
            self.last_bond=self.heat.bonded
        self.history.append({n:(s.pos.copy(),s.rotation().copy()) for n,s in states.items()})
        stable=0.
        if len(self.history)==self.history.maxlen:
            okay=True
            for n,s in states.items():
                for frame in self.history:
                    p,R=frame[n]
                    angle=math.degrees(math.acos(np.clip((np.trace(s.rotation()@R.T)-1)/2,-1.,1.)))
                    if np.linalg.norm(s.pos-p)>.002 or angle>2.:okay=False
            if okay:stable=1.
        clear=hand_clearance(states,actual_hands(env))
        if os.environ.get('SOLDER_DIAGNOSTICS')=='1' and round(self.t/dt)%5==0:
            arms={side:arm_hulls(env,side,LINKS) for side in ('left','right')}
            for hs in arms.values():
                for c,size in _panels():
                    d,_=hulls_to_box(hs,c,np.eye(3),size)
                    self.geometry['min_arm_cabinet_m']=min(self.geometry['min_arm_cabinet_m'],d)
                # Whole-box bounds are conservative, including the empty interior.
                d,_=hulls_to_box(hs,box.pos,box.rotation(),CARDBOARD_SIZE)
                self.geometry['min_arm_box_m']=min(self.geometry['min_arm_box_m'],d)
            lo1=np.array([h.vertices.min(0) for n,h in arms['left']]);hi1=np.array([h.vertices.max(0) for n,h in arms['left']])
            lo2=np.array([h.vertices.min(0) for n,h in arms['right']]);hi2=np.array([h.vertices.max(0) for n,h in arms['right']])
            delta=np.maximum(np.maximum(lo1[:,None,:]-hi2[None,:,:],lo2[None,:,:]-hi1[:,None,:]),0.)
            d=float(np.linalg.norm(delta,axis=2).min())
            self.geometry['min_arm_arm_aabb_m']=min(self.geometry['min_arm_arm_aabb_m'],d)
        under=any(mat_supports_joint(mat.pos,mat.rotation(),MAT_SIZE,joint,z) for z in (TOP,WORK_TOP))
        self.outcome=Outcome(unsafe_heat=self.heat.unsafe,cardboard_clear=cardboard_distance_xy(corners(box,CARDBOARD_SIZE),joint)>=.100,
            cardboard_supported=cardboard_supported(box),mat_unobstructed=mat_is_unobstructed(mat.pos,mat.rotation(),MAT_SIZE,corners(box,CARDBOARD_SIZE)),
            mat_under_joint=under,joint_repaired=self.heat.bonded,iron_off=not power,objects_released=clear>=10.,
            iron_parked=parked(iron),light_supported=under and light.tilt_deg()<=3. and abs(light.pos[2]-(mat.pos[2]+MAT_SIZE[2]/2))<=.003,
            workpiece_supported=workpiece_supported(light,mat),
            hands_clear_mm=clear,stable_seconds=stable)

    def report(self):
        return dict(t=self.t,heat=asdict(self.heat),contact=self.contact,geometry=self.geometry,events=self.events,
                    tip_distance_mm=self.tip_distance*1000,contact_distance_mm=self.contact_distance*1000,
                    outcome=asdict(self.outcome),score=score(self.outcome))


TRIAL=Trial()
m.on_reset(TRIAL.reset)
m.on_step(TRIAL.step)
