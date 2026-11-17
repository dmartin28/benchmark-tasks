"""Contact-driven reference motions. No object-pose writes or grasp attachments."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from yam_arena import ik
from yam_arena.urdf import axis_angle_matrix
from demokit import _as_grip,tilted
from yam_arena.constants import READY
from . import mechkit as m
from .scene import LIGHT_TARGET, CHANNEL_WIDTH, MAT_SIZE, TOP


KEYFRAMES=json.loads((Path(__file__).parents[1]/'control_keyframes.json').read_text())['keyframes']
PICK_R=np.array(KEYFRAMES[1]['rotation'])
TURN_R=axis_angle_matrix(np.array([0.,0.,1.]),-np.pi/2)


def note(ctx,label):
    states={}
    for name in ('led_light','soldering_iron','cardboard','soldering_mat'):
        s=ctx.state(name)
        states[name]=dict(pos=np.round(s.pos,6).tolist(),quat=np.round(s.quat_xyzw,6).tolist(),
                          vel=np.round(s.lin_vel,5).tolist())
    print('[motion] '+json.dumps(dict(t=round(ctx.t,2),phase=label,states=states,
              left_tcp=np.round(ctx.left.tcp(),6).tolist(),right_tcp=np.round(ctx.right.tcp(),6).tolist(),
              switch_deg=ctx.deg('iron_power_switch','power')+24.)),flush=True)


def wait(ctx,seconds):
    for _ in range(max(1,round(seconds*20))):yield


def joint_move(arm,q,seconds,label):
    """Follow a checked joint route when a Cartesian wrist turn crosses limits."""
    start=arm.ctx.observed(arm.side)[:6].copy()
    q=np.asarray(q,float)
    duration=max(seconds,1.5*float(np.max(np.abs(q-start)))/.7)
    n=max(1,round(duration*20))
    for k in range(1,n+1):
        arm.q=start+(q-start)*ik.smoothstep(k/n)
        arm.cmd=np.r_[arm.q,np.clip(arm.finger/-.04695,0.,1.)]
        yield
    arm.grip=_as_grip(ik.tcp_pose(q,arm.base)[1]);arm.trim=np.zeros(3)
    print(f'[motion] {arm.side} {label}: {duration:.2f}s',flush=True)


def solve_pose(arm,p,R):
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    def err(q):
        a,b=ik.tcp_pose(q,arm.base)
        return np.r_[a-p,.2*Rotation.from_matrix(R@b.T).as_rotvec()]
    candidates=[]
    for seed in (arm.q,np.array(READY)):
        fit=least_squares(err,seed,bounds=(ik.LOWER+1e-5,ik.UPPER-1e-5),max_nfev=180)
        if np.linalg.norm(fit.fun[:3])<.0005 and np.linalg.norm(fit.fun[3:])<.001:
            candidates.append(fit.x)
    if not candidates:raise RuntimeError('Tool pose is outside the robot workspace.')
    return min(candidates,key=lambda q:np.linalg.norm(q-arm.q))


def transfer(ctx):
    left=ctx.left
    grip=_as_grip(PICK_R)
    yield from wait(ctx,.5)
    initial=ctx.state('led_light')
    start=initial.pos.copy()
    delta=start-np.array([.490,.010,1.0475])
    note(ctx,'start')
    yield from left.move(np.array([.300,.290,1.140]),grip=grip,finger=-.011,t=2.,label='left outboard approach')
    yield from left.move(np.array([.410,.010,1.1021431])+delta,grip=grip,t=2.,label='channel pregrasp')
    yield from left.move(np.array([.450,.010,1.0621431])+delta,t=1.5,label='channel descent')
    yield from left.servo(np.array([.450,.010,1.0621431])+delta,tol=.0015,timeout=2.,label='settle channel grasp')
    note(ctx,'before closure')
    yield from left.close(width=CHANNEL_WIDTH,force=4.,t=.8,label='grasp aluminum channel')
    yield from wait(ctx,.3)
    yield from left.move(np.array([.450,.010,1.0871431])+delta,t=1.,label='channel proof lift')
    yield from wait(ctx,.3)
    note(ctx,'proof lift')
    light=ctx.state('led_light')
    if light.pos[2]-start[2]<.010:
        raise RuntimeError('Channel pickup failed: the free light did not follow the fingers.')
    if light.tilt_deg()>12.:
        raise RuntimeError(f'Channel slipped/tilted during pickup: {light.tilt_deg():.2f} degrees.')
    yield from left.move((.170,.010,1.0871431),t=3.,label='carry clear of cabinet')
    light=ctx.state('led_light')
    bounds=np.array([[x,y,z] for x in (-.024,.009) for y in (-.171,.113) for z in (-.001,.008)])
    if light.to_world(bounds)[:,0].max()>.335:
        raise RuntimeError('The carried light has not cleared the cabinet front.')
    yield from left.move((.170,.010,.8851431),t=2.,label='lower in open bench area')
    yield from left.move((.100,.020,.8851431),grip=_as_grip(TURN_R@PICK_R),t=3.,label='align light over mat')
    note(ctx,'over mat')
    target=LIGHT_TARGET+[0.,0.,.001]
    # Bring the actual object's bottom to the felt, correcting elastic arm droop.
    yield from left.move(left.at()+target-ctx.pos('led_light')+[0.,0.,.015],t=2.,label='lower light onto mat')
    ok=yield from left.servo(target,tol=.001,timeout=3.,read=lambda:ctx.pos('led_light'),label='seat channel on felt')
    if not ok:raise RuntimeError('Light placement could not reach the mat.')
    yield from left.move(left.at(),finger=-.015,t=.6,label='release light')
    yield from left.move(left.at()+[0.,0.,.100],t=1.,label='clear released light')
    left.trim=np.zeros(3)
    ready_p,ready_R=ik.tcp_pose(np.array(READY),left.base)
    yield from left.move((.180,.200,.950),grip=_as_grip(ready_R),t=2.,label='left retreat via open space')
    yield from left.move(ready_p,grip=_as_grip(ready_R),t=1.5,label='left ready')
    yield from wait(ctx,1.)
    note(ctx,'light released on mat')
    from .scoring import mat_supports_joint
    from .scene import JOINT_LOCAL
    light,mat=ctx.state('led_light'),ctx.state('soldering_mat')
    joint=light.pos+light.rotation()@JOINT_LOCAL
    if not mat_supports_joint(mat.pos,mat.rotation(),MAT_SIZE,joint,TOP):
        raise RuntimeError('Released light does not put the solder pad over the mat.')


def press_switch(ctx,on):
    r=ctx.right
    grip=tilted(65.,0.)
    # The lowest fingertip corner, rather than the palm, presses one rocker end.
    # Measured from the actual URDF fingertip collision hull, relative to TCP.
    toe=np.array([.00693936,-.00000923,-.01162154])
    point=np.array([.608 if on else .592,-.460,TOP+.0275])
    p=point-toe
    yield from r.move((.420,-.440,.960),grip=grip,finger=0.,t=1.5,label='switch outboard approach')
    yield from r.move(p+[0.,0.,.025],t=1.5,label='above rocker')
    yield from r.move(p+[0.,0.,.003],t=.8,label='rocker contact approach')
    for _ in range(70):
        angle=ctx.deg('iron_power_switch','power')+24.
        if (angle>=13. if on else angle<=11.):break
        from .runtime import arm_hulls
        vertices=np.vstack([h.vertices for n,h in arm_hulls(ctx.emb._env,'right',('tip_left','tip_right'))])
        measured=vertices[vertices[:,2]<vertices[:,2].min()+.0001].mean(0)
        correction=np.r_[np.clip(.4*(point[:2]-measured[:2]),-.0005,.0005),-.0002]
        yield from r.move(r.at()+correction,t=.05,label='press rocker with fingertip feedback')
    else:
        note(ctx,'switch press failed')
        raise RuntimeError(f'Rocker did not reach {"ON" if on else "OFF"}; angle={angle:.2f}')
    yield from r.move(r.at()+[0.,0.,.060],t=.7,label='clear rocker')
    yield from wait(ctx,.5)
    angle=ctx.deg('iron_power_switch','power')+24.
    if not (angle>=20. if on else angle<=4.):
        raise RuntimeError(f'Rocker failed to settle after the press: {angle:.2f} degrees')
    note(ctx,'iron switch '+('on' if on else 'off'))


def pick_iron(ctx):
    r=ctx.right
    grip=tilted(40.,0.)
    initial=ctx.state('soldering_iron')
    original=(initial.pos.copy(),initial.rotation().copy())
    grasp=initial.pos+initial.rotation()@np.array([-.0075,0.,0.])
    yield from r.move(grasp+[0.,0.,.100],grip=grip,finger=-.016,t=2.,label='iron pregrasp')
    yield from r.move(grasp,t=1.5,label='iron handle descent')
    yield from r.servo(grasp,tol=.001,timeout=1.5,label='settle iron grasp')
    yield from r.close(width=.022,force=8.,t=.6,label='grasp blue handle')
    yield from r.move(r.at()+[0.,0.,.002],t=.4,label='unweight iron cradles')
    yield from r.move(r.at()+[-.110,0.,0.],t=2.,label='withdraw iron axially from coil')
    yield from r.move(r.at()+[0.,0.,.150],t=1.5,label='lift extracted iron')
    yield from wait(ctx,.5)
    note(ctx,'iron lifted')
    iron=ctx.state('soldering_iron')
    if iron.pos[2]-initial.pos[2]<.100:raise RuntimeError('Iron did not follow handle grasp.')
    p,R=ik.tcp_pose(ctx.observed('right')[:6],r.base)
    return dict(original=original,offset=R.T@(iron.pos-p),relative_R=R.T@iron.rotation(),
                carry_q=ctx.observed('right')[:6].copy())


def park_iron(ctx,grasp):
    r=ctx.right
    root,Rtool=grasp['original']
    if 'solder_hover_q' in grasp:
        yield from joint_move(r,grasp['solder_hover_q'],2.,'return above joint')
        yield from joint_move(r,grasp['carry_q'],4.,'reverse checked tool route')
    # Raise the handle and point the nose slightly down while inserting. This
    # keeps the fingers above the front cradle while the tip enters the coil.
    insert_tool_R=axis_angle_matrix(np.array([0.,1.,0.]),.12)@Rtool
    R=insert_tool_R@grasp['relative_R'].T
    extracted=root+[-.110,0.,.015]
    p=extracted-R@grasp['offset']
    yield from r.move((.170,-.320,.990),grip=_as_grip(R),t=2.5,label='iron outboard return')
    yield from r.move(p+[0.,0.,.035],t=1.5,label='iron above rest entrance')
    yield from r.move(p,t=1.,label='iron rest entrance')
    yield from r.servo(extracted,tol=.001,timeout=2.,read=lambda:ctx.pos('soldering_iron'),label='align actual iron to holder')
    yield from r.move(r.at()+[.110,0.,0.],t=2.,label='insert iron into coil')
    seat_R=Rtool@grasp['relative_R'].T
    yield from r.move(root-seat_R@grasp['offset'],grip=_as_grip(seat_R),t=1.2,label='lower handle onto cradles')
    yield from r.move(r.at(),finger=-.016,t=.6,label='release iron handle')
    yield from r.move(r.at()+[0.,0.,.100],t=1.,label='clear iron rest')
    r.trim=np.zeros(3)
    yield from wait(ctx,.5)
    note(ctx,'iron parked')
    from .runtime import parked
    if not parked(ctx.state('soldering_iron')):raise RuntimeError('Iron return did not seat in its holder.')


def solder(ctx,grasp):
    from scipy.spatial.transform import Rotation
    from .scene import IRON_TIP,JOINT_LOCAL
    r=ctx.right
    direction=np.array([np.sqrt(.125),np.sqrt(.125),-np.sqrt(.75)])
    axis=np.cross([1.,0.,0.],direction)
    tool_R=(Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.arccos(direction[0])).as_matrix()
            @Rotation.from_rotvec([np.pi,0.,0.]).as_matrix())
    R=tool_R@grasp['relative_R'].T
    offset=grasp['offset']+grasp['relative_R']@IRON_TIP
    def tip():
        s=ctx.state('soldering_iron');return s.pos+s.rotation()@IRON_TIP
    def touch():
        s=ctx.state('led_light');return s.pos+s.rotation()@(JOINT_LOCAL+[0.,0.,.0005])
    hover=solve_pose(r,touch()+[0.,0.,.160]-R@offset,R)
    grasp['solder_hover_q']=hover.copy()
    yield from joint_move(r,hover,4.,'checked tool orientation route')
    yield from wait(ctx,.4)
    yield from r.move(r.at()+touch()+[0.,0.,.008]-tip(),t=2.,label='iron approach pad')
    yield from r.servo(lambda t:touch()+[0.,0.,.0015],tol=.0005,timeout=2.,read=tip,label='align actual tip over pad')
    # A small displacement, then feedback from observed tip and light poses.
    # The scorer determines reflow independently of this motion schedule.
    yield from r.servo(lambda t:touch(),tol=.00035,timeout=3.,read=tip,label='contact positive pad')
    for _ in range(32):
        yield from r.servo(lambda t:touch(),tol=.0004,timeout=.05,read=tip,label='hold gentle pad contact')
    note(ctx,'end solder contact')
    yield from r.move(r.at()+[0.,0.,.060],t=1.,label='withdraw hot tip upward')
    yield from wait(ctx,1.)


def demo_script(ctx):
    if ctx.mode=='none':
        yield from wait(ctx,3.)
        return
    if ctx.mode=='switch':
        yield from press_switch(ctx,False)
        yield from press_switch(ctx,True)
        yield from press_switch(ctx,False)
        return
    if ctx.mode=='iron':
        grasp=yield from pick_iron(ctx)
        yield from park_iron(ctx,grasp)
        return
    yield from transfer(ctx)
    if ctx.mode=='transfer':return
    grasp=yield from pick_iron(ctx)
    yield from solder(ctx,grasp)
    yield from park_iron(ctx,grasp)
    yield from press_switch(ctx,False)
    r=ctx.right
    p,R=ik.tcp_pose(np.array(READY),r.base)
    yield from r.move(p,grip=_as_grip(R),finger=-.04695,t=2.,label='right ready')
    yield from wait(ctx,2.)
    note(ctx,'finished')
