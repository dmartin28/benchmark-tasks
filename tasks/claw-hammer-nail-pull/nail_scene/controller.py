"""Contact-driven reference motions; no object pose writes or grasp constraints."""
import json
import os
from types import SimpleNamespace
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from yam_arena import ik,station
from yam_arena.constants import READY,FINGER_OPEN
from yam_arena.urdf import axis_angle_matrix
from demokit import down,_as_grip
from .scene import TOP,BOARD_SIZE,HAMMER_VERTICES,HAMMER_QUAT,claw_prong
from .material import NAIL_LENGTH


PICK_GRIP=down(-90)
GRASP_LENGTH=.180
LEVER_YAW=90.
WORK_SECTION=11


def wait(ctx,seconds):
    for _ in range(max(1,round(seconds*20))):yield


def note(ctx,phase):
    states={n:{'pos':s.pos.tolist(),'quat':s.quat_xyzw.tolist()}
            for n,s in ctx.states().items()}
    print('[motion] '+json.dumps({'time':round(ctx.t,2),'phase':phase,'states':states,
        'left_tcp':ctx.left.tcp().tolist(),'right_tcp':ctx.right.tcp().tolist()}),flush=True)


def solve_pose(arm,position,rotation):
    def error(q):
        p,R=ik.tcp_pose(q,arm.base)
        return np.r_[p-position,.2*Rotation.from_matrix(rotation@R.T).as_rotvec()]
    candidates=[]
    for seed in (arm.q,np.array(READY)):
        fit=least_squares(error,np.clip(seed,ik.LOWER+1e-5,ik.UPPER-1e-5),
                          bounds=(ik.LOWER+1e-5,ik.UPPER-1e-5),max_nfev=120)
        if np.linalg.norm(fit.fun[:3])<=.001 and np.linalg.norm(fit.fun[3:])<=.004:
            candidates.append(fit.x)
            if np.linalg.norm(fit.x-arm.q)<.5:break
    if not candidates:raise RuntimeError('Requested tool pose is outside the arm workspace')
    return min(candidates,key=lambda q:np.linalg.norm(q-arm.q))


def joint_move(arm,q,seconds,label):
    start=arm.ctx.observed(arm.side)[:6].copy()
    q=np.asarray(q)
    duration=max(seconds,1.5*float(np.max(np.abs(q-start)))/.65)
    for k in range(1,round(duration*20)+1):
        arm.q=start+(q-start)*ik.smoothstep(k/round(duration*20))
        arm.cmd=np.r_[arm.q,np.clip(arm.finger/FINGER_OPEN,0.,1.)]
        yield
    arm.grip=_as_grip(ik.tcp_pose(q,arm.base)[1]);arm.trim=np.zeros(3)
    print('[motion] '+arm.side+' '+label,flush=True)


def hold_observed(arm):
    """Unload position error at a completed contact, keeping the real grasp."""
    arm.q=arm.ctx.observed(arm.side)[:6].copy()
    arm.cmd=np.r_[arm.q,np.clip(arm.finger/FINGER_OPEN,0.,1.)]
    arm.grip=_as_grip(ik.tcp_pose(arm.q,arm.base)[1])
    arm.trim=np.zeros(3)


def checked_move(arm,position,rotation,seconds,label):
    """Use the bounded solver at every Cartesian sample, failing before a miss."""
    p0,R0=ik.tcp_pose(arm.q,arm.base)
    turn=Rotation.from_matrix(rotation@R0.T).as_rotvec()
    steps=max(1,round(seconds*20))
    for k in range(1,steps+1):
        f=ik.smoothstep(k/steps)
        R=Rotation.from_rotvec(turn*f).as_matrix()@R0
        arm.q=solve_pose(arm,p0*(1-f)+np.asarray(position)*f,R)
        arm.cmd=np.r_[arm.q,np.clip(arm.finger/FINGER_OPEN,0.,1.)]
        yield
    arm.grip=_as_grip(rotation);arm.trim=np.zeros(3)
    print('[motion] '+arm.side+' '+label,flush=True)


def tool_target(board,nail,pitch,clearance=0.,advance=0.):
    """A supported rocking pose, with the nail centered in the real claw slot.

    `advance` shifts the hammer along +x in board coordinates for insertion.
    At 130 degrees the entire blade lies below the exposed nail head, allowing
    a lateral insertion through the open V. The following rock raises it.
    """
    local_R=(axis_angle_matrix(np.array([0.,0.,1.]),np.radians(LEVER_YAW))@
             axis_angle_matrix(np.array([1.,0.,0.]),np.radians(-pitch)))
    section=claw_prong(1)[0].v.reshape(-1,4,3)[WORK_SECTION]
    index=max((0,3),key=lambda k:(local_R@section[k])[2])
    anchor=section[index].copy();anchor[0]=0.
    local_head=(nail.pos+nail.rotation()@np.array([0,0,NAIL_LENGTH])-board.pos)@board.rotation()
    root=np.r_[local_head[:2]-(local_R@anchor)[:2],
                BOARD_SIZE[2]/2-(HAMMER_VERTICES@local_R.T)[:,2].min()+clearance]
    root[0]+=advance
    return board.pos+board.rotation()@root,board.rotation()@local_R


def servo_tool(ctx,grasp,position,rotation,seconds=2.,tol=.0006,angle_tol=.6,label='tool pose'):
    """Correct compliance from the measured held tool, commanding only the arm."""
    arm=ctx.right
    cmd_p,cmd_R=ik.tcp_pose(arm.q,arm.base)
    tcp_in_tool=-grasp['relative_R'].T@grasp['offset']
    good=0
    for _ in range(round(seconds*20)):
        tool=ctx.state('hammer')
        pe=np.asarray(position)-tool.pos
        re=Rotation.from_matrix(rotation@tool.rotation().T).as_rotvec()
        if np.linalg.norm(pe)>.060 or np.linalg.norm(re)>np.radians(35):
            raise RuntimeError('Hammer pose error exceeded the safe correction range')
        if np.linalg.norm(pe)<tol and np.linalg.norm(re)<np.radians(angle_tol):
            good+=1
            if good>=3:break
        else:good=0
        # Move the grasp point through the lever arc as well as turning it.
        # Correcting only the tool root rotates the wrist in place and asks
        # finger friction to supply the entire extraction moment.
        grasp_error=(np.asarray(position)+rotation@tcp_in_tool-
                     tool.pos-tool.rotation()@tcp_in_tool)
        cmd_p+=np.clip(.25*grasp_error,-.002,.002)
        correction=.22*re
        n=np.linalg.norm(correction)
        if n>np.radians(1.2):correction*=np.radians(1.2)/n
        cmd_R=Rotation.from_rotvec(correction).as_matrix()@cmd_R
        arm.q=solve_pose(arm,cmd_p,cmd_R)
        arm.cmd=np.r_[arm.q,np.clip(arm.finger/FINGER_OPEN,0.,1.)]
        yield
    arm.grip=_as_grip(cmd_R)
    arm.trim=np.zeros(3)
    tool=ctx.state('hammer')
    error=np.linalg.norm(np.asarray(position)-tool.pos)*1000
    angle=np.linalg.norm(Rotation.from_matrix(rotation@tool.rotation().T).as_rotvec())*180/np.pi
    print(f'[motion] {label}: residual {error:.3f} mm {angle:.3f} deg',flush=True)
    return good>=3


def pick_hammer(ctx):
    r,l=ctx.right,ctx.left
    yield from wait(ctx,.4)
    initial=ctx.state('hammer')
    grip_point=initial.pos+initial.rotation()@np.array([0.,-GRASP_LENGTH,0.])+[0.,0.,.006]
    # Left prepares to steady the work while the right picks up the hammer.
    yield from ctx.together(
        r.move(grip_point+[0,0,.110],grip=PICK_GRIP,finger=-.016,t=2.,label='above hammer handle'),
        l.move((.060,.145,TOP+.150),grip=down(),finger=0.,t=2.,label='left above board'))
    yield from r.move(grip_point,t=1.5,label='descend around handle')
    yield from r.servo(grip_point,tol=.001,timeout=1.5,label='settle handle approach')
    yield from r.close(t=.7,label='grasp ordinary wooden handle')
    yield from wait(ctx,.3)
    note(ctx,'hammer grasp')
    yield from r.arc(initial.pos,(1.,0.,0.),-44.,t=3.,
                     label='rock handle upward with head supported on table')
    yield from r.move(r.at()+[0,-.100,.120],t=2.5,label='lift hammer into clear right workspace')
    yield from wait(ctx,.5)
    tool=ctx.state('hammer')
    note(ctx,'hammer lifted')
    bottom=(HAMMER_VERTICES@tool.rotation().T+tool.pos)[:,2].min()
    if bottom<TOP+.035:
        raise RuntimeError('Hammer pickup failed: tool did not follow the handle grasp')
    p,R=ik.tcp_pose(ctx.observed('right')[:6],r.base)
    return {'original':initial,'offset':R.T@(tool.pos-p),
            'relative_R':R.T@tool.rotation(),'pickup_q':ctx.observed('right')[:6].copy()}


def engage(ctx,grasp):
    r,l=ctx.right,ctx.left
    board,nail=ctx.state('board'),ctx.state('nail')
    root,Rtool=tool_target(board,nail,130.,clearance=.075,advance=.055)
    R=Rtool@grasp['relative_R'].T
    p=root-R@grasp['offset']
    outboard=np.array([.260,-.120,TOP+.260])
    q=solve_pose(r,outboard,R)
    yield from joint_move(r,q,4.,'turn hammer in clear right workspace')
    yield from r.move(p,grip=_as_grip(R),t=2.,label='carry turned hammer above board')
    yield from wait(ctx,.4)
    note(ctx,'hammer in levering orientation')
    root,Rtool=tool_target(ctx.state('board'),ctx.state('nail'),130.,clearance=.002,advance=.055)
    p=root-(Rtool@grasp['relative_R'].T)@grasp['offset']
    yield from r.move(p,grip=_as_grip(Rtool@grasp['relative_R'].T),t=2.,label='lower beside nail')
    yield from servo_tool(ctx,grasp,root,Rtool,seconds=2.,label='align open claw entrance')
    # Press the table-supported board well away from the head and lever plane.
    yield from l.move((.060,.145,TOP+.020),t=2.,label='press the long end of cedar with closed pads')
    yield from wait(ctx,.5)
    for advance in np.linspace(.055,0.,23)[1:]:
        root,Rtool=tool_target(ctx.state('board'),ctx.state('nail'),130.,clearance=.0005,advance=advance)
        yield from servo_tool(ctx,grasp,root,Rtool,seconds=.15,tol=.0005,label='claw insertion correction')
    root,Rtool=tool_target(ctx.state('board'),ctx.state('nail'),130.,clearance=.0005)
    yield from servo_tool(ctx,grasp,root,Rtool,seconds=1.5,tol=.0005,label='settle inserted claw')
    yield from wait(ctx,.4)
    note(ctx,'claw inserted')


def lever(ctx,grasp):
    r=ctx.right
    for pitch in np.linspace(130.,76.,109)[1:]:
        root,Rtool=tool_target(ctx.state('board'),ctx.state('nail'),float(pitch),clearance=0.)
        yield from servo_tool(ctx,grasp,root,Rtool,seconds=.15,tol=.0004,
                              angle_tol=.35,label=f'lever {pitch:.1f}')
        if round(pitch*2)%10==0:note(ctx,f'lever pitch {pitch:.1f}')
        board,nail=ctx.state('board'),ctx.state('nail')
        point_height=((nail.pos-board.pos)@board.rotation())[2]-BOARD_SIZE[2]/2
        if point_height>=.0011:
            hold_observed(r)
            note(ctx,'nail point clear of wood')
            return float(pitch)
    yield from wait(ctx,.5)
    note(ctx,'end levering')


def raise_freed_nail(ctx):
    r=ctx.right
    yield from wait(ctx,.4)
    hold_observed(r)
    yield from r.move(r.tcp()+[0,0,.100],t=2.,label='raise freed nail clear of board')
    yield from wait(ctx,.8)
    note(ctx,'freed nail raised for collection')


def collect_nail(ctx):
    l=ctx.left
    # Approach horizontally from behind the nail, below the claw. The prior
    # lift keeps the palm clear of the board; the nail is near the pad ends.
    angle=np.radians(90.)
    approach=np.array([np.sin(angle),0.,-np.cos(angle)])
    close=np.array([0.,-1.,0.])
    rotation=np.column_stack([close,np.cross(approach,close),approach])
    def target():
        n=ctx.state('nail')
        return n.pos+n.rotation()@np.array([0,0,NAIL_LENGTH-.028])-approach*.0095
    hold_observed(l)
    yield from l.move(l.tcp()+[0,0,.150],finger=-.009,t=1.5,label='release board pressure')
    goal=target()
    yield from l.move(goal-approach*.080,grip=_as_grip(rotation),t=2.,label='approach freed shaft from behind')
    yield from l.move(target(),t=1.5,label='put pad ends around freed shaft')
    yield from l.servo(lambda _:target(),tol=.0005,timeout=2.,label='align nail grasp')
    yield from l.close(t=.5,label='pinch the freed smooth shaft')
    yield from wait(ctx,.3)
    before=ctx.state('nail').pos.copy()
    exit_vector=ctx.state('hammer').rotation()@np.array([0.,-.035,.065])
    yield from l.move(l.at()+exit_vector,t=2.5,label='draw nail through open end of claw')
    yield from wait(ctx,.4)
    note(ctx,'nail taken from claw')
    if np.linalg.norm(ctx.state('nail').pos-before)<.035:
        raise RuntimeError('Nail did not follow the collecting hand')
    p,R=ik.tcp_pose(ctx.observed('left')[:6],l.base)
    def nail_center():
        n=ctx.state('nail')
        return n.pos+n.rotation()@np.array([0,0,NAIL_LENGTH/2])
    nail_offset=R.T@(nail_center()-p)
    delivery=down()
    tray=ctx.state('tray')
    above=tray.pos+[0,0,.160]
    yield from l.move(above-delivery.rot@nail_offset,grip=delivery,t=2.5,label='carry nail above tray')
    yield from l.servo(above,read=nail_center,tol=.002,
                       timeout=1.5,label='center nail over tray')
    release=ctx.state('tray').pos+[0,0,.055]
    yield from l.move(release-delivery.rot@nail_offset,t=2.,label='lower nail over tray opening')
    yield from l.servo(release,read=nail_center,tol=.002,
                       timeout=1.5,label='settle nail release height')
    yield from l.open(t=.5,label='release nail into tray')
    yield from wait(ctx,.8)
    hold_observed(l)
    yield from l.move(l.tcp()+[.080,0,.100],t=1.5,label='clear tray')
    yield from joint_move(l,np.array(READY),3.,'left retreat through its own workspace')
    note(ctx,'nail released in tray')


def park_hammer(ctx,parking_xy=(.315,-.240)):
    r=ctx.right
    hold_observed(r)
    tool=ctx.state('hammer')
    p,R=ik.tcp_pose(ctx.observed('right')[:6],r.base)
    grasp={'offset':R.T@(tool.pos-p),'relative_R':R.T@tool.rotation()}
    # A tall fixed carry pose can exceed the wrist range after a different
    # amount of handle rotation in the grasp. Check the observed grasp first.
    root=None
    for height in (.040,.060,.080):
        candidate=np.array([*parking_xy,TOP+height])
        goal=candidate-R@grasp['offset']
        probe=SimpleNamespace(base=r.base,q=r.q.copy())
        start=ik.tcp_pose(r.q,r.base)[0]
        try:
            for f in np.linspace(0.,1.,16):
                probe.q=solve_pose(probe,start*(1-f)+goal*f,R)
        except RuntimeError:continue
        root=candidate;break
    if root is None:raise RuntimeError('No reachable carry route for the observed hammer grasp')
    yield from checked_move(r,root-R@grasp['offset'],R,3.,'carry hammer clear of board')
    # Parking needs tabletop support, not millimetre tool-pose convergence.
    # Use reachable nominal poses; contact and arm compliance seat the head.
    flat=Rotation.from_quat(HAMMER_QUAT).as_matrix()
    leaned=Rotation.from_euler('x',-44.,degrees=True).as_matrix()@flat
    start=ctx.state('hammer').rotation()
    R=start@grasp['relative_R'].T
    root[2]=TOP-(HAMMER_VERTICES@start.T)[:,2].min()+.004
    yield from checked_move(r,root-R@grasp['offset'],R,2.,'lower hammer beside the table')
    turn=Rotation.from_matrix(leaned@start.T).as_rotvec()
    for fraction in np.linspace(0.,1.,41)[1:]:
        rotation=Rotation.from_rotvec(turn*fraction).as_matrix()@start
        root[2]=TOP-(HAMMER_VERTICES@rotation.T)[:,2].min()+.004
        R=rotation@grasp['relative_R'].T
        yield from checked_move(r,root-R@grasp['offset'],R,.20,'turn hammer close above table')
    root[2]=TOP-(HAMMER_VERTICES@leaned.T)[:,2].min()+.0005
    R=leaned@grasp['relative_R'].T
    yield from checked_move(r,root-R@grasp['offset'],R,1.,'seat head on table')
    for angle in np.linspace(-44.,-8.,19)[1:]:
        rotation=Rotation.from_euler('x',float(angle),degrees=True).as_matrix()@flat
        root[2]=TOP-(HAMMER_VERTICES@rotation.T)[:,2].min()+.0005
        R=rotation@grasp['relative_R'].T
        yield from checked_move(r,root-R@grasp['offset'],R,.25,'lower handle with head supported')
    yield from r.open(t=.5,label='leave hammer on table')
    yield from wait(ctx,.7)
    yield from retreat_right(ctx)


def retreat_right(ctx):
    r=ctx.right
    hold_observed(r)
    # The empty hand can change orientation. Raising the shoulder first avoids
    # the wrist limit caused by lifting vertically with a fixed tool attitude.
    lift=r.q.copy();lift[1]-=.25
    yield from joint_move(r,lift,1.5,'lift empty right hand clear of bench')
    yield from joint_move(r,np.array(READY),3.,'right retreat through its own workspace')
    yield from wait(ctx,1.5)
    note(ctx,'released endpoint with hands clear')


def demo_script(ctx):
    if ctx.mode=='none':
        yield from wait(ctx,3.);return
    grasp=yield from pick_hammer(ctx)
    if ctx.mode=='pickup':return
    yield from engage(ctx,grasp)
    if ctx.mode=='engage':return
    yield from lever(ctx,grasp)
    if ctx.mode=='lever':return
    yield from raise_freed_nail(ctx)
    if ctx.mode=='raise':return
    yield from collect_nail(ctx)
    if ctx.mode=='collect':return
    yield from park_hammer(ctx)
