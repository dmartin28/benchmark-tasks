"""Native force-limited material response and episode measurements.

The robot's contacts move the nail. No runtime pose/velocity setters are used.
Material parameters remain estimates pending measurements on physical specimens.
"""
from dataclasses import dataclass,field
import numpy as np
from . import mechkit as m
from .material import (Withdrawal,INITIAL_EMBEDMENT,ELASTIC_STIFFNESS,
                       NAIL_LENGTH,NAIL_HEAD_DIAMETER,withdrawal_force)
from .scene import (BOARD_SIZE,BOARD_START,NAIL_IN_BOARD,TOP,TRAY_SIZE,
                    claw_prong,NAIL_VERTICES)
from .scoring import Outcome,StabilityWindow,fits_tray


@dataclass
class Trial:
    material: Withdrawal = field(default_factory=Withdrawal)
    outcome: Outcome = field(default_factory=Outcome)
    claw_work_m: float = 0.
    last_extraction: float = 0.
    max_extraction: float = 0.
    stable_s: float = 0.
    previous: dict = field(default_factory=dict)
    final_state: dict = field(default_factory=dict)
    stability: StabilityWindow = field(default_factory=StabilityWindow)


TRIAL=Trial()


def _joint():
    import isaaclab.sim as sim_utils
    stage=sim_utils.get_current_stage()
    prim=stage.GetPrimAtPath('/World/envs/env_0/nail/wood_joint')
    return prim if prim.IsValid() else None


@m.before_reset
def restore_constraint(env):
    prim=_joint()
    if prim is None:return
    prim.GetAttribute('physics:jointEnabled').Set(True)
    prim.GetAttribute('physics:collisionEnabled').Set(False)
    for name,bounds in [('rotX',(-3.,3.)),('rotY',(-3.,3.))]:
        prim.GetAttribute(f'limit:{name}:physics:low').Set(bounds[0])
        prim.GetAttribute(f'limit:{name}:physics:high').Set(bounds[1])
        prim.GetAttribute(f'drive:{name}:physics:stiffness').Set(15*np.pi/180)
    prim.GetAttribute('drive:transZ:physics:targetPosition').Set(0.)
    prim.GetAttribute('drive:transZ:physics:maxForce').Set(withdrawal_force(INITIAL_EMBEDMENT))


@m.on_reset
def reset_trial(env):
    import torch
    for side in ('left','right'):
        robot=env.unwrapped.scene[f'{side}_robot']
        joints=robot.find_joints('joint[7-8]')[0]
        robot.write_joint_effort_limit_to_sim_index(
            limits=torch.full((1,len(joints)),40.,device=robot.device),joint_ids=joints)
    fresh=Trial()
    TRIAL.__dict__.clear();TRIAL.__dict__.update(fresh.__dict__)
    restore_constraint(env)


def distance_box(points,position,rotation,low,high):
    local=(np.asarray(points)-position)@rotation
    return np.linalg.norm(np.maximum(np.maximum(np.asarray(low)-local,
                                               local-np.asarray(high)),0),axis=1)


def supported(s,half_height,tilt_limit=8.):
    return (abs(s.pos[2]-half_height-TOP)<.004 and s.tilt_deg()<=tilt_limit and
            -.19<s.pos[0]<.91 and -.65<s.pos[1]<.65)


def claw_at_head(hammer,nail,board):
    """Video-equivalent evidence of a head supported by the two claw blades.

    This never grants attachment or pulls the nail; ordinary mesh collision
    must supply the force. Tool-use evidence is accumulated only on withdrawal.
    """
    head=nail.pos+nail.rotation()@np.array([0.,0.,NAIL_LENGTH])
    local=(head-hammer.pos)@hammer.rotation()
    if abs(local[0])>NAIL_HEAD_DIAMETER/2+.002:return False
    for sign in (-1,1):
        vertices=claw_prong(sign)[0].v
        delta=(vertices-local)@hammer.rotation().T
        axis=nail.rotation()[:,2]
        vertical=delta@axis
        radial=np.linalg.norm(delta-vertical[:,None]*axis,axis=1)
        if not np.any((radial<=NAIL_HEAD_DIAMETER/2+.0015)&
                      (vertical>=-.003)&(vertical<=.0015)):
            return False
    # A fulcrum on the board distinguishes levering from carrying nearby.
    crown=np.array([[x,y,z] for x in (-.010,.010)
                    for y,z in ((.015,.012),(.009,.024),(.012,-.034))])
    world=crown@hammer.rotation().T+hammer.pos
    local_board=(world-board.pos)@board.rotation()
    return bool(np.any((np.abs(local_board[:,0])<BOARD_SIZE[0]/2+.003)&
                       (np.abs(local_board[:,1])<BOARD_SIZE[1]/2+.003)&
                       (np.abs(local_board[:,2]-BOARD_SIZE[2]/2)<.005)))


@m.on_step
def update_trial(env):
    states={name:m.state(env,name) for name in ('board','nail','hammer','tray')}
    board,nail,hammer,tray=(states[n] for n in ('board','nail','hammer','tray'))
    local=(nail.pos-board.pos)@board.rotation()
    extraction=float(local[2]-NAIL_IN_BOARD[2])
    advance=max(0.,min(INITIAL_EMBEDMENT,extraction)-
                   min(INITIAL_EMBEDMENT,TRIAL.max_extraction))
    if not TRIAL.material.released and claw_at_head(hammer,nail,board):
        TRIAL.claw_work_m+=advance
    TRIAL.last_extraction=extraction
    TRIAL.max_extraction=max(TRIAL.max_extraction,extraction)
    rest,force=TRIAL.material.update(extraction)
    prim=_joint()
    if prim is None:raise RuntimeError('Missing native nail material joint')
    if TRIAL.material.released:
        prim.GetAttribute('physics:collisionEnabled').Set(True)
        prim.GetAttribute('physics:jointEnabled').Set(False)
    else:
        prim.GetAttribute('drive:transZ:physics:targetPosition').Set(float(rest))
        prim.GetAttribute('drive:transZ:physics:maxForce').Set(float(force))
        # Remaining wood supports a deeply seated shank more than its last few mm.
        angle=3. if extraction<.010 else min(45.,3.+4200*(extraction-.010))
        for name in ('rotX','rotY'):
            prim.GetAttribute(f'limit:{name}:physics:low').Set(-angle)
            prim.GetAttribute(f'limit:{name}:physics:high').Set(angle)
            stiffness=15*max(0.,1-extraction/INITIAL_EMBEDMENT)**2
            prim.GetAttribute(f'drive:{name}:physics:stiffness').Set(stiffness*np.pi/180)
    points=m.fingertips(env)
    distances=[distance_box(points,board.pos,board.rotation(),-BOARD_SIZE/2,BOARD_SIZE/2),
        distance_box(points,hammer.pos,hammer.rotation(),[-.012,-.273,-.050],[.012,.02,.061]),
        distance_box(points,nail.pos,nail.rotation(),[-.0032,-.0032,0.],[.0032,.0032,.0391]),
        distance_box(points,tray.pos,tray.rotation(),[-.07,-.055,-.0015],[.07,.055,.0185])]
    hands_clear=min(float(x.min()) for x in distances)>=.050
    dt=float(env.unwrapped.step_dt)
    poses={name:(s.pos,s.quat_xyzw) for name,s in states.items()}
    TRIAL.stable_s=TRIAL.stability.update(poses,dt,hands_clear)
    in_tray=(NAIL_VERTICES@nail.rotation().T+nail.pos-tray.pos)@tray.rotation()
    collected=bool(TRIAL.material.released and fits_tray(in_tray,TRAY_SIZE))
    invalid=any(s.pos[2]<TOP-.030 for s in states.values())
    # Actual hammer extent must reach the tabletop; root height alone is insufficient.
    from .scene import HAMMER_VERTICES
    hammer_world=HAMMER_VERTICES@hammer.rotation().T+hammer.pos
    hammer_supported=(abs(hammer_world[:,2].min()-TOP)<.004 and
        np.all((hammer_world[:,:2]>[-.19,-.65])&(hammer_world[:,:2]<[.91,.65])))
    head_local=((nail.pos+nail.rotation()@np.array([0,0,NAIL_LENGTH])-
                 board.pos)@board.rotation())
    head_rise=float(head_local[2]-(BOARD_SIZE[2]/2+NAIL_LENGTH-INITIAL_EMBEDMENT))
    TRIAL.outcome=Outcome(head_rise,TRIAL.claw_work_m,TRIAL.material.released,
        collected,supported(board,BOARD_SIZE[2]/2,5.),
        bool(hammer_supported),
        supported(tray,.0015,5.),hands_clear,TRIAL.stable_s,invalid)
    TRIAL.final_state={'extraction_mm':extraction*1000,'head_rise_mm':head_rise*1000,
        'claw_work_mm':TRIAL.claw_work_m*1000,
        'withdrawal_resistance_N':force,'released':TRIAL.material.released,
        'nail_in_tray':collected,'hands_clear':hands_clear,'stable_s':TRIAL.stable_s}
