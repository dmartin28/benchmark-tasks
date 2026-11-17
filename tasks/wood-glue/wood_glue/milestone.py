"""Measured-state check for the pickup-and-invert development milestone."""
import math
import numpy as np

from .scene import GRASP_LOCAL, TOP


def pose_status(bottle,low,tcp,gap):
    angle = math.degrees(math.acos(float(np.clip(bottle.rot[2,2],-1.,1.))))
    clearance = float(low[2]-TOP)
    grasp_error = float(np.linalg.norm(bottle.world(GRASP_LOCAL)-tcp))
    stable = np.linalg.norm(bottle.linear) < .02 and np.linalg.norm(bottle.angular) < .15
    # The bottle must be beside a closed hand, clear of the bench,
    # inverted and stable under native gravity; an ungrasped falling bottle cannot qualify.
    qualified = angle >= 170. and clearance >= .030 and grasp_error <= .012 and gap < .058 and stable
    return dict(qualified=bool(qualified),inversion_deg=angle,bench_clearance_mm=clearance*1000,
                grasp_error_mm=grasp_error*1000,gripper_gap_mm=float(gap*1000),stable=bool(stable))


def evaluate(emb,poses,states):
    from yam_arena import ik,station
    from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS
    from .runtime import arr
    robot = emb._base.scene["right_robot"]
    q = dict(zip(robot.joint_names,arr(robot.data.joint_pos)[0]))
    tcp,_ = ik.tcp_pose(np.array([q[n] for n in ARM_JOINTS]),station.BASES["right"])
    gap = -sum(q[n] for n in FINGER_JOINTS)
    return pose_status(poses["glue_bottle"],states["glue_bottle"].low,tcp,gap)


def held_glue_status(leg,leg_low,left_tcp,left_gap,bottle_status,model):
    from .controller import LEG_GRASP_LOCAL
    from .scene import LEG, RISER
    film = model.films[LEG.name+"/miter"]
    from .assembly import bead_status
    bead = bead_status(film)
    normal = leg.rot@film.surface.normal
    error = float(np.linalg.norm(leg.world(LEG_GRASP_LOCAL)-left_tcp))
    clearance = float(leg_low[2]-TOP)
    stable = np.linalg.norm(leg.linear) < .01 and np.linalg.norm(leg.angular) < .1
    coverage = film.coverage(model.settings)
    wet = float((film.thickness[film.mask] >= model.settings.min_film).mean())
    volume = float(film.volume.sum()*1e6)
    qualified = (clearance >= .040 and error <= .015 and left_gap < .060 and stable
                 and normal[2] >= math.cos(math.radians(15))
                 and bottle_status["qualified"] and not model.failed
                 and bead['qualified'])
    return dict(qualified=bool(qualified),leg_bench_clearance_mm=clearance*1000,
                leg_lift_above_support_mm=(clearance-RISER)*1000,
                leg_grasp_error_mm=error*1000,leg_stable=bool(stable),
                face_up_deg=math.degrees(math.acos(float(np.clip(normal[2],-1,1)))),
                film_coverage=coverage,wet_coverage=wet,glue_on_held_face_ml=volume,
                max_film_mm=float(film.thickness.max()*1000),bead=bead)


def evaluate_hold_glue(emb,poses,states,model):
    from yam_arena import ik,station
    from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS
    from .runtime import arr
    from .scene import LEG
    robot = emb._base.scene["left_robot"]
    q = dict(zip(robot.joint_names,arr(robot.data.joint_pos)[0]))
    tcp,_ = ik.tcp_pose(np.array([q[n] for n in ARM_JOINTS]),station.BASES["left"])
    gap = -sum(q[n] for n in FINGER_JOINTS)
    return held_glue_status(poses[LEG.name],states[LEG.name].low,tcp,gap,
                            evaluate(emb,poses,states),model)


def held_corner_status(poses,tcps,rotations,gaps,model,parked,bonded):
    """Measured grasps and glue contact for the held-corner submission endpoint."""
    from .assembly import joint_geometry,wet_contact_length
    from .scene import HEAD,LEG,LENGTH,THICK,WIDTH
    candidates = {}
    for side in ('left','right'):
        candidates[side] = {}
        for spec in (LEG,HEAD):
            local = poses[spec.name].local(tcps[side])
            error = float(np.linalg.norm([local[1],local[2]-(THICK-.0005)]))
            aligned = float(abs(rotations[side][:,0]@poses[spec.name].rot[:,1]))
            holding = (WIDTH+.003<=local[0]<=LENGTH-.015 and error<=.008 and
                       .045<=gaps[side]<=.058 and aligned>=math.cos(math.radians(10)))
            candidates[side][spec.name] = dict(piece=spec.name,grasp_error_mm=error*1000,
                along_piece_mm=float(local[0]*1000),gripper_gap_mm=gaps[side]*1000,
                aligned=bool(aligned>=math.cos(math.radians(10))),holding=bool(holding))
    # Either hand may hold either piece; grasp position along its full-width section is free.
    assignments = [(LEG.name,HEAD.name),(HEAD.name,LEG.name)]
    assignment = max(assignments,key=lambda pair:sum(candidates[side][name]['holding']
                     for side,name in zip(('left','right'),pair)))
    hands = {side:candidates[side][name] for side,name in zip(('left','right'),assignment)}
    joint = joint_geometry(poses)
    qualified = (all(h['holding'] for h in hands.values()) and model.coating_score()==.5
                 and joint['precision_seated'] and parked and wet_contact_length(model,poses)>=.040
                 and (bonded or not model.require_bond))
    return dict(qualified=bool(qualified),hands=hands,joint=joint,bottle_parked=bool(parked),bonded=bool(bonded))


def evaluate_held_corner(emb,poses,states,model):
    from yam_arena import ik,station
    from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS
    from .runtime import arr
    from .scene import BOTTLE_BASE
    tcps,rotations,gaps = {},{},{}
    for side in ('left','right'):
        robot = emb._base.scene[side+'_robot']
        q = dict(zip(robot.joint_names,arr(robot.data.joint_pos)[0]))
        tcps[side],rotations[side] = ik.tcp_pose(np.array([q[n] for n in ARM_JOINTS]),station.BASES[side])
        gaps[side] = -sum(q[n] for n in FINGER_JOINTS)
    bottle = poses['glue_bottle']
    parked = (np.max(np.abs(bottle.pos[:2]-BOTTLE_BASE[:2]))<=.004 and abs(bottle.pos[2]-BOTTLE_BASE[2])<=.004
              and bottle.rot[2,2]>=math.cos(math.radians(10)) and bottle.still())
    return held_corner_status(poses,tcps,rotations,gaps,model,parked,model.bonded)
