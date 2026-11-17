"""Scripted demo: pick the can, turn it upside down with a handover, squeeze the trigger, and foam set points round
the gap.

Stages (``--stage``, each includes the ones before it):
- ``turn``: the left hand picks the can off the table (angled down onto it, fingers straight across it), lifts it,
  and turns it ~97 deg about a level axis; the right hand takes its lower body, the left lets go and goes home; the
  right turns it the rest of the way, upside down.
- ``squeeze``: the right brings the can in front of the hole; the left comes in from the side with its jaw shut, the
  two fingers one thin block between the trigger and the straw, and pushes the trigger toward the can to about 40%,
  70% and 100% in turn (learning the press), then lets it spring back.
- ``ring``: the right brings the straw to the left, top and right of the gap in turn (tip 4 cm back along the straw
  from the point, then in along it, 7 mm into the gap); at each the left presses the trigger fully until about 9 cm3
  of liquid foam is out, then the straw comes back out. Between points the left lets go and drops out of the way
  while the right turns the can to the next point's attitude. Then the hands swap roles for the bottom of the gap
  (the right's grip can't hold the can there): the left takes the can's body, the right lets go, and the left holds
  the straw in the bottom while the right presses. The foam rises and spreads along the gap until the doses meet;
  the demo holds until it has visibly risen.

Grips are fixed in the can's frame; where a hand must meet the can (the handover, the trigger), its target is
recomputed every control step from the can's measured pose (live feedback), as the accepted child-resistant-cap demo
does. Offline (no simulator), the planned pose stands in. Both hands close fully when holding (the gripper's full
force) and wait for the grip to build before moving.

    python demo.py seal_pipe_gap_with_expanding_foam.py <out_dir> [--stage ring] [--max-steps 2400] [--no-record]

Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import geometry as g  # noqa: E402
from demokit import run, together, turn, wait  # noqa: E402

APPROACH_FROM = 0.08    # each hand starts this far back along its approach line
SQUEEZE_FROM = 0.07     # the presser starts this far back from the adapter
RETREAT_BACK = 0.04     # after a dose the presser stays this far out along its approach (DOSE_BACK, where it slid to)
RETREAT_DROP = 0.03     # and drops this far, below the (upside-down) can, while the can moves off: any further and
                        # its wrist (joint 4) runs onto its stop at the bottom, left and right points
PARK_BACK = 0.06        # it waits for the next point this far back along that point's approach
PARK_DROP = 0.03        # and this far down (joint margins >= 19 deg at points 0, 2, 4; offline)
GRIP_OPEN, GRIP_SHUT = 1.0, 0.0
TURN2_RISE = 0.06       # the right hand rises this much while it finishes turning the can (the 45 deg straw's tip
                        # swings within ~1 cm of the table otherwise, less the arm's sag: it caught and spun the can)
SETTLE_S = 1.0          # after closing: let the grip force build before moving
STAGES = ("turn", "squeeze", "ring")


def side_grip(approach, fingers) -> np.ndarray:
    """The tool rotation with the hand pointing along ``approach`` and the fingers closing along ``fingers``."""
    from yam_arena.demo import grip_rotation

    g0 = grip_rotation(0.0)                                  # pointing down, fingers closing along world y
    app_l, fin_l = g0.T @ [0.0, 0.0, -1.0], g0.T @ [0.0, 1.0, 0.0]
    a, f = np.asarray(approach, float), np.asarray(fingers, float)
    local = np.column_stack([app_l, fin_l, np.cross(app_l, fin_l)])
    world = np.column_stack([a, f, np.cross(a, f)])
    return world @ local.T


def approach_of(rot) -> np.ndarray:
    return rot @ (side_grip([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]).T @ np.array([0.0, 0.0, -1.0]))


def turned(angle: float) -> np.ndarray:
    """Turning by ``angle`` about the level turn axis."""
    return turn([math.cos(g.TURN_AXIS_AZ), math.sin(g.TURN_AXIS_AZ), 0.0], angle)


def _yaw(angle: float) -> np.ndarray:
    return turn([0.0, 0.0, 1.0], angle)


R_START = _yaw(g.CAN_YAW)                                  # the can's orientation as it stands on the table


def _frame(a, b) -> np.ndarray:
    e1 = np.asarray(a, float) / np.linalg.norm(a)
    e2 = np.asarray(b, float) - (np.asarray(b, float) @ e1) * e1
    e2 /= np.linalg.norm(e2)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def ring_attitude(k: int):
    """The can's orientation at set point k and the straw's direction: the straw along the planned elevation and
    azimuth, the valve as near straight down as that allows, then rolled about the straw."""
    elev, az, roll = (math.radians(v) for v in g.RING_ATTITUDES[k])
    d = np.array([math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev)])
    c = float(g.STRAW_DIR @ g.VALVE_DIR)
    down = np.array([0.0, 0.0, -1.0])
    w = down - (down @ d) * d
    w /= np.linalg.norm(w)
    v = turn(d, roll) @ (c * d + math.sqrt(1 - c * c) * w)
    return _frame(d, v) @ _frame(g.STRAW_DIR, g.VALVE_DIR).T, d


@dataclass(frozen=True)
class Plan:
    pick: np.ndarray        # the left hand's grip point on the can standing on the table
    pick_rot: np.ndarray
    pick_dir: np.ndarray    # its approach direction
    handover: np.ndarray    # the left hand's grip point at the handover
    take_c: np.ndarray      # the right hand's grip point, can frame
    take_rot_c: np.ndarray  # and rotation, can frame
    squeeze_c: np.ndarray   # the presser's tool point, can frame
    squeeze_rot_c: np.ndarray
    swap_c: np.ndarray      # the left hand's body grip once the roles are swapped (SWAP_POINT), can frame
    swap_rot_c: np.ndarray


def plan() -> Plan:
    from yam_arena import station

    pick = g.can_start_pos() + np.array([0.0, 0.0, g.PICK_HOLD])
    to = pick[:2] - np.asarray(station.BASES["left"], float)[:2]
    az = math.atan2(to[1], to[0]) + g.PICK_AZ_OFF
    level = np.array([math.cos(az), math.sin(az), 0.0])
    a = math.cos(g.PICK_TILT) * level + math.sin(g.PICK_TILT) * np.array([0.0, 0.0, -1.0])
    f = np.cross([0.0, 0.0, 1.0], level)
    handover = np.array([g.HANDOVER[0], g.HANDOVER[1], g.TOP + g.HANDOVER[2]])
    fb = np.array([math.cos(g.TAKE_FINGERS), math.sin(g.TAKE_FINGERS), 0.0])
    ab = math.cos(g.TAKE_TILT) * np.cross([0.0, 0.0, 1.0], fb) + math.sin(g.TAKE_TILT) * np.array([0.0, 0.0, 1.0])
    a_sq = g.press_approach()                          # in from the side, under the straw
    hold, phi, tilt = g.SWAP_GRIP
    fs = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi)), 0.0])
    a_s = math.cos(math.radians(tilt)) * np.cross([0.0, 0.0, 1.0], fs) + math.sin(math.radians(tilt)) * np.array([0.0, 0.0, 1.0])
    return Plan(pick, side_grip(a, f), a, handover, np.array([0.0, 0.0, g.TAKE_HOLD]), side_grip(ab, fb),
                g.press_point(0.0), side_grip(a_sq, g.PRESS_FLIP * g.press_normal()),
                np.array([0.0, 0.0, hold]), side_grip(a_s, fs))


def can_at_handover(p: Plan):
    """The can's base centre and orientation (its own frame) at the handover, as planned."""
    R = turned(g.TURN_FIRST) @ R_START
    return p.handover - R @ np.array([0.0, 0.0, g.PICK_HOLD]), R


def can_pose(ctx, p: Plan):
    """The can's base centre and orientation: measured when the simulator runs, else where the holding hand's
    command puts it (offline dry runs)."""
    if hasattr(ctx, "states"):
        s = ctx.states()["foam_can"]
        return np.asarray(s.pos, float), s.rotation()
    holder = ctx.flags.get("holder")
    if holder == "right":
        R = ctx.right.rot @ p.take_rot_c.T
        return ctx.right.pos - R @ p.take_c, R
    if holder == "left" and ctx.flags.get("swapped"):
        R = ctx.left.rot @ p.swap_rot_c.T
        return ctx.left.pos - R @ p.swap_c, R
    return can_at_handover(p)


def take_live(ctx, p: Plan, far_s: float = 2.5, in_s: float = 1.5):
    """Right hand: to its grip on the can's lower body wherever the can really is (re-aimed every control step)."""
    from yam_arena import ik

    def target(back: float):
        O, R = can_pose(ctx, p)
        rot = R @ p.take_rot_c
        return O + R @ p.take_c - back * approach_of(rot), rot

    pos, rot = target(APPROACH_FROM)
    yield from ctx.right.move(pos, rot, far_s, grip=GRIP_OPEN)
    n = max(1, round(in_s * 20))
    for k in range(1, n + 1):
        ctx.right.aim(*target(APPROACH_FROM * (1.0 - ik.smoothstep(k / n))))
        yield


def joint_move(arm, pos, rot, seconds: float, grip: float | None = None, near: bool = False):
    """Generator: move ``arm`` to the tool pose ``pos``/``rot`` by interpolating its joints (smoothstep), for long
    repositioning moves where the straight Cartesian line passes out of reach. The IK starts from the arm's current
    joints, then from a fixed set of seeds (deterministic)."""
    from yam_arena import ik
    from yam_arena.constants import READY

    seeds = [arm.q, np.array(READY, float)] + [np.array(READY, float) + d for d in
                                               np.random.default_rng(0).uniform(-1.0, 1.0, (24, 6))]
    best = None                  # among accurate solutions: the most room to the limits, or (near) the smallest swing
    for s in seeds:
        q, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=np.clip(s, ik.LOWER, ik.UPPER),
                                base_pos=arm.base, iters=150)
        room = float(min((q - ik.LOWER).min(), (ik.UPPER - q).min()))
        if room < 0.05 or err > 2e-4 or rerr > 0.003:
            continue
        key = -float(np.abs(q - arm.q).max()) if near else room     # ``near``: the smallest swing from where it is
        if best is None or key > best[1]:
            best = (q, key, err)
    q1 = best[0] if best is not None else arm.q
    arm.worst_mm = max(arm.worst_mm, (best[2] if best else 1.0) * 1000)
    q0, g0 = arm.q.copy(), arm.grip
    g1 = g0 if grip is None else grip
    n = max(1, round(seconds * 20))
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        arm.q = q0 + s * (q1 - q0)
        arm.grip = g0 + s * (g1 - g0)
        arm.pos, arm.rot = ik.tcp_pose(arm.q, arm.base)
        yield


def press_rot_c(k: int | None = None) -> np.ndarray:
    """The presser's tool rotation in the can's frame at set point k: in from the side under the straw, the pad's
    normal along the jaw (turned over where PRESS_FLIP says so)."""
    return side_grip(g.press_approach(k), g.press_params(k)[2] * g.press_normal(k))


def presser(ctx):
    """The hand working the trigger: the left, the right once the roles are swapped for the bottom point."""
    return ctx.right if ctx.flags.get("swapped") else ctx.left


PRESS_SAG = 0.028       # m: the presser's arm gives this much at a full press (~7 mm per N of the trigger's
                        # spring, measured: stage7/7b runs), pushed past the planned pose in proportion to the press
PRESS_GAIN = 0.002      # m of extra push per control step per unit of press short of the target (closed loop)
PRESS_TRIM_MAX = 0.012  # the closed loop's trim, either way
PRESS_EXTRA_MAX = 0.040 # never more than this past the planned pose
PRESS_AIM_MAX = 0.97    # the closed loop aims no further than this (the trigger's stop is at 1)


def _press_feedback(ctx) -> None:
    """The extra push toward the can past the planned pose: a feed-forward for the arm's sag under the trigger's
    spring (PRESS_SAG per full press), trimmed by a closed loop on the trigger's measured angle; while letting go, the
    trim unwinds. (Offline there is no measurement: the planned pose stands.)"""
    if not hasattr(ctx, "states"):
        return
    want = ctx.flags.get("press", 0.0)
    trim = ctx.flags.get("press_trim", 0.0)
    if want > 0.1:
        lever = float(ctx.states()["foam_can"].joints.get("lever", 0.0))
        got = float(np.clip(lever / g.LEVER_TRAVEL, 0.0, 1.0))
        trim = float(np.clip(trim + PRESS_GAIN * (min(want, PRESS_AIM_MAX) - got), -PRESS_TRIM_MAX, PRESS_TRIM_MAX))
    else:
        trim *= 0.8
    ctx.flags["press_trim"] = trim
    ctx.flags["press_extra"] = float(np.clip(PRESS_SAG * max(want, 0.0) + trim, 0.0, PRESS_EXTRA_MAX))


PRESS_RAMP_S = 1.2      # the press builds up this slowly, and lets go over RELEASE_RAMP_S: the holding arm is soft (the
RELEASE_RAMP_S = 0.8    # push moves the can ~2 cm), and its servo needs the time to hold the straw in the gap
DOSE_BACK = 0.04        # after each dose the presser slides this far out along its approach, clear of the trigger,
                        # before the straw comes out (its soft arm lags the moving can and would brush the trigger)


def _dispensed(ctx) -> float | None:
    """Liquid foam out of the can so far (m3), as seen in the simulator; None offline."""
    task = getattr(ctx, "task", None)
    return None if task is None or not hasattr(ctx, "states") else float(task.LEDGER.liquid_total)


def squeeze_target(ctx, p: Plan, back: float | None = None):
    """The presser's pose on the trigger: the shut fingers' back on the trigger's end for the current press (hovering
    just above it at 0), plus the closed loop's extra push toward the can, ``back`` metres back along its approach
    (default: the current flag ``press_back``)."""
    if back is None:
        back = ctx.flags.get("press_back", 0.0)
    O, R = can_pose(ctx, p)
    k = ctx.flags.get("press_k")
    rot = R @ press_rot_c(k)
    point = g.press_point(ctx.flags.get("press", 0.0), k) - ctx.flags.get("press_extra", 0.0) * g.press_normal(k)
    return O + R @ point - back * approach_of(rot), rot


def presser_in(ctx, p: Plan, far_s: float = 3.0, in_s: float = 1.5, close: bool = True):
    """The presser: to the trigger, wherever the can is (re-aimed every control step), its fingertip hovering just
    above the trigger's end; with ``close``, touching it."""
    from yam_arena import ik

    ctx.flags["press_back"] = 0.0
    pos, rot = squeeze_target(ctx, p, SQUEEZE_FROM)
    yield from presser(ctx).move(pos, rot, 1.0, grip=g.PRESS_GRIP)     # (already waiting there: settle on the can)
    n = max(1, round(in_s * 20))
    for k in range(1, n + 1):
        presser(ctx).aim(*squeeze_target(ctx, p, SQUEEZE_FROM * (1.0 - ik.smoothstep(k / n))))
        yield
    ctx.flags["press"] = 0.0
    if close:
        yield from press_to(ctx, p, 0.05, 0.6)     # the fingertip on the trigger (not yet past its dead zone)


def press_to(ctx, p: Plan, frac: float, seconds: float):
    """The presser: push the trigger to press fraction ``frac`` over ``seconds`` (the hand moves toward the can, the
    fingertip on the trigger's end), staying on the trigger as the can moves."""
    from yam_arena import ik

    arm = presser(ctx)
    f0 = ctx.flags.get("press", 0.0)
    n = max(1, round(seconds * 20))
    for k in range(1, n + 1):
        ctx.flags["press"] = f0 + ik.smoothstep(k / n) * (frac - f0)
        _press_feedback(ctx)
        arm.aim(*squeeze_target(ctx, p))
        yield


def let_go(ctx, p: Plan, seconds: float = 0.8):
    """The presser: lift the fingertip off the trigger."""
    yield from press_to(ctx, p, 0.0, seconds)


def follow(ctx, p: Plan, seconds: float):
    """The presser: stay on the trigger (same press) while the can moves."""
    for _ in range(max(1, round(seconds * 20))):
        _press_feedback(ctx)
        presser(ctx).aim(*squeeze_target(ctx, p))
        yield


def ring_tip(k: int) -> np.ndarray:
    """Set point k: RING_TIP_R from the pipe's axis at the point's angle, RING_IN_DEPTH into the gap."""
    th = g.ring_point_angle(k) + math.radians(g.RING_ANGLE_TRIM.get(k, 0.0))
    r = g.RING_TIP_R_AT.get(k, g.RING_TIP_R)
    return np.array([g.FACE_X + g.RING_IN_DEPTH, g.HOLE_CENTRE[1] + r * math.cos(th), g.HOLE_CENTRE[2] + r * math.sin(th)])


def ring_can_origin(k: int, inside: bool) -> np.ndarray:
    """Where the can's base centre goes for set point k: the straw tip on the point; ``inside`` False is RING_BACK
    back along the straw's own line, in front of the hole."""
    R, d = ring_attitude(k)
    tip = ring_tip(k) - (0.0 if inside else g.RING_BACK) * d
    return tip - R @ g.TIP


def hold_pose(p: Plan, k: int, inside: bool):
    """The holding (right) hand's pose for set point k, with its handover grip."""
    R, _ = ring_attitude(k)
    return ring_can_origin(k, inside) + R @ p.take_c, R @ p.take_rot_c


def hand_pose(ctx, arm):
    """The hand's tool pose: measured from its joints when the simulator runs, else as commanded."""
    from yam_arena import ik

    if hasattr(ctx, "states"):
        from scenekit import arm_joints
        from yam_arena.constants import ARM_JOINTS

        q = arm_joints(ctx.emb._env, arm.side, ARM_JOINTS)
        return ik.tcp_pose(q, arm.base)
    return arm.pos, arm.rot


SERVO_GAIN = 0.15                # share of the error corrected per control step
SERVO_STEP = 0.002               # m: the most the hand's command moves per step on account of the error
SERVO_TURN = math.radians(1.0)   # and the most it turns
SERVO_LEAD = 0.10                # m: the command never leads the measured hand by more than this (anti-windup: a
                                 # blocked or out-of-reach hand doesn't wind the command up and fling the can)
SERVO_LEAD_TURN = math.radians(15.0)
SETTLE_TOL = 0.006               # m: the can is "on" its target when its base centre is this close
SETTLE_TOL_TURN = math.radians(3.0)


def track(ctx, arm, pos, rot, ff=None) -> None:
    """One control step of closed-loop tracking: move the arm's command by a capped share of the error between the
    target and the hand's measured pose, so a hand that sags under load (the arms are springs) is brought onto the
    target over a few steps; ``ff`` (a shift and a turn) is the planned target's own motion this step, added
    straight to the command (feedforward), so a moving target isn't left behind by the capped correction. Offline
    (no simulator) the hand goes straight to the target."""
    from yam_arena import ik

    u_, _, vt_ = np.linalg.svd(np.asarray(rot, float))           # keep it a rotation (composition drifts)
    rot = u_ @ vt_
    if not hasattr(ctx, "states"):
        arm.aim(pos, rot)
        return
    Pm, Rm = hand_pose(ctx, arm)
    dp = SERVO_GAIN * (np.asarray(pos, float) - Pm)
    if np.linalg.norm(dp) > SERVO_STEP:
        dp *= SERVO_STEP / np.linalg.norm(dp)
    dR = np.asarray(rot, float) @ Rm.T                              # the measured hand's turn still to make
    angle = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
    u = SERVO_GAIN if angle * SERVO_GAIN <= SERVO_TURN or angle == 0 else SERVO_TURN / angle
    step_R = ik.slerp_matrix(np.eye(3), dR, u)
    ff_p, ff_R = ff if ff is not None else (np.zeros(3), np.eye(3))
    new_p, new_R = arm.pos + ff_p + dp, ff_R @ step_R @ arm.rot
    lead = new_p - Pm
    if np.linalg.norm(lead) > SERVO_LEAD:
        new_p = Pm + lead * SERVO_LEAD / np.linalg.norm(lead)
    lead_R = new_R @ Rm.T
    a = math.acos(max(-1.0, min(1.0, (np.trace(lead_R) - 1) / 2)))
    if a > SERVO_LEAD_TURN:
        new_R = ik.slerp_matrix(np.eye(3), lead_R, SERVO_LEAD_TURN / a) @ Rm
    from scenekit import arm_joints
    from yam_arena.constants import ARM_JOINTS

    q_meas = np.asarray(arm_joints(ctx.emb._env, arm.side, ARM_JOINTS), float)
    q, err = robust_ik(new_p, new_R, arm.base, [arm.q, q_meas])
    p_got, R_got = ik.tcp_pose(q, arm.base)
    if err > 0.003:                       # out of reach: command what is reachable (no wind-up past the edge)
        new_p, new_R = p_got, R_got
    arm.q, arm.pos, arm.rot = q, np.asarray(new_p, float), np.asarray(new_R, float)
    ctx.flags["ik_mm"] = err * 1000
    Rq = ik.tcp_pose(arm.q, arm.base)[1]
    cos = lambda M: max(-1.0, min(1.0, (np.trace(M) - 1) / 2))
    ctx.flags["ik_rot_deg"] = math.degrees(math.acos(cos(new_R @ Rq.T)))
    ctx.flags["lead_deg"] = math.degrees(math.acos(cos(new_R @ Rm.T)))
    room = np.minimum(arm.q - ik.LOWER, ik.UPPER - arm.q)
    ctx.flags["limit"] = f"j{int(np.argmin(room)) + 1}:{math.degrees(room.min()):.0f}"
    ctx.flags[f"q_{arm.side}"] = " ".join(f"{v:.0f}" for v in np.degrees(arm.q))


def robust_ik(pos, rot, base, seeds, iters: int = 80):
    """IK from each seed, and from each with the elbow (joint 3) bent a little more -- a straight elbow is a
    singularity the solver can't leave on its own; the most accurate answer, but staying near the first seed: every
    radian away from it counts as BRANCH_COST of error, so a target the warm start reaches to within a few mm never
    flips the arm into another branch (seen: the holder's sag-raised target flipped its wrist onto joint 4's stop)."""
    from yam_arena import ik

    quat = ik.matrix_to_quat_xyzw(rot)
    best = None
    q_ref = np.asarray(seeds[0], float)
    tries = list(seeds) + [np.asarray(s_, float) + np.r_[0.0, 0.0, b, 0.0, 0.0, 0.0] for s_ in seeds for b in (-0.3, -0.6, 0.3)]
    for k, s_ in enumerate(tries):
        q, err, rerr = ik.solve(pos, quat, q_init=np.clip(s_, ik.LOWER, ik.UPPER), base_pos=base, iters=iters)
        score = err + 0.02 * rerr + 1e-4 * k + BRANCH_COST * float(np.linalg.norm(q - q_ref))
        if best is None or score < best[2]:
            best = (q, err, score)
        if k == 0 and err < 1e-3:
            break                          # the warm start is fine: don't jump branches
    return best[0], float(best[1])


BRANCH_COST = 0.01      # m of IK error a radian of joint motion away from the warm start is worth (robust_ik)


def servo(ctx, p: Plan, holder, origin_at, R_target, seconds: float):
    """See servo_path; here the target orientation is fixed."""
    yield from servo_path(ctx, p, holder, origin_at, lambda u: R_target, seconds)


def servo_path(ctx, p: Plan, holder, origin_at, R_at, seconds: float):
    """Holding hand: put the can on its target pose, measured (live feedback), every control step. ``origin_at(u)``
    gives the can's target base centre for u from 0 to 1 over ``seconds``. The hand's target comes from the can's
    measured pose in the measured hand (so turning or slipping in the grip is taken out), and the hand tracks it
    closed-loop (so the arm's sag is taken out)."""
    n = max(1, round(seconds * 20))
    O, Rm = can_pose(ctx, p)
    Ph, Rh = hand_pose(ctx, holder)
    rel0_R, rel0_p = Rm.T @ Rh, Rm.T @ (Ph - O)                 # the grip at the start, for the feedforward
    prev = origin_at(0.0) + R_at(0.0) @ rel0_p, R_at(0.0) @ rel0_R
    for k in range(1, n + 1):
        O_t, R_t = origin_at(k / n), R_at(k / n)
        O, Rm = can_pose(ctx, p)
        Ph, Rh = hand_pose(ctx, holder)
        rel_R, rel_p = Rm.T @ Rh, Rm.T @ (Ph - O)
        nxt = O_t + R_t @ rel0_p, R_t @ rel0_R
        track(ctx, holder, O_t + R_t @ rel_p, R_t @ rel_R, ff=(nxt[0] - prev[0], nxt[1] @ prev[1].T))
        prev = nxt
        yield


def settle(ctx, p: Plan, holder, O_t, R_t, max_s: float = 3.0, min_s: float = 0.3):
    """Holding hand: servo the can onto a fixed pose until it is there (SETTLE_TOL) or ``max_s`` runs out."""
    n_min, n_max = round(min_s * 20), round(max_s * 20)
    for k in range(1, n_max + 1):
        yield from servo(ctx, p, holder, lambda u: O_t, R_t, 0.05)
        if k >= n_min and hasattr(ctx, "states"):
            O, Rm = can_pose(ctx, p)
            dR = R_t @ Rm.T
            ang = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
            if np.linalg.norm(O - O_t) < SETTLE_TOL and ang < SETTLE_TOL_TURN:
                return
        elif k >= n_min:
            return


AWAY_BACK = 0.10        # between set points the can goes this far away from the presser's hand
AWAY_OFF_WALL = 0.08    # first straight off the wall this far (the straw clears the pipe stub), then away
AWAY_UP_SHARE = 0.3     # how much of the away move may go up
AWAY_FROM_PRESSER = True  # pull the can straight away from the presser's hand rather than back along the straw
TURN_S = 2.4            # the turn from one point's attitude to the next (the presser repositions meanwhile)
END_HOLD_S = 14.0       # after the last dose: hold while the foam visibly rises (the run ends on success)
END_CLEAR = np.array([-0.08, -0.10, -0.04])  # then the can goes this much further, back, right and down, out of the
                                           # top camera's line to the pipe


def away_offset(O, R, presser_pos) -> np.ndarray:
    """Where the can goes, relative to where it is once off the wall, to get clear between set points: AWAY_BACK
    straight away from the presser's hand (never toward the wall)."""
    if not AWAY_FROM_PRESSER:
        return -AWAY_BACK * (R @ g.STRAW_DIR)
    u = O + R @ g.ADAPTER_C - presser_pos
    u[0] = min(u[0], 0.0)
    u[2] *= AWAY_UP_SHARE                 # mostly sideways: the holder runs out of reach upwards
    return AWAY_BACK * u / np.linalg.norm(u)


def open_jaw(ctx, p: Plan, seconds: float = 0.4):
    """The presser: let the trigger spring back and lift the fingers well clear of it."""
    yield from press_to(ctx, p, -1.0, seconds)


def presser_wait_pose(p: Plan, k: int, back: float = PARK_BACK):
    """Where the presser waits for set point k: its approach point (``back`` from the trigger) for the can at the
    point's standoff, as planned."""
    R_k, _ = ring_attitude(k)
    rot = R_k @ press_rot_c(k)
    return ring_can_origin(k, inside=False) + R_k @ g.press_point(0.0, k) - back * approach_of(rot), rot


def visit(ctx, p: Plan, k: int, holder, first: bool = False, fresh: bool = False):
    """One set point. Unless it's the first (where the presser is already on the trigger with the can at the
    standoff): the presser lets go, backs off along its approach and drops below the can (which is upside down, its
    body above the presser's hand); the holder takes the can straight off the wall, then away from the presser, and
    turns it to this point's attitude while the presser goes to its parked pose for this point; the holder brings the
    can back in to the standoff and settles it there; only then does the presser rise and come in onto the trigger
    (aimed live) -- the can never moves towards a waiting hand. Then the straw goes in along its own line, the
    presser squeezes fully for DOSE_S, lets the lever spring back and opens, and the straw comes back out. ``fresh``:
    the presser has just let go of the can (the swap), so it has no trigger to let go of first."""
    from yam_arena import ik

    ctx.flags.update(phase=f"ring{k}", at_ring=True)
    R_k, d_k = ring_attitude(k)
    O_out, O_in = ring_can_origin(k, inside=False), ring_can_origin(k, inside=True)
    if first:
        # the presser came in open round the trigger at the standoff (a closed hand drags the can with its own sag
        # as the straw goes in); settle the can at the standoff before going in
        ctx.flags["sub"] = "settle"
        yield from settle(ctx, p, holder, O_out, R_k)            # (the open jaw has ~3 cm to spare each side)
    else:
        ctx.flags["sub"] = "release"
        if not fresh:
            yield from together(open_jaw(ctx, p), follow(ctx, p, 0.4))
            back_pos, back_rot = squeeze_target(ctx, p, RETREAT_BACK)
            yield from presser(ctx).move(back_pos, back_rot, 0.2)      # (already about there: DOSE_BACK)
            yield from presser(ctx).move(back_pos - [0.0, 0.0, RETREAT_DROP], back_rot, 0.6)
        O0, R0 = can_pose(ctx, p)
        off = np.array([AWAY_OFF_WALL, 0.0, 0.0])
        O_off, O_off_k = O0 - off, O_out - off
        # (after the swap the can turns in place off the wall: moved away from the right hand it would be pushed
        # toward the left arm's own side, which runs its wrist onto its stop; offline)
        O_away = O_off if fresh else O_off + away_offset(O0, R0, presser(ctx).pos)
        ctx.flags["press_k"] = k                    # from here on the presser works this point's trigger pose
        park_pos, park_rot = presser_wait_pose(p, k, PARK_BACK)
        park_pos = park_pos - [0.0, 0.0, PARK_DROP]
        O_far = O_off_k if fresh else O_off_k + away_offset(O_out, R_k, park_pos)
        ctx.flags["sub"] = "away"
        yield from servo(ctx, p, holder, lambda u: O0 + ik.smoothstep(u) * (O_off - O0), R0, 0.8)
        yield from servo(ctx, p, holder, lambda u: O_off + ik.smoothstep(u) * (O_away - O_off), R0, 1.0)
        ctx.flags["sub"] = "turn"                    # the presser repositions while the can turns
        yield from together(servo_path(ctx, p, holder, lambda u: O_away + ik.smoothstep(u) * (O_far - O_away),
                                       lambda u: ik.slerp_matrix(R0, R_k, ik.smoothstep(u)), TURN_S),
                            joint_move(presser(ctx), park_pos, park_rot, TURN_S, grip=g.PRESS_GRIP, near=True))
        ctx.flags["sub"] = "in"
        yield from servo(ctx, p, holder, lambda u: O_far + ik.smoothstep(u) * (O_off_k - O_far), R_k, 1.5)
        yield from servo(ctx, p, holder, lambda u: O_off_k + ik.smoothstep(u) * (O_out - O_off_k), R_k, 1.2)
        ctx.flags["sub"] = "settle"
        yield from settle(ctx, p, holder, O_out, R_k)
        ctx.flags["sub"] = "rise"
        n = 40
        hold = servo(ctx, p, holder, lambda u: O_out, R_k, n / 20)

        def come_in():
            ctx.flags["press_back"] = 0.0
            for j in range(1, n + 1):
                s_ = ik.smoothstep(j / n)
                drop = PARK_DROP * max(0.0, 1.0 - 2.0 * s_)                    # rise first, then in
                back = PARK_BACK * (1.0 - max(0.0, 2.0 * s_ - 1.0))
                pos, rot = squeeze_target(ctx, p, back)
                presser(ctx).aim(pos - [0.0, 0.0, drop], rot)
                yield

        yield from together(hold, come_in())
    # in along the straw's line, then settle
    ctx.flags["sub"] = "squeeze"
    yield from together(servo(ctx, p, holder, lambda u: O_out + ik.smoothstep(u) * (O_in - O_out), R_k, 1.2),
                        follow(ctx, p, 1.2))
    yield from together(servo(ctx, p, holder, lambda u: O_in, R_k, 0.6), follow(ctx, p, 0.6))

    done = []

    def squeeze():
        yield from press_to(ctx, p, 0.0, 0.4)                # the fingertip just above the trigger
        start = _dispensed(ctx)
        yield from press_to(ctx, p, g.DOSE_PRESS, PRESS_RAMP_S)  # slowly: the holder's servo takes up the push
        # hold until the dose is out, watching the bead (the trigger's travel sets the flow, and how far the hand
        # gets it down varies); offline, for DOSE_S
        for j in range(round(g.DOSE_S_MAX * 20)):
            got = _dispensed(ctx)
            if got is None and j >= round(g.DOSE_S * 20):
                break
            if got is not None and got - start >= (g.DOSE_CC - g.DOSE_TAIL_CC) * 1e-6:
                break
            yield from follow(ctx, p, 0.05)
        yield from press_to(ctx, p, 0.0, RELEASE_RAMP_S)
        yield from follow(ctx, p, 0.3)                       # let the lever spring back
        yield from open_jaw(ctx, p)
        m = 10                                               # slide out sideways, clear of the trigger
        for j in range(1, m + 1):
            ctx.flags["press_back"] = DOSE_BACK * ik.smoothstep(j / m)
            yield from follow(ctx, p, 0.05)
        done.append(True)

    def hold():
        while not done:
            yield from servo(ctx, p, holder, lambda u: O_in, R_k, 0.05)

    yield from together(hold(), squeeze())
    yield from together(servo(ctx, p, holder, lambda u: O_in + ik.smoothstep(u) * (O_out - O_in), R_k, 0.8),
                        follow(ctx, p, 0.8))


def script(ctx, stage: str = "ring", points=None):
    from yam_arena import ik

    p = plan()
    L, R = ctx.left, ctx.right
    ctx.flags.update(holder=None, phase="pick", press=0.0, at_ring=False)
    # 1. left: in along its line, grip, lift straight up
    yield from L.move(p.pick - APPROACH_FROM * p.pick_dir + [0, 0, 0.04], p.pick_rot, 2.5, grip=GRIP_OPEN)
    yield from L.move(p.pick - APPROACH_FROM * p.pick_dir, p.pick_rot, 1.0)
    yield from L.move(p.pick, p.pick_rot, 1.5)
    yield from L.set_grip(GRIP_SHUT, 1.0)
    ctx.flags.update(holder="left")
    yield from wait(SETTLE_S)
    lift = p.pick + [0.0, 0.0, g.PICK_LIFT]
    yield from L.move(lift, p.pick_rot, 3.0)
    # 2. left: turn the first part while carrying the can to the handover point
    ctx.flags.update(phase="turn")
    steps = 12
    for k in range(1, steps + 1):
        u = k / steps
        yield from L.move(lift + u * (p.handover - lift), turned(g.TURN_FIRST * u) @ p.pick_rot, 4.0 / steps)
    yield from wait(0.5)
    # 3. right: to the lower body wherever the can is, grip; left lets go and backs off
    ctx.flags.update(phase="handover")
    yield from take_live(ctx, p)
    yield from R.set_grip(GRIP_SHUT, 1.0)
    yield from wait(SETTLE_S)
    yield from L.set_grip(GRIP_OPEN, 0.8)
    ctx.flags.update(holder="right")
    back = turned(g.TURN_FIRST) @ p.pick_dir
    yield from L.move(p.handover - APPROACH_FROM * back, None, 1.5)
    # 4. right: turn the rest in place while the left goes home
    ctx.flags.update(phase="turn2")
    rest = g.TURN_TOTAL - g.TURN_FIRST
    axis = [math.cos(g.TURN_AXIS_AZ), math.sin(g.TURN_AXIS_AZ), 0.0]
    hand0, rot0 = R.pos.copy(), R.rot.copy()

    def finish():
        n = 10
        for k in range(1, n + 1):                  # rising as it turns: the straw swings down past the table
            yield from R.move(hand0 + [0.0, 0.0, TURN2_RISE * k / n], turn(axis, rest * k / n) @ rot0, 3.0 / n)

    yield from together(finish(), L.go_home(3.0))
    yield from wait(1.0)
    if stage == "turn":
        yield from wait(2.0)
        return
    # 5. right: bring the can in front of the hole, lined up with the first set point but well back (no foam reaches
    #    the wall from here)
    ctx.flags.update(phase="to_ring")
    order = [k for k in g.RING_ORDER if points is None or k in points]
    if not order:
        raise SystemExit("--points needs at least one of the right-held points " + str(g.RING_ORDER))
    first = order[0]                         # (with --points, the first point visited)
    # the left gets into place first, while the can is still out of the way (its joint-space path would cut through
    # the can once the can is in front of the hole): its approach point, worked out for the can at the first standoff
    R_first, _ = ring_attitude(first)
    ctx.flags["press_k"] = first
    wait_pos, wait_rot = presser_wait_pose(p, first, SQUEEZE_FROM)
    yield from joint_move(L, wait_pos, wait_rot, 3.0, grip=g.PRESS_GRIP)
    end_pos, end_rot = hold_pose(p, first, inside=False)
    yield from R.move(end_pos, end_rot, 3.2)
    yield from settle(ctx, p, R, ring_can_origin(first, inside=False), R_first)     # take out the arm's sag
    # 6. left: to the trigger; learn the squeeze (40%, 70%, full), then let it spring back to a light touch
    ctx.flags.update(phase="squeeze")
    yield from presser_in(ctx, p, close=(stage == "squeeze"))
    if stage == "squeeze":
        for frac in (0.4, 0.7, 1.0):
            yield from press_to(ctx, p, frac, 0.5)
            yield from follow(ctx, p, g.SQUEEZE_S)
            yield from press_to(ctx, p, 0.0, 0.5)
            yield from follow(ctx, p, 0.5)
        yield from let_go(ctx, p)
        yield from wait(2.0)
        return
    # 7. the ring: at each set point, straw in along its own line, squeeze, straw out, on to the next
    for n_, k in enumerate(order):
        yield from visit(ctx, p, k, R, first=(n_ == 0 and k == first))
    holder = R
    # 8. the bottom: the hands swap roles (the right's grip can't hold the can there), then the left holds the straw in
    #    the bottom of the gap and the right presses
    if points is None or g.SWAP_POINT in points:
        yield from swap(ctx, p)
        holder = L
        yield from visit(ctx, p, g.SWAP_POINT, L, fresh=True)
    # 9. done: the presser lets go, backs off along its approach and goes home; the holder takes the straw back off the
    #    wall and carries the can out of the top camera's view of the pipe while the foam finishes rising (the run
    #    ends when the task succeeds)
    ctx.flags.update(phase="done", at_ring=False, sub="-")
    pos, rot = squeeze_target(ctx, p, RETREAT_BACK)
    yield from presser(ctx).move(pos, rot, 0.8)
    O0, R0 = can_pose(ctx, p)
    off = np.array([AWAY_OFF_WALL, 0.0, 0.0])
    yield from together(servo(ctx, p, holder, lambda u: O0 - ik.smoothstep(u) * off, R0, 1.2),
                        presser(ctx).move(pos - [0.0, 0.0, RETREAT_DROP], rot, 1.2))
    O_clear = O0 - off + END_CLEAR
    yield from together(servo(ctx, p, holder, lambda u: O0 - off + ik.smoothstep(u) * END_CLEAR, R0, 2.0),
                        presser(ctx).go_home(2.0))
    yield from together(servo(ctx, p, holder, lambda u: O_clear, R0, END_HOLD_S), evidence(ctx, END_HOLD_S))


SWAP_UP = 0.0           # the left's body grip is approached from APPROACH_FROM back along its approach, this much higher
SWAP_ROUND = 0.03       # and on the way there it bows this far out from the can's axis
SWAP_RELEASE_BACK = 0.07  # the right backs off its grip this far along its approach before shutting its jaw


def swap(ctx, p: Plan):
    """The roles swap at the last right-held point's standoff: the left (just off the trigger, slid out along its
    approach) goes round to its body grip on the can (straight moves: a joint-space path could swing the wrist through
    the can), comes in aimed live and closes; the right opens and backs off along its approach. From here the left
    holds and the right presses."""
    from yam_arena import ik

    L, R = ctx.left, ctx.right
    ctx.flags.update(phase="swap", sub="-")
    pos, rot = squeeze_target(ctx, p, RETREAT_BACK)
    yield from L.move(pos - [0.0, 0.0, RETREAT_DROP], rot, 0.4)
    O, Rc = can_pose(ctx, p)
    body_rot = Rc @ p.swap_rot_c
    offset = -APPROACH_FROM * approach_of(body_rot) + np.array([0.0, 0.0, SWAP_UP])
    start = O + Rc @ p.swap_c + offset
    mid = 0.5 * (L.pos + start)                  # round the can: the straight line passes within a mm of its body
    axis = Rc[:, 2]
    out = (mid - O) - ((mid - O) @ axis) * axis
    mid = mid + SWAP_ROUND * out / np.linalg.norm(out)
    yield from L.move(mid, ik.slerp_matrix(L.rot, body_rot, 0.5), 1.0)  # the jaw still shut (opening on the way
    yield from L.move(start, body_rot, 1.0)                            # brushes the can), then open 8 cm off it
    yield from L.set_grip(GRIP_OPEN, 0.6)
    n = 30
    for j in range(1, n + 1):
        O, Rc = can_pose(ctx, p)
        L.aim(O + Rc @ p.swap_c + (1.0 - ik.smoothstep(j / n)) * offset, Rc @ p.swap_rot_c)
        yield
    yield from L.set_grip(GRIP_SHUT, 1.0)
    yield from wait(SETTLE_S / 2)
    yield from R.set_grip(GRIP_OPEN, 0.8)
    ctx.flags.update(holder="left", swapped=True, press=0.0, press_back=0.0, press_trim=0.0, press_extra=0.0)
    yield from R.move(R.pos - SWAP_RELEASE_BACK * approach_of(R.rot), None, 1.0)
    yield from R.set_grip(g.PRESS_GRIP, 0.5)     # shut, clear of the can (it slides past as it comes off the wall)


EVIDENCE_DIR: list = []         # set by the command line: where closeup.png and score_report.json go
CLOSEUP_BACK = 0.25             # the close-up camera sits on the pipe's axis this far in front of the wall face (off the
                                # axis, the pipe stub would hide part of the ring)


def evidence(ctx, seconds: float):
    """Hold for ``seconds``; meanwhile, in the simulator, every half second save a close-up of the hole's face
    (closeup.png: the foam drawn as markers) and score_report.json (the score's ingredients), so the last ones written
    match the run's end."""
    ann = None
    if hasattr(ctx, "states") and EVIDENCE_DIR:
        try:
            ann = _closeup_annotator()
        except Exception as error:     # evidence only: never stop the demo for it
            print(f"close-up camera unavailable: {error}", flush=True)
    for k in range(max(1, round(seconds * 20))):
        yield
        if EVIDENCE_DIR and k % 10 == 9:
            _write_evidence(ctx, ann)


def _closeup_annotator():
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cam = UsdGeom.Camera.Define(stage, "/World/closeup_cam")
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 50.0))
    cam.GetFocalLengthAttr().Set(40.0)
    e = np.asarray(g.HOLE_CENTRE, float) - [CLOSEUP_BACK, 0.0, 0.0]
    z = (e - g.HOLE_CENTRE) / np.linalg.norm(e - g.HOLE_CENTRE)      # a USD camera looks down its -z, +y up
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(*map(float, x), 0.0, *map(float, y), 0.0,
                                                            *map(float, z), 0.0, *map(float, e), 1.0))
    # Arena renders each environment in its own scene partition; without this the camera sees only the markers
    cam.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    product = rep.create.render_product(cam.GetPath().pathString, (960, 720))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([product])
    return ann


def _write_evidence(ctx, ann) -> None:
    import json

    import foam

    out = pathlib.Path(EVIDENCE_DIR[0])
    out.mkdir(parents=True, exist_ok=True)
    led = ctx.task.LEDGER
    fill, bulge = led.fill(cured=True), led.bulge_heights()
    report = {
        "step": int(ctx.task._STATE.get("steps", 0)),
        "score": foam.score(led),
        "sectors_sealed": int((fill >= g.SECTOR_FULL).sum()),
        "sectors": g.SECTORS,
        "max_bulge_mm": round(float(bulge.max()) * 1000, 1),
        "liquid_in_gap_cc": round(led.ring_liquid() * 1e6, 2),
        "liquid_outside_gap_cc": round(sum(v for _, v, _ in led.mess) * 1e6, 2),
        "rise_shown": round(led.rise_shown(), 3),
        "fill_per_sector": [round(float(v), 3) for v in fill],
        "bulge_mm_per_sector": [round(float(v) * 1000, 1) for v in bulge],
    }
    (out / "score_report.json").write_text(json.dumps(report, indent=1))
    if ann is not None:
        try:
            # the env's renderer updates only its own cameras; this extra render product needs the RTX update forced
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

            ensure_isaac_rtx_render_update(force=True)
            data = np.asarray(ann.get_data())
            if data.size and data.ndim == 3:
                from PIL import Image

                Image.fromarray(data[..., :3].astype(np.uint8)).save(out / "closeup.png")
        except Exception as error:
            print(f"close-up not saved: {error}", flush=True)


def _measured_slip(ctx, can) -> float:
    """How far the holding hand's tool point has moved along the can's axis from where it gripped (m)."""
    from scenekit import arm_joints
    from yam_arena import ik
    from yam_arena.constants import ARM_JOINTS

    side = ctx.flags.get("holder")
    if side is None:
        return 0.0
    arm = ctx.left if side == "left" else ctx.right
    q = arm_joints(ctx.emb._env, side, ARM_JOINTS)
    tcp, _ = ik.tcp_pose(q, arm.base)
    along = float((can.rotation().T @ (tcp - np.asarray(can.pos)))[2])
    if side == "left" and ctx.flags.get("swapped"):
        return along - g.SWAP_GRIP[0]
    return along - (g.TAKE_HOLD if side == "right" else g.PICK_HOLD)


def _ring_offsets(ctx, can) -> str:
    """At a ring point: the straw tip's offset (mm, world x y z) from the set point, and the holding hand's lag
    (mm, measured tool point to its command)."""
    phase = ctx.flags.get("phase", "")
    if phase in ("handover", "turn2", "to_ring", "squeeze") and ctx.flags.get("holder") == "right":
        return _grip_turn(ctx, can)
    if not phase.startswith("ring"):
        return ""
    k = int(phase[4:])
    tip = np.asarray(can.pos) + can.rotation() @ g.TIP
    d = (tip - ring_tip(k)) * 1000
    holder = getattr(ctx, ctx.flags.get("holder") or "right")
    lag = (hand_pose(ctx, holder)[0] - holder.pos) * 1000
    return (f"  tip-target {d[0]:+6.1f} {d[1]:+6.1f} {d[2]:+6.1f} mm  hand lag {np.linalg.norm(lag):5.1f} mm"
            f"  ik {ctx.flags.get('ik_mm', 0.0):5.1f} mm rot {ctx.flags.get('ik_rot_deg', 0.0):4.1f} lead "
            f"{ctx.flags.get('lead_deg', 0.0):4.1f} deg limit {ctx.flags.get('limit', '-')} q_right [{ctx.flags.get('q_right', '')}]"
            + _grip_turn(ctx, can))


_PLAN = []


def _grip_turn(ctx, can) -> str:
    """How far the can has turned in the right hand's grip from the planned grip (deg), and the right arm's joints."""
    from scenekit import arm_joints
    from yam_arena.constants import ARM_JOINTS

    if not _PLAN:
        _PLAN.append(plan())
    _, Rh = hand_pose(ctx, ctx.right)
    D = (can.rotation().T @ Rh) @ _PLAN[0].take_rot_c.T
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(D) - 1) / 2))))
    ax = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    Hc = can.rotation().T @ (hand_pose(ctx, ctx.right)[0] - np.asarray(can.pos))     # the hand's tool point, can frame
    q = np.degrees(np.asarray(arm_joints(ctx.emb._env, "right", ARM_JOINTS), float))
    return f"  grip turn {ang:4.1f} deg about can [{ax[0]:+.3f} {ax[1]:+.3f} {ax[2]:+.3f}] tcp_c [{Hc[0]:+.4f} {Hc[1]:+.4f} {Hc[2]:+.4f}]  q_right_meas [{' '.join(f'{v:.0f}' for v in q)}] cmd [{' '.join(f'{v:.0f}' for v in np.degrees(ctx.right.q))}]"


def log(ctx, step, info) -> str:
    S = ctx.task._STATE
    can = ctx.states()["foam_can"]
    led = ctx.task.LEDGER
    return (f"{ctx.flags.get('phase', ''):9s} {ctx.flags.get('sub', '-'):8s} holder {ctx.flags.get('holder')}  slip {_measured_slip(ctx, can) * 100:+5.1f} cm  "
            f"valve {S['valve_from_down_deg']:5.1f} deg  lever {S['lever_deg']:4.1f} deg  press {S['press']:.2f}  "
            f"push +{ctx.flags.get('press_extra', 0.0) * 1000:4.1f} mm  "
            f"flow {S['flow_cc_s']:4.2f} cc/s  sector {S['sector']}  liquid {led.liquid_total * 1e6:5.2f} cc  "
            f"(ring {led.ring_liquid() * 1e6:5.2f}, mess {sum(v for _, v, _ in led.mess) * 1e6:5.2f})  "
            f"filled {led.coverage(cured=False) * 100:3.0f}% (cured {led.coverage() * 100:3.0f}%)  "
            f"bulge {led.bulge_heights().max() * 1000:4.1f} mm  rise {led.rise_shown():.2f}"
            + _ring_offsets(ctx, can))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--stage", choices=STAGES, default="ring")
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--points", default=None, help="comma-separated set points to visit (diagnostics)")
    a = ap.parse_args()
    pts = None if a.points is None else {int(x) for x in a.points.split(",")}
    EVIDENCE_DIR.append(a.out)
    run(a.task, a.out, lambda ctx: script(ctx, a.stage, pts), max_steps=a.max_steps, record=not a.no_record, log=log,
        log_every=a.log_every)
