"""Scripted full-score demo: joint targets from TCP keyframes and IK. No simulator imports.

The left arm lifts the oily-waste can's lid by pushing up under its front tab, swings it to 58 degrees
(the hinge stops at 60) and holds it. The right arm pinches the crumpled rag by its highest fold: it
comes down over the fold with the fingers open across it, both fingertips pressing on the cloth either
side, and closes, squeezing the fold between the pads. It lifts the rag, carries it high to beside the
can, slides it in over the can's right-hand wall, goes down inside under the open lid and lets go, keeping the
whole rag under the open lid; it then lifts a little and backs out the same way. The left arm lowers the lid shut. Everything else is ordinary
joint control: no object is moved, attached or edited by this script. It reads the rag's measured points
once, at the start, to find the fold to pinch (`grasp_from_nodes`), and the can's fixed geometry.
"""
import numpy as np

import can as C
import layout as L
from yam_arena import ik
from yam_arena.constants import PAD_TIP_PAST_TCP, READY

CONTROL_HZ = 20
BASES = {"left": np.array((0.0, 0.305, 0.75)), "right": np.array((0.0, -0.305, 0.75))}


def down(closing_xy, lean_deg=0.0):
    """Gripper pointing down, fingers closing along the horizontal direction `closing_xy`, the approach
    leaned `lean_deg` away from the robot about that closing axis (so both fingertips stay level)."""
    x = np.r_[np.asarray(closing_xy, float) / np.linalg.norm(closing_xy), 0.0]
    along = np.array([-x[1], x[0], 0.0])
    if along[0] < 0:
        along = -along
    a = np.radians(lean_deg)
    z = np.sin(a) * along + np.cos(a) * np.array([0.0, 0.0, -1.0])
    return ik.matrix_to_quat_xyzw(np.column_stack([x, np.cross(z, x), z]))


def tilt(pitch_deg):
    """Approach tilted from straight down (0) toward +x (90); fingers close in the vertical x-z plane."""
    a = np.radians(pitch_deg)
    z = np.array([np.sin(a), 0.0, -np.cos(a)])
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    y = np.cross(z, x)
    return ik.matrix_to_quat_xyzw(np.column_stack([x, y, z]))


DOWN_X = down((1.0, 0.0))                                  # straight down, fingers close along world x
LIFT = tilt(70.0)                                          # nearly level, pointing at the can

CX, CY = L.CAN_XY
RIM = L.TOP + C.BODY_H
HINGE = np.array([CX + C.LID_R, CY, RIM + C.LID_T / 2])
TAB_X = CX - (C.LID_R + C.TAB[0] / 2 - 0.004)
UNDER_TAB = np.array([TAB_X, CY, RIM - 0.014])     # TCP just under the tab, fingers closed
OPEN_DEG = 58.0
GRASP_OPEN = 0.42                                   # normalized finger opening (about 38 mm) over the crest
FOLD_DEPTH = 0.012                                  # fingertips only this far below the crest: a shallow pinch
                                                    # at the top, so the rag hangs wholly below the fingers
GRASP_LEAN = 15.0                                   # deg: lean of the approach along the fold (reach at this height)
ROLL = 2.6                                          # rad added to the wrist roll (joint 6) to turn the open hand over


def grasp_from_nodes(nodes):
    """Where and how to pinch the rag: its highest fold.

    Returns (xy of the crest, crest height z, horizontal closing direction across the fold). The fold's
    direction is the principal axis of the points within 5 cm of the crest and 15 mm below it; the
    fingers close across it, choosing the sign nearer world +x (less wrist turn from the ready pose).
    """
    n = np.asarray(nodes, float)
    top = n[np.argmax(n[:, 2])]
    near = n[(np.linalg.norm(n[:, :2] - top[:2], axis=1) < 0.05) & (n[:, 2] > top[2] - 0.015)]
    c = near[:, :2] - near[:, :2].mean(0)
    _, v = np.linalg.eigh(c.T @ c)
    along = v[:, 1]
    across = np.array([-along[1], along[0]])
    if across[0] < 0:
        across = -across
    return near[:, :2].mean(0), float(top[2]), across


def lid_arc(theta_deg, start=UNDER_TAB):
    """Where the point under the tab goes when the lid is open theta degrees (rotation about the hinge)."""
    a = np.radians(theta_deg)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return HINGE + rot @ (start - HINGE)


def keyframes(nodes=None):
    """Per arm: list of (end time s, kind, target, orientation, normalized grip[, wrist roll]).

    kind 'p' is a TCP point, 'arc' a lid angle in degrees. Orientation is slerped from the previous frame.
    """
    if nodes is None:
        import task
        nodes = task.rag_start("rag")
    g, crest, across = grasp_from_nodes(nodes)
    q_grasp = down(across, GRASP_LEAN)
    approach = ik.quat_xyzw_to_matrix(q_grasp)[:, 2]
    tips = np.r_[g, max(crest - FOLD_DEPTH, L.BOARD_TOP + 0.008)]
    pinch = tips - approach * PAD_TIP_PAST_TCP
    hover = pinch + (0.0, 0.0, 0.08)
    lifted = np.r_[g, L.TOP + 0.32]
    # Carry high to just outside the can's right-hand wall (clear of the open lid, which spans the can's
    # width), then keep the rag under the open lid: pinched at its top, it hangs about 15 cm below the
    # hand, and the lid's underside is about 0.42 m above the bench over the right wall and over the middle
    # of the mouth. So: over the wall with the hand at 0.395 m (rag bottom about 0.24 m; the rim is at
    # 0.218 m), on to the middle of the mouth, down to 0.32 m (12 cm clear of the left hand) and let go;
    # the rag falls to the can's floor.
    carry = np.array([CX - 0.04, CY - 0.18, L.TOP + 0.41])
    beside = np.array([CX - 0.02, CY - 0.16, L.TOP + 0.41])
    over_wall = np.array([CX, CY - 0.08, L.TOP + 0.395])
    inside = np.array([CX + 0.01, CY - 0.01, L.TOP + 0.37])
    drop = np.array([CX + 0.03, CY + 0.01, L.TOP + 0.32])
    up = drop + (0, 0, 0.02)
    park = np.array([0.20, -0.10, L.TOP + 0.38])          # high, between the board and the wastebasket
    # The ready pose leaves the right hand just above the wastebasket's mouth: lift straight up clear of
    # its rim first, then cross to the rag high.
    start, _ = ik.tcp_pose(np.array(READY, float), BASES["right"])
    clear = np.r_[start[:2], L.TOP + 0.40]
    right = [
        (1.0, "p", clear, DOWN_X, 1.0),
        (2.5, "p", hover + (0.0, 0.0, 0.02), q_grasp, GRASP_OPEN),
        (3.0, "p", hover, q_grasp, GRASP_OPEN),
        (4.0, "p", pinch, q_grasp, GRASP_OPEN),
        (6.5, "p", pinch, q_grasp, 0.0),                       # close across the fold
        (7.7, "p", lifted, q_grasp, 0.0),
        (9.7, "p", carry, DOWN_X, 0.0),
        (10.7, "p", beside, DOWN_X, 0.0),
        (11.7, "p", over_wall, DOWN_X, 0.0),
        (12.9, "p", inside, DOWN_X, 0.0),
        (13.9, "p", drop, DOWN_X, 0.0),
        (14.5, "p", drop, DOWN_X, 1.0),                        # let go inside the can, under the lid
        (15.5, "p", drop, DOWN_X, 1.0),                        # wait for the rag to fall off the fingers
        (16.3, "p", up, DOWN_X, 1.0),
        (17.3, "p", over_wall, DOWN_X, 1.0),
        (18.2, "p", beside, DOWN_X, 1.0),
        (19.4, "p", park, DOWN_X, 1.0),
    ]
    left = [
        (3.0, "p", np.array([TAB_X - 0.05, CY, RIM - 0.03]), LIFT, 0.0),
        (5.0, "p", UNDER_TAB, LIFT, 0.0),
        (7.0, "p", UNDER_TAB, LIFT, 0.0),
        (9.5, "arc", OPEN_DEG, LIFT, 0.0),
        (18.6, "arc", OPEN_DEG, LIFT, 0.0),                    # hold open until the right arm is out
        (20.6, "arc", 0.0, LIFT, 0.0),
        (22.1, "p", np.array([TAB_X - 0.07, CY, RIM - 0.05]), LIFT, 0.0),
    ]
    return {"left": left, "right": right}


END_S = 22.1
DROP_TIMES = (13.9,)        # the right arm's release point(s), for the offline clearance test


class ArmPlan:
    def __init__(self, side, frames):
        self.side, self.frames, self.base = side, frames, BASES[side]
        self.q = np.array(READY, float)
        self.prev_pos, R = ik.tcp_pose(self.q, self.base)
        self.prev_rot = np.asarray(R) if np.asarray(R).shape == (3, 3) else ik.quat_xyzw_to_matrix(R)
        self.prev_angle, self.prev_grip, self.prev_end, self.i = 0.0, 1.0, 0.0, 0
        self.prev_roll = 0.0
        self.worst_mm = 0.0
        self.last_mm = 0.0

    def target(self, t):
        while self.i < len(self.frames) - 1 and t > self.frames[self.i][0]:
            end, kind, tgt, quat, g, *rest = self.frames[self.i]
            if kind == "p":
                self.prev_pos = np.asarray(tgt, float)
            else:
                self.prev_angle, self.prev_pos = tgt, lid_arc(tgt)
            self.prev_rot = ik.quat_xyzw_to_matrix(quat)
            self.prev_grip, self.prev_end, self.prev_roll = g, end, (rest[0] if rest else 0.0)
            self.i += 1
        end, kind, tgt, quat, g, *rest = self.frames[self.i]
        s = ik.smoothstep(min(1.0, max(0.0, (t - self.prev_end) / max(1e-6, end - self.prev_end))))
        grip = self.prev_grip + s * (g - self.prev_grip)
        roll = self.prev_roll + s * ((rest[0] if rest else 0.0) - self.prev_roll)
        rot = ik.slerp_matrix(self.prev_rot, ik.quat_xyzw_to_matrix(quat), s)
        if kind == "p":
            pos = self.prev_pos + s * (np.asarray(tgt, float) - self.prev_pos)
        else:
            pos = lid_arc(self.prev_angle + s * (tgt - self.prev_angle))
        return pos, ik.matrix_to_quat_xyzw(rot), grip, roll

    def action7(self, t):
        pos, quat, grip, roll = self.target(t)
        q, pe, _ = ik.solve(pos, quat, self.q, base_pos=self.base, iters=60)
        if pe > 0.002:   # high above the bench a straight-down grip is out of reach: keep the position
            q, pe, _ = ik.solve(pos, quat, self.q, base_pos=self.base, iters=120, rot_weight=0.03)
        if pe > 0.004:
            q, pe, _ = ik.solve(pos, quat, q, base_pos=self.base, iters=200, rot_weight=0.005)
        self.worst_mm = max(self.worst_mm, 1000 * pe)
        self.last_mm = 1000 * pe
        self.q = q
        cmd = q.copy()
        cmd[5] = np.clip(cmd[5] + roll, ik.LOWER[5], ik.UPPER[5])   # wrist roll on top of the IK solution
        return np.r_[cmd, grip]


class Demo:
    """14-D absolute joint targets [left j1..j6, left grip, right j1..j6, right grip] per control step.

    `nodes`: the rag's measured points at the start of the trial (the scripted demo may read the
    simulator's state to plan; it never moves anything). Defaults to the authored starting shape.
    """

    def __init__(self, nodes=None):
        frames = keyframes(nodes)
        self.arms = {s: ArmPlan(s, frames[s]) for s in ("left", "right")}

    def action(self, t):
        t = min(t, END_S)
        return np.r_[self.arms["left"].action7(t), self.arms["right"].action7(t)]
