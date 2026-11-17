"""Scoring rules on measured states: pure numpy, no simulator. Used by task.py and the tests.

A rag is "in the can" only when every point of it is inside the can's shell, below the rim. Thin simulated
cloth pressed against the 5 mm wall collider can push a point into or just through it, so the radius test
allows 3 mm past the wall's outer surface; fabric draped over the wall always has points at or above the
rim, which the rim test rejects.
A rag is "in the bin" when any point of it is inside the open wastebasket, below its rim.
The lid is "shut" at 3 degrees or less (a rag caught under the lid props it open further, or leaves
fabric above the rim, so the rag no longer counts as in the can).
The robot must be holding nothing: a rag that a hand still holds never counts as in the can, and full
success needs both hands empty.

  1.0   every rag (this task has one) in the can and the lid shut
  0.5   every rag in the can, lid not shut
  0     any rag in the bin, or anything else

(`score_from` also keeps the 0.25 rule for "some but not all rags in, lid shut", which cannot occur with a
single rag.)
"""
import numpy as np

import can as C
import layout as L

LID_SHUT_DEG = 3.0
REST_SPEED = 0.05          # m/s: a rag whose points move slower than this on average is at rest
INNER_R = C.BODY_R - C.WALL
IN_CAN_R = C.BODY_R + 0.003
RIM_Z = L.TOP + C.BODY_H
BIN_INNER_R = L.BIN_R - L.BIN_WALL
BIN_RIM_Z = L.TOP + L.BIN_H


def rag_in_can(nodes, can_xy=L.CAN_XY):
    n = np.asarray(nodes, float)
    r = np.hypot(n[:, 0] - can_xy[0], n[:, 1] - can_xy[1])
    return bool(np.all(r < IN_CAN_R) and np.all(n[:, 2] < RIM_Z) and np.all(n[:, 2] > L.TOP))


def rag_in_bin(nodes, bin_xy=L.BIN_XY):
    n = np.asarray(nodes, float)
    r = np.hypot(n[:, 0] - bin_xy[0], n[:, 1] - bin_xy[1])
    return bool(np.any((r < BIN_INNER_R) & (n[:, 2] < BIN_RIM_Z) & (n[:, 2] > L.TOP)))


def lid_shut(angle_deg):
    return float(angle_deg) <= LID_SHUT_DEG


def at_rest(velocities):
    """Mean point speed below 5 cm/s. (Single cloth nodes keep a little numerical jitter at rest, so the
    mean is used; a falling or carried rag moves far faster.)"""
    v = np.asarray(velocities, float)
    return bool(len(v) == 0 or np.mean(np.linalg.norm(v, axis=1)) < REST_SPEED)


def score_from(rags_nodes, lid_angle_deg, held=None):
    """rags_nodes: list of (N, 3) node arrays, one per rag; held: per rag, whether a hand holds it."""
    held = list(held) if held is not None else [False] * len(rags_nodes)
    if any(rag_in_bin(n) for n in rags_nodes):
        return 0.0
    inside = sum(rag_in_can(n) and not h for n, h in zip(rags_nodes, held))
    shut = lid_shut(lid_angle_deg)
    if inside == len(rags_nodes):
        return 1.0 if shut else 0.5
    if inside >= 1 and shut:
        return 0.25
    return 0.0


def success_from(rags_nodes, rags_vel, lid_angle_deg, held=None):
    """Full score with both hands empty and every rag at rest (nothing still falling or being carried)."""
    held = list(held) if held is not None else [False] * len(rags_nodes)
    return (score_from(rags_nodes, lid_angle_deg, held) == 1.0 and not any(held)
            and all(at_rest(v) for v in rags_vel))
