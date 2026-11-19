"""On the station table, at real size: a small pine wall cupboard whose door squeaks at its top hinge, and a can of
WD-40 with its Smart Straw flipped up.

The cabinet (fixed): 340 mm tall, 306 mm wide, 200 mm deep, 18 mm pine, standing on the counter against the wall, its
front face 46 cm in front of the arm bases. Its inset door (266 x 300 x 18 mm pine, a wooden knob on its left-hand
stile) hangs on two 2-1/2 in brass-plated butt hinges on its right-hand edge, at the robot's midline. Each hinge has
five knuckles on a 9 mm barrel; the top hinge is the squeaky one.

The can (loose): an 8 oz WD-40 Multi-Use Product can with the Smart Straw, 66.7 mm across and 171 mm tall, 0.31 kg,
standing on the table to the right of the cabinet, its red straw flipped up (the stream setting) and pointing away
from the robot, 186 mm out from the can's axis. Pressing the red actuator pad on its domed top at least 1.5 mm (about
10 N) opens the valve.

Goal: each of the four gaps between the top hinge's knuckles given a short burst through the straw; then the door
swung open and shut three times to work the oil in; then the can stood back on the table and the door left shut.
See scoring.py. The usual way: one hand holds the can with the straw's tip a centimetre from the gap; the other
presses the pad for a moment; step to the next gap; set the can down; swing the door by its knob.

The stream (a rule standing in for the spray, see ``_spray``): while the actuator is pressed past 1.5 mm, a straight
stream leaves the straw's tip along it; where it meets the top hinge's barrel decides which gap it lands in.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
         --scorer task_score ...
Demo:    python demokit.py task.py <out_dir>          (see demokit.py)
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the assemblies are built as articulations)
import rigstate  # noqa: E402
import scoring as S  # noqa: E402

rigstate.install()

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = L.SUCCESS_HOLD_S

INSTRUCTION = ("The cabinet door's top hinge squeaks. Give each gap between its knuckles a short burst of WD-40 "
               "through the straw, swing the door open and shut three times to work it in, then stand the can on "
               "the table and leave the door shut.")

MM = L.MM
DT = 1.0 / 20.0  # the control step (control_hz=20, as the README's commands run it)


# -- the cabinet --------------------------------------------------------------------------------------------------------
def _carcass():
    d, b, w, h = L.CAB_DEPTH, L.BOARD, L.OPEN_W, L.CAB_H
    out = [
        mk.box((d / 2, -b / 2, h / 2), (d, b, h), L.PINE),  # right-hand side (the hinge side)
        mk.box((d / 2, w + b / 2, h / 2), (d, b, h), L.PINE),  # left-hand side
        mk.box(((d - L.BACK_T) / 2, w / 2, b / 2), (d - L.BACK_T, w, b), L.PINE_DARK),  # bottom
        mk.box(((d - L.BACK_T) / 2, w / 2, h - b / 2), (d - L.BACK_T, w, b), L.PINE_DARK),  # top
        mk.box((d - L.BACK_T / 2, w / 2, h / 2), (L.BACK_T, w + 2 * b, h), L.HARDBOARD),  # back
        mk.box((d / 2, w / 2, h / 2), (0.006, w - 0.01, 0.006), L.PINE_DARK),  # a shelf cleat inside (it shows when open)
    ]
    for centre in (L.TOP_HINGE_Z, L.BOTTOM_HINGE_Z):  # the frame leaf's knuckles (1st, 3rd, 5th) and the pin's ends
        for lo, hi, on_door in L.knuckles(centre):
            if not on_door:
                out.append(mk.cylinder((L.AXIS_X, L.AXIS_Y, (lo + hi) / 2), L.BARREL_R, hi - lo, color=L.BRASS))
        top = centre + L.HINGE_LEN / 2
        out.append(mk.cylinder((L.AXIS_X, L.AXIS_Y, top + L.PIN_HEAD_H / 2), L.PIN_HEAD_R, L.PIN_HEAD_H, color=L.BRASS))
    return out


def _door():
    out = [mk.box((L.DOOR_T / 2, (L.DOOR_Y0 + L.DOOR_Y1) / 2, (L.DOOR_Z0 + L.DOOR_Z1) / 2),
                  (L.DOOR_T, L.DOOR_W, L.DOOR_H), L.PINE),
           mk.cylinder((-L.NECK_L / 2, L.KNOB_Y, L.KNOB_Z), L.NECK_D / 2, L.NECK_L, axis="X", color=L.KNOB_WOOD),
           mk.cylinder((-L.NECK_L - L.KNOB_L / 2, L.KNOB_Y, L.KNOB_Z), L.KNOB_D / 2, L.KNOB_L, axis="X",
                       color=L.KNOB_WOOD)]
    for centre in (L.TOP_HINGE_Z, L.BOTTOM_HINGE_Z):  # the door leaf's knuckles (2nd, 4th)
        for lo, hi, on_door in L.knuckles(centre):
            if on_door:
                out.append(mk.cylinder((L.AXIS_X, L.AXIS_Y, (lo + hi) / 2), L.BARREL_R, hi - lo, color=L.BRASS))
    return out


DOOR = mk.hinge("door", "Z", (L.AXIS_X, L.AXIS_Y, 0.0), _door(), upper=L.DOOR_OPEN_MAX, mass=L.DOOR_MASS,
                damping=L.DOOR_DAMPING, friction=L.DOOR_FRICTION)
CABINET = mk.assembly("cabinet", (L.CAB_X, L.CAB_Y, L.TOP), _carcass(), joints=(DOOR,), color=L.PINE,
                      friction=(0.6, 0.5))


# -- the can --------------------------------------------------------------------------------------------------------------
def _can_body():
    h = L.CAN_BODY_H
    out = [mk.cylinder((0.0, 0.0, 0.27 * h), L.CAN_R, 0.54 * h, color=L.CAN_BLUE),
           mk.cylinder((0.0, 0.0, 0.745 * h), L.CAN_R, 0.41 * h, color=L.CAN_YELLOW),  # the yellow label panel
           mk.cylinder((0.0, 0.0, 0.975 * h), L.CAN_R, 0.05 * h, color=L.CAN_BLUE),
           mk.cylinder((0.0, 0.0, h - L.SEAM_H / 2), L.SEAM_R, L.SEAM_H, color=L.CAN_SILVER)]  # the top seam
    for r, z0, z1 in L.DOME:  # the silver dome, in two steps
        out.append(mk.cylinder((0.0, 0.0, (z0 + z1) / 2), r, z1 - z0, color=L.CAN_SILVER))
    return out


def _valve():
    """The Smart Straw actuator, which rides down on the valve: the red pad (a flat-topped middle with rounded ends),
    the grey nozzle arm under its front, and the red straw, flipped up level."""
    x0, x1 = L.PAD_X
    z0, top = L.PAD_Z0, L.ACT_TOP
    (nx0, nz0), (nx1, nz1) = L.NOZZLE_FROM, L.NOZZLE_TO
    n_len = math.hypot(nx1 - nx0, nz1 - nz0)
    return [mk.box(((x0 + x1) / 2, 0.0, (z0 + top) / 2), (x1 - x0, L.PAD_W, top - z0), color=L.PAD_RED),
            mk.cylinder((x0, 0.0, top - L.PAD_BACK_R), L.PAD_BACK_R, L.PAD_W, axis="Y", color=L.PAD_RED),
            mk.cylinder((x1, 0.0, top - L.PAD_FRONT_R), L.PAD_FRONT_R, L.PAD_W, axis="Y", color=L.PAD_RED),
            mk.cylinder(((nx0 + nx1) / 2, 0.0, (nz0 + nz1) / 2), L.NOZZLE_R, n_len, axis="X", color=L.NOZZLE_GREY,
                        ypr=(0.0, -math.atan2(nz1 - nz0, nx1 - nx0), 0.0)),  # rising toward its end
            mk.cylinder((L.STRAW_X0 + L.STRAW_L / 2, 0.0, L.NOZZLE_Z), L.STRAW_R, L.STRAW_L, axis="X",
                        color=L.STRAW_RED)]


VALVE = mk.slider("valve", "-Z", _valve(), upper=L.VALVE_TRAVEL, mass=0.02, spring=L.VALVE_K,
                  damping=L.VALVE_DAMPING)
CAN = mk.assembly("can", (L.CAN_START[0], L.CAN_START[1], L.TOP + 0.5 * MM), _can_body(), joints=(VALVE,),
                  mass=L.CAN_MASS, yaw=L.CAN_START_YAW, color=L.CAN_BLUE, friction=L.CAN_FRICTION,
                    tight_fit=True)

OBJECTS = [CABINET, CAN]


# -- the stream, every control step ----------------------------------------------------------------------------------------
SPRAY = S.Spray()
_EP = {"flood_said": False, "swings_said": 0}
_TRACE = os.environ.get("TASK_TRACE")


def _new_episode() -> None:
    SPRAY.reset()
    _EP.update(flood_said=False, swings_said=0)


mk.on_reset(_new_episode)
rigstate.on_reset(_new_episode)


@mk.on_step
def _spray(states) -> None:
    can, cab = states["can"], states["cabinet"]
    valve = float(can.joint("valve"))
    door = float(cab.joint("door"))
    was = SPRAY.oiled
    landed = SPRAY.step(valve, can.rotation(), can.origin, door, DT)
    for k, (a, b) in enumerate(zip(was, SPRAY.oiled)):
        if b and not a:
            print(f"[oil-squeaky-hinge] step {SPRAY.step_n}: {L.SEAM_NAMES[k]} oiled", flush=True)
    if SPRAY.flooded and not _EP["flood_said"]:
        _EP["flood_said"] = True
        print(f"[oil-squeaky-hinge] step {SPRAY.step_n}: sprayed over {L.SPRAY_LIMIT_G} g in all: flooded", flush=True)
    if SPRAY.swings > _EP["swings_said"]:
        _EP["swings_said"] = SPRAY.swings
        print(f"[oil-squeaky-hinge] step {SPRAY.step_n}: door swing {SPRAY.swings}", flush=True)
    if _TRACE:
        tip, d = S.straw_jet(can.rotation(), can.origin, valve)
        with open(_TRACE, "a") as f:
            f.write(json.dumps({"step": SPRAY.step_n, "valve_mm": round(valve / MM, 3),
                                "door_deg": round(math.degrees(door), 2),
                                "tip": [round(float(v), 4) for v in tip], "dir": [round(float(v), 3) for v in d],
                                "landed": landed, "dose": [round(v, 3) for v in SPRAY.dose],
                                "spray_s": round(SPRAY.spray_time, 3), "swings": SPRAY.swings}) + "\n")


# -- scoring ----------------------------------------------------------------------------------------------------------
def _hands_distance(can, cab):
    def distance(points):
        return np.minimum(can.distance(points), cab.distance(points, bodies=("door",)))
    return distance


def reading(objects) -> S.Reading:
    can, cab = objects["can"], objects["cabinet"]
    arms = rigstate.arms()
    released = rigstate.released(arms, _hands_distance(can, cab), clearance=L.CLEAR)
    door_v = abs(float(cab.joint_vel.get("door", 0.0))) if cab.joint_vel else 0.0
    still = float(np.linalg.norm(can.lin_vel)) < L.STILL_V and door_v < 0.05
    return S.judge(SPRAY, float(cab.joint("door")), can.rotation(), can.origin, released, still)


def score(objects) -> float:
    """See scoring.py: 1.0 all four gaps oiled, the door worked three times, the can stood on the table and the door
    shut, let go; 0.75 oiled and swung at least once; 0.5 all four gaps oiled; 0.25 two; 0 otherwise or over 4 g sprayed."""
    return reading(objects).score


def success(objects) -> bool:
    return score(objects) >= 1.0


# -- demo -------------------------------------------------------------------------------------------------------------
def demo_script(ctx):
    """The scripted demo (demo.py), run by demokit.py. TASK_DEMO=full (default): the whole task; spray: the four gaps
    only; flood: hold the valve open 5 s at the hinge (a cheat check: scores 0); swing: work the door without
    oiling (scores 0); idle: the arms stay at their ready pose (the setup screenshot)."""
    if ctx.mode in ("full", "spray", "flood", "swing", "aim"):
        import demo

        yield from demo.demo_script(ctx)
        return
    for _ in range(40):
        yield
