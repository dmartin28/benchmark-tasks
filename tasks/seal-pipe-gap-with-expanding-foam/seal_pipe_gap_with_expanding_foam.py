"""On the station table, 42 cm in front of the arms: a wall section standing across the table and facing the robot,
as wide as the table (1.30 m) and as tall: a 38 mm timber rim board with drywall behind it. A 22 mm copper pipe runs
straight through a 50 mm hole in it, 22 cm above the table, leaving a 14 mm gap all round the pipe. A dark mat lies in
front of the wall. In front of the left arm, a can of expanding gap-filler foam stands upright on the table: a 6.5 cm
blue can, a small round actuator on its valve, a long white straw rising from the actuator's top at 45 deg, and a
narrow orange trigger under the straw, pushed toward the can to dispense. The can only dispenses upside down.

Run it:  inspect-robots run --embodiment yam_arena -E task=seal_pipe_gap_with_expanding_foam.py -E object_jitter=0 \\
         -E control_hz=20 --scorer task_score ...
Demo:    python demo.py seal_pipe_gap_with_expanding_foam.py <out_dir>

The foam (foam.py): liquid leaves the straw at a rate set by the lever's press, only with the can upside down; what
lands inside the gap doubles as it rises and spreads along the gap, a share of any excess bulging out of the face;
what lands outside the gap is mess (not scored). Scored once the foam has cured: the share of the ring sealed, less for
bulge (see score()).
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

import numpy as np  # noqa: E402

import foam  # noqa: E402
import geometry as g  # noqa: E402
from foamviz import FoamViz  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = g.SUCCESS_HOLD_S

INSTRUCTION = ("Seal the gap around the pipe where it passes through the wall with the spray foam. The can only "
               "dispenses upside down. Fill the gap only about halfway so the foam can rise.")

TOP = g.TOP

# -- objects --------------------------------------------------------------------------------------------------------
CAN = UsdBody("foam_can", usd="assets/can.usda", pos=tuple(float(v) for v in g.can_start_pos()),
              rot_xyzw=g.can_rot_xyzw(), size=(2 * g.CAN_R, 2 * g.CAN_R, float(g.TIP[2])), mass=g.CAN_MASS,
              articulation=True, joint_pos=(("lever", 0.0),), drives=(("lever", g.LEVER_K, g.LEVER_DAMPING),),
              friction=(1.2, 1.0), solver_iters=32)
WALL = UsdBody("wall", usd="assets/wall.usda", pos=tuple(float(v) for v in g.HOLE_CENTRE),
               size=(g.RIM_T + g.BACKER_T, g.WALL_W, g.WALL_H), friction=(0.5, 0.4))
MAT = Box("mat", pos=g.MAT_CENTER, size=g.MAT_SIZE, color=(0.16, 0.16, 0.17))
OBJECTS = [WALL, MAT, CAN]

# -- monitor: the lever, the can's attitude and the flow they would give ---------------------------------------------
_STATE: dict = {}
LEDGER = foam.Ledger()
_VIZ = FoamViz()
DT = 1.0 / 20.0                                                   # the control period (-E control_hz=20)


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(steps=0, lever_deg=0.0, press=0.0, valve_from_down_deg=180.0, flow_cc_s=0.0, liquid_cc=0.0,
                  base_z0=None, lift=0.0, max_lift=0.0, tilt_deg=0.0, lowest=0.0, sector=None)
    LEDGER.reset()
    _VIZ.hide()


_reset_state()


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    can = object_states(environment, ["foam_can"])["foam_can"]
    R = can.rotation()
    lever = float(can.joints.get("lever", 0.0))
    press = foam.press_fraction(lever)
    angle = foam.valve_from_down(R)
    q = foam.flow_rate(press, angle)
    if _STATE["base_z0"] is None:
        _STATE["base_z0"] = float(can.pos[2])
    _STATE["steps"] += 1
    _STATE["lever_deg"] = math.degrees(lever)
    _STATE["press"] = press
    _STATE["valve_from_down_deg"] = math.degrees(angle)
    _STATE["flow_cc_s"] = q * 1e6
    _STATE["liquid_cc"] += q * 1e6 * DT
    _STATE["sector"] = LEDGER.step(can.pos + R @ g.TIP, q, DT)
    if _STATE["steps"] % 4 == 0:
        _VIZ.draw(environment, LEDGER)
    _STATE["lift"] = float(can.pos[2]) - _STATE["base_z0"]
    _STATE["max_lift"] = max(_STATE["max_lift"], _STATE["lift"])
    _STATE["tilt_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    ends = [float((can.pos + R @ p)[2]) for p in ((0.0, 0.0, 0.0), (0.0, 0.0, float(g.ADAPTER_C[2] + g.ADAPTER[2] / 2)))]
    _STATE["lowest"] = min(ends) - TOP


def success(objects) -> bool:
    """Full score, and the foam has visibly finished rising."""
    return score(objects) >= 1.0 and LEDGER.rise_shown() >= g.RISE_SHOWN


def score(objects) -> float:
    """Once the foam has cured, how many of the ring's 36 sectors (10 deg each) are sealed (cured foam filling at
    least 90% of the sector, the stand-in for foam spanning the gap at the face): 0.25 from 9, 0.5 from 18, 0.75 from
    27, 1.0 from 35. Less 0.25 for foam standing more than 10 mm proud of the wall face; capped at 0.5 past 20 mm.
    Never below 0. Foam outside the gap earns nothing and costs nothing."""
    return foam.score(LEDGER)
