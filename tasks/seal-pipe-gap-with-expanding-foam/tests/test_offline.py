"""Offline checks (no simulator, no GPU):  python -m pytest -q tests/   (in the Isaac Lab-Arena environment)

The task file loads; the flow law's dead zone and upside-down gate; the foam's spreading and the scoring; and a
kinematic dry run of the demo (pick, half turn, handover, upside down, three points round the ring, the hands
swapping roles, the bottom point) through
yam_arena's IK and URDF collision hulls: every pose reached, every joint well inside its limits, the valve ending
straight down, the straw inside the gap, and nothing colliding that shouldn't.
"""

import math
import pathlib
import sys
from functools import lru_cache

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from yam_arena import analysis, ik, task_file  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402

import demo  # noqa: E402
import foam  # noqa: E402
import geometry as g  # noqa: E402
from demokit import Arm  # noqa: E402

TASK_PATH = HERE / "seal_pipe_gap_with_expanding_foam.py"
JAW_PER_GRIP = 0.0939   # m between the fingertip pads per unit of grip command (URDF: 2 x 46.95 mm)
FINGERS = ("tip_left", "tip_right")


def test_the_task_file_loads_with_its_objects():
    task = task_file.load(TASK_PATH)
    assert task.scene == "station"
    assert set(task.names) == {"wall", "mat", "foam_can"}
    assert task.module.INSTRUCTION.startswith("Seal the gap")


@pytest.mark.parametrize("press, angle, expected", [
    (0.0, 0.0, 0.0),                                  # lever at rest
    (g.DEAD_ZONE, 0.0, 0.0),                          # end of the dead zone: still nothing
    (1.0, 0.0, g.Q_MAX),                              # full press, straight down
    (1.0, math.radians(60), g.Q_MAX),                 # still full flow at 60 deg from down
    (1.0, math.radians(75), g.Q_MAX / 2),             # half way through the fade
    (1.0, math.radians(90), 0.0),                     # level: no foam
    (1.0, math.radians(180), 0.0),                    # upright: no foam
])
def test_the_flow_law(press, angle, expected):
    assert foam.flow_rate(press, angle) == pytest.approx(expected)


def test_flow_rises_with_press_past_the_dead_zone():
    flows = [foam.flow_rate(p, 0.0) for p in np.linspace(g.DEAD_ZONE, 1.0, 9)]
    assert all(b > a for a, b in zip(flows, flows[1:]))


# -- the foam: spreading along the gap, and the score ----------------------------------------------------------------
def _ledger(doses, mess_cc=0.0):
    """A ledger with the given liquid doses (sector, cm3), all laid down at t=0, plus some mess."""
    led = foam.Ledger()
    for s_, cc in doses:
        led.deposits.append((s_, cc * 1e-6, 0.0))
    if mess_cc:
        led.mess.append((np.zeros(3), mess_cc * 1e-6, 0.0))
    led.t = 1000.0
    return led


DEMO_DOSES = [(27, 9.5), (0, 9.5), (9, 9.5), (18, 9.5)]          # bottom, left, top, right


def test_spreading_conserves_the_foam():
    rng = np.random.default_rng(1)
    landed = rng.uniform(0.0, 4.0, g.SECTORS) * foam.SECTOR_VOLUME
    held, bulge = foam.spread(landed)
    assert held.sum() + bulge.sum() == pytest.approx(landed.sum())
    assert held.max() <= foam.SECTOR_VOLUME * (1 + 1e-9) and bulge.min() >= 0.0


def test_one_dose_fills_only_its_part_of_the_ring():
    fill = _ledger([(0, 9.5)]).fill(cured=True)
    filled = np.nonzero(fill >= g.SECTOR_FULL)[0]
    assert 7 <= len(filled) <= 12
    assert fill[18] == 0.0                                       # the far side stays empty


def test_four_doses_seal_the_ring():
    led = _ledger(DEMO_DOSES)
    assert led.coverage() == 1.0
    assert led.bulge_heights().max() < g.BULGE_OK
    assert foam.score(led) == 1.0


def test_too_little_foam_leaves_gaps():
    led = _ledger([(s_, 6.0) for s_, _ in DEMO_DOSES])
    assert led.coverage() < 0.95 and foam.score(led) < 1.0


def test_one_big_dose_bulges_instead_of_filling_the_ring():
    led = _ledger([(0, 38.0)])
    assert led.coverage() < 0.75
    assert led.bulge_heights().max() > g.BULGE_MAX
    assert foam.score(led) <= 0.5


def test_doubling_the_doses_overfills():
    led = _ledger([(s_, 2 * cc) for s_, cc in DEMO_DOSES])
    assert led.coverage() == 1.0 and foam.score(led) <= 0.5


def test_foam_outside_the_gap_earns_nothing():
    led = foam.Ledger()
    led.step(np.array([g.FACE_X - 0.01, 0.0, g.HOLE_Z + 0.02]), g.Q_MAX, 1.0)   # in front of the wall face
    led.step(np.array([g.FACE_X + 0.01, 0.0, g.HOLE_Z + 0.018]), g.Q_MAX, 0.1)  # in the gap
    assert len(led.mess) == 1 and len(led.deposits) == 1
    assert foam.score(_ledger([], mess_cc=38.0)) == 0.0                   # all of it outside: nothing sealed
    assert foam.score(_ledger(DEMO_DOSES, mess_cc=6.0)) == 1.0            # judged at the gap's face only


def test_nothing_scores_nothing():
    assert foam.score(foam.Ledger()) == 0.0


# -- the kinematic dry run ------------------------------------------------------------------------------------------
class _Ctx:
    def __init__(self):
        self.left, self.right = Arm("left"), Arm("right")
        self.flags = {}


def _yaw(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@lru_cache(maxsize=1)
def _dry_run():
    """Every control step: both arms' commands, and the can's pose (base centre, rotation without its starting yaw)."""
    ctx = _Ctx()
    can_o, can_r = g.can_start_pos().copy(), np.eye(3)
    holder, rel = None, None
    frames = []
    for _ in demo.script(ctx):
        h = ctx.flags.get("holder")
        if h != holder:                                       # a hand has just taken the can: lock the grip
            holder = h
            arm = getattr(ctx, h)
            rel = (can_r.T @ arm.rot, can_r.T @ (arm.pos - can_o))
        if holder is not None:
            arm = getattr(ctx, holder)
            can_r = arm.rot @ rel[0].T
            can_o = arm.pos - can_r @ rel[1]
        frames.append(dict(q={s: getattr(ctx, s).q.copy() for s in ("left", "right")},
                           grip={s: getattr(ctx, s).grip for s in ("left", "right")},
                           holder=holder, can_o=can_o.copy(), can_r=can_r.copy(),
                           at_ring=bool(ctx.flags.get("at_ring", False)), phase=ctx.flags.get("phase")))
    return ctx, frames


def test_the_demo_is_reachable_well_inside_the_joint_limits():
    ctx, frames = _dry_run()
    assert ctx.left.worst_mm < 0.5 and ctx.right.worst_mm < 0.5
    for side in ("left", "right"):
        Q = np.array([f["q"][side] for f in frames])
        margin = math.degrees(min((Q - ik.LOWER).min(), (ik.UPPER - Q).min()))
        assert margin > 5.0, f"{side} arm comes within {margin:.1f} deg of a joint limit"


def _valve_from_down(f):
    valve = f["can_r"] @ g.VALVE_DIR
    return math.degrees(math.acos(np.clip(-valve[2], -1.0, 1.0)))


def test_the_turn_over_ends_upside_down_and_the_ring_work_keeps_foam_flowing():
    _, frames = _dry_run()
    turned = [f for f in frames if f["phase"] == "turn2"][-1]
    assert _valve_from_down(turned) < 5.0
    for f in frames:
        if f["at_ring"]:                                   # within the full-flow gate at every set point
            assert _valve_from_down(f) < math.degrees(g.GATE_FULL)
    end = frames[-1]
    lowest = min((end["can_o"] + end["can_r"] @ p)[2] for p in ((0, 0, 0), g.ADAPTER_C + [0, 0, g.ADAPTER[2] / 2]))
    assert lowest - g.TOP > 0.10


def test_the_right_hand_takes_the_can_away_from_the_left_hand():
    gap = g.PICK_HOLD - g.TAKE_HOLD
    pad_half = 0.027                                          # the pads reach this far either side along the can
    assert gap - 2 * pad_half > 0.03


def test_the_open_hand_clears_the_can():
    assert JAW_PER_GRIP * demo.GRIP_OPEN > 2 * g.CAN_R + 0.02


def _hulls(f, side):
    q = {f"joint{i + 1}": float(v) for i, v in enumerate(f["q"][side])}
    d = f["grip"][side] * FINGER_OPEN
    q.update(joint7=d, joint8=d)
    from yam_arena import station
    return analysis.world_hulls(analysis.fk_link_poses_world(q, station.BASES[side]))


def _cylinder_sd(points, origin, rot, radius, z0, z1):
    """Signed distance of points to a cylinder along the frame's z from z0 to z1 (negative inside)."""
    local = (points - origin) @ rot
    rad = np.hypot(local[:, 0], local[:, 1]) - radius
    ax = np.maximum(z0 - local[:, 2], local[:, 2] - z1)
    outside = np.hypot(np.maximum(rad, 0), np.maximum(ax, 0))
    return outside + np.minimum(np.maximum(rad, ax), 0)


def _segment_sd(points, a, b, radius):
    ab = b - a
    t = np.clip((points - a) @ ab / (ab @ ab), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1) - radius


GRIP_PHASES = ("pick", "handover", "swap")   # where a hand closes on the can's body or lets go of it


def _clearances(yaw):
    """Smallest clearances over the dry run: arm to arm, arms to the can body and adapter (fingers excluded), every
    link to the straw, and the left fingers to the table at the pickup. Fingers may touch the can's body only when
    they hold it: the holding hand's, and at the pickup and the handover a hand that is closing on it or letting go
    (its grip command below fully open). There an open hand's fingers are reported apart ("open_fingers"): at the
    grip pose the convex finger hulls overlap the can by ~2 mm even fully open, an artefact of the hulls."""
    _, frames = _dry_run()
    out = dict(arms=np.inf, body=np.inf, straw=np.inf, table=np.inf, open_fingers=np.inf)
    for f in frames[::8]:
        hl, hr = _hulls(f, "left"), _hulls(f, "right")
        out["arms"] = min(out["arms"], analysis.hull_set_distance(hl, hr)[0])
        R = f["can_r"] @ _yaw(yaw)
        a = f["can_o"] + R @ g.STRAW_BASE
        b = f["can_o"] + R @ g.TIP
        for side, hs in (("left", hl), ("right", hr)):
          for name, h in hs:
            v = h.vertices
            out["straw"] = min(out["straw"], _segment_sd(v, a, b, g.STRAW_R).min())
            gripping = f["phase"] in GRIP_PHASES and f["grip"][side] < demo.GRIP_OPEN
            if name in FINGERS and f["phase"] in GRIP_PHASES and not gripping and side != f["holder"]:
                out["open_fingers"] = min(out["open_fingers"],
                                          _cylinder_sd(v, f["can_o"], R, g.CAN_R, 0.0, g.CAN_H).min())
            elif name not in FINGERS or side != f["holder"] and not gripping:
                body = _cylinder_sd(v, f["can_o"], R, g.CAN_R, 0.0, g.CAN_H).min()
                if name not in FINGERS:                          # the presser's fingers do touch the adapter
                    body = min(body, analysis.box_signed_distance(v, f["can_o"] + R @ g.ADAPTER_C, R,
                                                                  np.array(g.ADAPTER)).min())
                out["body"] = min(out["body"], body)
        if f["holder"] is None or f["holder"] == "left" and f["can_o"][2] < g.TOP + 0.06:
            for name, h in hl:
                if name in FINGERS:
                    out["table"] = min(out["table"], float(h.vertices[:, 2].min()) - g.TOP)
    return out


def test_the_arms_and_the_can_stay_clear_of_the_wall():
    _, frames = _dry_run()
    centre = np.array([g.FACE_X + (g.RIM_T + g.BACKER_T) / 2, 0.0, g.TOP + g.WALL_H / 2])
    size = np.array([g.RIM_T + g.BACKER_T, g.WALL_W, g.WALL_H])
    yaw = _yaw(g.CAN_YAW)
    arms = can = np.inf
    for f in frames[::4]:
        R = f["can_r"] @ yaw
        body = [g.ADAPTER_C, np.zeros(3), np.array([g.CAN_R, 0, 0.1]), np.array([-g.CAN_R, 0, 0.1]), np.array([0, 0, g.CAN_H])]
        pts = np.array([f["can_o"] + R @ p for p in body + ([] if f["at_ring"] else [g.TIP])])   # at the ring the
        # straw is meant to be in the hole: test_the_straw_stays_inside_the_gap covers it there
        can = min(can, analysis.box_signed_distance(pts, centre, np.eye(3), size).min())
        for side in ("left", "right"):
            for _, h in _hulls(f, side):
                arms = min(arms, analysis.box_signed_distance(h.vertices, centre, np.eye(3), size).min())
    assert arms > 0.015, f"an arm comes within {arms * 100:.1f} cm of the wall"   # (closest: 1.7 cm, the left on its
    # way to the first point's approach)
    assert can > 0.02, f"the can or its straw comes within {can * 100:.1f} cm of the wall"


def test_the_straw_stays_inside_the_gap():
    """At the ring, every bit of straw past the wall face is inside the gap with room to spare, and no bit of it
    reaches into the pipe stub in front of the wall."""
    _, frames = _dry_run()
    ring = [f for f in frames if f["at_ring"]]
    assert ring
    worst_out = worst_in = worst_pipe = np.inf
    for f in ring:
        R = f["can_r"] @ _yaw(g.CAN_YAW)
        a, b = f["can_o"] + R @ g.STRAW_BASE, f["can_o"] + R @ g.TIP
        for u in np.linspace(0.0, 1.0, 60):
            p = a + u * (b - a)
            r = math.hypot(p[1] - g.HOLE_CENTRE[1], p[2] - g.HOLE_CENTRE[2])
            if p[0] >= g.FACE_X:
                worst_out = min(worst_out, g.RING_OUT - g.STRAW_R - r)     # clear of the hole's edge
                worst_in = min(worst_in, r - g.RING_IN - g.STRAW_R)        # clear of the pipe
            elif p[0] >= g.FACE_X - g.PIPE_STUB:
                worst_pipe = min(worst_pipe, r - g.PIPE_R - g.STRAW_R)     # clear of the pipe stub
    assert worst_out > 0.0005, f"the straw comes {-worst_out * 1000:.1f} mm into the hole's edge"
    assert worst_in > 0.0005, f"the straw comes {-worst_in * 1000:.1f} mm into the pipe"
    assert worst_pipe > 0.0015, f"the straw passes within {worst_pipe * 1000:.1f} mm of the pipe stub"   # (closest:
    # 1.8 mm at the right point, aimed 12 deg up the ring)


def test_the_set_points_lie_in_their_sectors():
    """Left, top, right and bottom (sectors 0, 9, 18, 27), each aimed RING_ANGLE_TRIM along the ring from its sector
    (against the drift a press gives the held can): the straw's tip lands in the gap, in the sector it is aimed at."""
    points = g.RING_ORDER + (g.SWAP_POINT,)
    assert sorted(g.ring_point_sector(k) for k in points) == [0, 9, 18, 27]
    for k in points:
        o = demo.ring_can_origin(k, inside=True)
        tip = o + demo.ring_attitude(k)[0] @ g.TIP
        cell = foam.ring_cell(tip)
        aim = (g.ring_point_angle(k) + math.radians(g.RING_ANGLE_TRIM.get(k, 0.0))) % (2 * math.pi)
        assert cell is not None and cell[0] == int(aim / (2 * math.pi / g.SECTORS)) % g.SECTORS, (k, cell)


def test_nothing_collides_that_shouldnt():
    c = _clearances(g.CAN_YAW)
    assert c["arms"] > 0.01, f"the arms come within {c['arms'] * 100:.1f} cm of each other"
    assert c["body"] > 0.0, f"a palm or arm link reaches {-c['body'] * 1000:.1f} mm into the can"
    assert c["open_fingers"] > -0.003, f"open fingers reach {-c['open_fingers'] * 1000:.1f} mm into the can"
    assert c["straw"] > 0.005, f"an arm comes within {c['straw'] * 100:.1f} cm of the straw"   # (closest: 6 mm, the
    # right's fingers sliding out from between the trigger and the straw after the bottom dose)
    assert c["table"] > 0.005, f"the left fingers come within {c['table'] * 1000:.1f} mm of the table"
