"""Every dimension and mechanism number of the task, in one place. Pure Python.

Units: metres, kilograms, seconds, radians unless a name says otherwise. World frame: the station's (x forward from
the arms, y to the robot's left, z up), with the table top at ``TOP``.

Can frame (``can.usda``): origin at the centre of the can's base, z up the can's axis toward the valve, the straw in
the x-z plane pointing +x and ``STRAW_UP`` above the can's horizontal. The trigger is a narrow lever under the straw,
hinged on the actuator's side, level at rest and pressed down toward the can to dispense.

The can is modelled on a 12 oz straw-applicator gap-filler can like GREAT STUFF Gaps & Cracks (not an exact replica;
the trigger force is an estimate).
"""

from __future__ import annotations

import math

import numpy as np
from yam_arena.task_file import TABLE_TOP

TOP = TABLE_TOP["station"]

# -- the wall: rim board, backer and copper pipe (one fixed body; asset origin at the hole's centre on the face) ------
FACE_X = 0.42                     # the rim board's front face, facing the robot
TABLE_WIDTH = 1.30                # the station table's left-right size (yam_arena.station.TABLE_SIZE[1])
WALL_W = WALL_H = TABLE_WIDTH     # a wall section as wide as the table and as tall as it is wide, centred on y = 0
RIM_T = 0.038                     # 1.5 in timber
BACKER_T = 0.012                  # drywall behind it closes the back of the ring
HOLE_R = 0.025                    # 2 in hole
HOLE_Z = TOP + 0.22               # high enough that the trigger can still be squeezed with the straw in the bottom of
                                  # the gap (the adapter rides above the straw); at 0.15 the bottom points are out of reach
HOLE_CENTRE = np.array([FACE_X, 0.0, HOLE_Z])
PIPE_R = 0.0111                   # 3/4 in copper, 22.2 mm outside
PIPE_STUB = 0.09                  # how far the pipe sticks out toward the robot
PIPE_BEHIND = 0.06                # and past the backer
BORE_STAVES = 24                  # box colliders forming the round bore
STAVE_T = 0.004

# -- the ring the foam fills -------------------------------------------------------------------------------------
RING_IN, RING_OUT, RING_DEPTH = PIPE_R, HOLE_R, RIM_T
SECTORS, SLICES = 36, 4
RING_VOLUME = math.pi * (RING_OUT**2 - RING_IN**2) * RING_DEPTH        # about 60 cm3

# -- the mat in front of the wall --------------------------------------------------------------------------------
MAT_SIZE = (0.12, 0.30, 0.003)   # just in front of the wall, clear of the can
MAT_CENTER = (FACE_X - MAT_SIZE[0] / 2, 0.0, TOP + MAT_SIZE[2] / 2)

# -- the can ----------------------------------------------------------------------------------------------------------
CAN_R, CAN_H = 0.0325, 0.19       # the body; the valve cup sits on top of it
CUP_R, CUP_H = 0.0125, 0.010      # the valve cup, up to the actuator
CAN_MASS = 0.45                   # a full 340 g can
CAN_COM_Z = 0.075                 # the liquid sits low
# the actuator: a small plastic cylinder on the valve cup, the straw out of its side near the top
ACT_R, ACT_H = 0.0065, 0.020
ADAPTER = (2 * ACT_R, 2 * ACT_R, ACT_H)   # the actuator's bounding box (clearance checks)
ADAPTER_Z0 = CAN_H + CUP_H        # its bottom
ADAPTER_C = np.array([0.0, 0.0, ADAPTER_Z0 + ACT_H / 2])
STRAW_UP = math.radians(45)       # the straw rises this far above the can's horizontal
STRAW_R = 0.003
# the straw leaves the actuator's side at its top (the straw's top flush with the actuator's), well clear of the
# trigger below it
STRAW_BASE = np.array([ACT_R, 0.0, ADAPTER_Z0 + ACT_H - STRAW_R])
STRAW_LEN = 0.18 + (0.015 - ACT_R) / math.cos(STRAW_UP)
STRAW_DIR = np.array([math.cos(STRAW_UP), 0.0, math.sin(STRAW_UP)])
TIP = STRAW_BASE + STRAW_LEN * STRAW_DIR                               # the straw's tip, where the foam leaves
VALVE_DIR = np.array([0.0, 0.0, 1.0])                                  # "upside down" means this points down

# the trigger: a narrow lever under the straw, hinged on the actuator's side (LEVER_HINGE_UP above its bottom), pointing
# LEVER_REST_DOWN below the can's horizontal at rest (so 55 deg below the straw); pressed further down toward the can by
# LEVER_TRAVEL to dispense (a plunger in the actuator pushes the valve stem down), a spring returns it; fully pressed it
# clears the can's top and the valve cup by ~6 mm. Joint ``lever``: 0 at rest, LEVER_TRAVEL fully pressed.
LEVER_LEN, LEVER_W, LEVER_T = 0.0415, 0.008, 0.003
LEVER_HINGE_UP = 0.011
LEVER_HINGE = np.array([ACT_R, 0.0, ADAPTER_Z0 + LEVER_HINGE_UP])
LEVER_REST_DOWN = math.radians(10)
LEVER_TRAVEL = math.radians(15)
LEVER_CONTACT = 0.035             # where a finger presses, out along the lever from its hinge
LEVER_FORCE = 4.5                 # N at the contact point, fully pressed (light: the holding hand takes the push)
LEVER_K = LEVER_FORCE * LEVER_CONTACT / LEVER_TRAVEL                   # N m / rad, the return spring
LEVER_DAMPING = 0.004
LEVER_MASS = 0.004

# -- where the can starts: standing upright on the table, where the left arm picks it up most easily (offline IK) ----
CAN_XY = (0.24, 0.07)             # the can's front is ~15 cm from the wall face
CAN_YAW = math.radians(-90)       # the straw's direction as the can stands; set by the clearance check

# -- the demo's turn-over: pick, turn half way, hand over, turn the rest (offline IK search, joints > 20 deg inside
# their limits throughout). Angles are about a level axis; "can frame" here ignores the can's starting yaw, since
# both grips are symmetric about the can's axis. -------------------------------------------------------------------------
GRIP_PATCH = 0.010                # m: the torsional contact patch on the can body (the pads are ~2 cm wide)
PICK_HOLD = 0.125                 # the left hand grips this far up the can (close to its centre of mass)
PICK_TILT = math.radians(36)      # the hand angled down onto the can; fingers level across it
PICK_AZ_OFF = math.radians(-15)   # approach this far off the left arm's reach line to the can
PICK_LIFT = 0.12                  # straight up off the table
TURN_AXIS_AZ = math.radians(-74.9)  # the level axis the can turns about (roughly left-right)
TURN_FIRST = math.radians(96.7)   # the left hand's share of the turn
TURN_TOTAL = math.radians(180)    # upside down: the valve straight down
HANDOVER = (0.186, -0.004, 0.314) # the left hand's grip point at the handover (z above the table); well back from
                                  # the wall so the straw clears it through the whole turn
# The right hand takes the can with a grip that also works at every ring point (offline search: joints >= 19 deg
# inside their limits at all 8 points). In the can's own frame (straw along +x, lever on the +y face):
TAKE_HOLD = 0.040                 # 4 cm up the can, 8.5 cm below the left hand
TAKE_FINGERS = math.radians(159.8)  # its finger axis across the can, from the can's x toward its y
TAKE_TILT = math.radians(-22.1)   # its approach tipped along the can's axis (toward the base)

# -- the trigger and the ring (offline search) --------------------------------------------------------------------
RING_POINTS = 8                   # set points round the gap, every 45 deg from the robot's left (+y), counter-clockwise
def ring_point_sector(k: int) -> int:
    """The sector of set point k: every 45 deg, rounded to a sector so the point sits at a sector's centre."""
    return int(math.floor(k * SECTORS / RING_POINTS))


def ring_point_angle(k: int) -> float:
    """Set point k's angle round the hole (rad, 0 at the robot's left, counter-clockwise seen from the robot)."""
    return (ring_point_sector(k) + 0.5) * 2 * math.pi / SECTORS


# The can's attitude at each set point (offline search, playground press_attitude_exact.py / swap_bottom.py): the
# straw's elevation and azimuth (deg, world) and the can's roll about the straw (deg), the valve kept as near straight
# down as that allows. The straw always slants in from outside the ring, so the rest of it clears the pipe stub. The
# demo doses the left, top and right (0, 2, 4) with the right holding the can and the left pressing; then the hands
# swap roles for the bottom (6): there the straw has to come in nearly level (it would hit the pipe stub otherwise),
# the can tipped ~40 deg off upside down, which the right's grip can't hold within its wrist's range once the can has
# turned the ~12 deg in that grip seen on the GPU (stage9).
RING_ATTITUDES = {0: (-42.0, -14.5, -15.9), 2: (-41.2, -12.2, -10.2), 4: (-39.9, -16.1, -13.3), 6: (-14.3, -24.1, 5.2)}
RING_ORDER = (0, 2, 4)            # left, top, right (sectors 0, 9, 18), right holding
SWAP_POINT = 6                    # then the bottom (sector 27), the roles swapped: the left holds, the right presses
# the left's grip on the can's body for the swap (offline search): (how far up the can, m; its finger axis across the
# can from the can's x toward its y, deg; its approach tipped along the can's axis, deg)
SWAP_GRIP = (0.131, 305.2, 31.7)
RING_BACK = 0.04                  # the straw goes in and out along its own line by this much
RING_TIP_R = 0.016                # the tip's radius from the pipe's axis: 2 mm inside the gap's middle, because the
                                  # slanted straw crosses the wall face ~4 mm further out than its tip
RING_TIP_R_AT = {2: 0.0150, 4: 0.0195, 6: 0.0185}   # points where the straw passes closer to the hole's edge (2:
                                  # tip further in) or to the pipe and its stub (4, 6: further out); offline clearance check
# where each point is aimed round the ring, deg from its nominal angle (+: counter-clockwise seen from the robot): the
# press pushes the can and the soft holding arm lets the tip drift along the gap before its servo catches up, the same
# way every time (GPU runs stage9/10/10b: the foam landed +1.2, -0.6, -1.8, -0.9 sectors off at points 0, 2, 4, 6);
# point 4's correction is held to 12 deg by the straw's clearance to the pipe stub, so it still lands ~1.5 sectors
# early, and point 6 is left at its own angle to close up behind it (the foam model on the GPU runs' doses: the ring
# fills even with point 4 landing a sector either way)
RING_ANGLE_TRIM = {0: -12.0, 2: 6.0, 4: 12.0}

RING_IN_DEPTH = 0.007           # how far the straw pushes into the gap (along the hole's axis): the slanted straw
                                  # stays inside the 14 mm gap over this depth
# The presser works the trigger with the back of a finger: the hand comes in from the side (along the can's y,
# PRESS_SIDE, turned PRESS_YAW toward the straw), its jaw shut and closing along (about) the can's axis, so the two
# fingers make one thin block that slides in between the trigger and the straw; the hand then moves toward the can,
# the lower finger's back pushing the trigger's end down. Nothing hangs below the trigger (no room there by the can).
# No pinch is possible: the holding hand takes the push.
PRESS_GRIP = 0.0                  # jaw shut
PAD_HALF_W = 0.0132               # the fingertip's half-width (the last 16 mm of the finger), across the approach
PRESS_X = 0.050                   # the tool point along the can's x: the fingertip over the trigger's end and lip
PRESS_TIP_IN = 0.007              # the trigger's centre this far in from the tool point along the approach (the
                                  # thin fingertip over it, the thicker finger behind it off the straw's plane)
PRESS_SIDE = 1.0                  # approach along +/- the can's y ...
PRESS_YAW = math.radians(17.6)    # ... turned this far (rad) about the can's axis toward its +x (in under the straw)
PRESS_TILT = math.radians(22.6)   # the pad tilted this far (rad) about the approach (+: its outer end lower, like the
                                  # pressed trigger)
PRESS_FLIP = -1.0                 # the hand the right way up (+1) or turned 180 deg about its approach (-1): the
                                  # fingers are alike, the wrist is not
PRESS_HOVER = 0.002               # the fingertip this far above the trigger before pressing
PRESS_PRELOAD = 0.0015            # and this far into it while pressing (the arm's own compliance)
PRESS_LIFT = 0.004                 # clear of it after a dose (press -1): a little, the straw is just above
# the shut fingers' outer faces (tool frame, m): (along the approach from, to, half-width across it, depth below the
# jaw centre): the thin tips, then the thicker fingers behind them (collision model)
PAD_ZONES = ((-0.003, 0.0128, 0.0132, 0.0063), (-0.0207, -0.003, 0.0201, 0.0100))
SQUEEZE_S = 1.5                   # how long each press is held (squeeze stage)
# set points pressed differently (offline search): k -> (PRESS_YAW deg, PRESS_TILT deg, PRESS_FLIP, PRESS_SIDE); the
# bottom is pressed by the right hand (SWAP_POINT)
PRESS_AT: dict[int, tuple[float, float, float, float]] = {6: (5.4, 27.2, -1.0, -1.0)}


def press_params(k: int | None = None) -> tuple[float, float, float, float]:
    """The press's yaw, tilt (rad), flip and side at set point k (None: the default)."""
    if k in PRESS_AT:
        yaw, tilt, flip, side = PRESS_AT[k]
        return math.radians(yaw), math.radians(tilt), flip, side
    return PRESS_YAW, PRESS_TILT, PRESS_FLIP, PRESS_SIDE


def press_approach(k: int | None = None) -> np.ndarray:
    """The presser's approach direction (can frame) at set point k: level in the can's frame, square to the straw's
    plane, turned PRESS_YAW toward the straw."""
    yaw, _, _, side = press_params(k)
    return np.array([math.sin(yaw), side * math.cos(yaw), 0.0])


def _trigger_points(press: float, n: int = 61) -> np.ndarray:
    """Points on the trigger's top face, can frame, pressed ``press`` (0..1)."""
    a = LEVER_REST_DOWN + float(np.clip(press, 0.0, 1.0)) * LEVER_TRAVEL
    c, s_ = math.cos(a), math.sin(a)
    pts = [(LEVER_HINGE[0] + t * c, y, LEVER_HINGE[2] - t * s_)
           for t in np.linspace(0.0, LEVER_LEN, n) for y in np.linspace(-LEVER_W / 2, LEVER_W / 2, 5)]
    return np.array(pts)


def press_normal(k: int | None = None) -> np.ndarray:
    """The pad's outward normal (can frame) at set point k, from the jaw centre toward the pushing pad: the can's +z
    tilted PRESS_TILT about the approach."""
    a = press_approach(k)
    tilt = press_params(k)[1]
    e = np.cross([0.0, 0.0, 1.0], a)           # level, square to the approach
    if e[0] < 0 or (e[0] == 0 and e[1] < 0):
        e = -e                                 # toward the trigger's end: a positive tilt lowers the pad's outer end
    return np.array([0.0, 0.0, 1.0]) * math.cos(tilt) + e * math.sin(tilt)


def press_point(press: float, k: int | None = None) -> np.ndarray:
    """The presser's tool point (can frame) for a press fraction: the trigger's centre line at PRESS_X lies
    PRESS_TIP_IN along the approach from the tool point, and the shut fingers' lower faces (PAD_ZONES, below the jaw
    centre along press_normal) rest on the trigger's highest point under them (PRESS_HOVER above it at 0,
    PRESS_PRELOAD into it once pressing, PRESS_LIFT clear of it at -1), at set point k (PRESS_AT)."""
    a, n = press_approach(k), press_normal(k)
    w = np.cross(a, [0.0, 0.0, 1.0])           # the footprint, taken level (the tilt is small)
    x0, y0 = np.array([PRESS_X, 0.0]) - PRESS_TIP_IN * a[:2]
    rel = _trigger_points(press) - np.array([x0, y0, 0.0])
    u, v = rel @ a, np.abs(rel @ w)
    need = -1.0
    for u0, u1, hw, depth in PAD_ZONES:
        under = (u >= u0) & (u <= u1) & (v <= hw)
        if under.any():
            need = max(need, float((rel[under] @ n).max()) + depth)
    off = float(np.interp(press, (-1.0, 0.0, 0.1), (PRESS_LIFT, PRESS_HOVER, -PRESS_PRELOAD)))
    return np.array([x0, y0, (need + off) / n[2]])


def can_start_pos() -> np.ndarray:
    """Where the can's base centre starts: on the table."""
    return np.array([CAN_XY[0], CAN_XY[1], TOP])


def can_rot_xyzw() -> tuple[float, float, float, float]:
    """The can's starting orientation: upright, turned CAN_YAW about the vertical."""
    return (0.0, 0.0, math.sin(CAN_YAW / 2), math.cos(CAN_YAW / 2))

# -- dispensing --------------------------------------------------------------------------------------------------
DEAD_ZONE = 0.25                  # of the lever's travel, before any foam
Q_MAX = 6.0e-6                    # m3/s of liquid at full press (6 cm3/s)
FLOW_EXP = 1.5                    # Q = Q_MAX * p^1.5 past the dead zone
GATE_FULL = math.radians(60)      # valve within this of straight down: full flow
GATE_NONE = math.radians(90)      # fading to none here (and none with the can upright)

# -- the foam ---------------------------------------------------------------------------------------------------
EXPANSION = 2.0                   # the dispensed bead doubles as it cures (so a gap filled halfway ends up full)
TAU = 3.0                         # s: rise time constant, standing in for minutes of real rise
SPREAD_VENT = 0.02                # per round of spreading along the gap, this share of a sector's excess escapes out
SPREAD_ROUNDS = 400               # of the face as a bulge instead (one 9.5 cm3 dose fills ~9-10 sectors)
BULGE_FOOTPRINT = 3.0             # bulging foam mushrooms over this many times its sector's face on the wall
MESS_THICK = 0.010                # cured foam outside the ring lies about this thick

# -- the demo's dose at each point --------------------------------------------------------------------------------------
DOSE_PRESS = 1.0                  # the lever pressed against its stop
DOSE_CC = 9.5                     # cm3 of liquid per point, 38 for the four (a little more than the gap's half, 30 cm3:
                                  # dosing at points has to cover the spaces between them; the foam model with the doses
                                  # where they land on the GPU: 9.5 fills the whole ring with an 8-9 mm bulge)
DOSE_TAIL_CC = 1.3                # about this much still comes out while the trigger is let go (GPU stage10b)
DOSE_S = 1.45                     # s held at the stop when nothing measures the dose (offline dry runs)
DOSE_S_MAX = 2.2                  # and never longer than this

# -- scoring ---------------------------------------------------------------------------------------------------------
SECTOR_FULL = 0.90                # a sector counts as sealed when its cured foam fills 90% of its volume: the stand-in
                                  # for the real trial's check, foam spanning the gap at the face across the sector (the
                                  # straw puts the foam in near the face, so the face closes before the back)
COVERAGE_TIERS = ((0.25, 0.25), (0.50, 0.5), (0.75, 0.75), (0.95, 1.0))   # (share of the ring filled, score)
BULGE_OK, BULGE_MAX = 0.010, 0.020   # m proud of the wall face: -0.25 past the first, score capped at 0.5 past the second
RISE_SHOWN = 0.95                 # success waits until the shown rise is this far along
SUCCESS_HOLD_S = 1.0
