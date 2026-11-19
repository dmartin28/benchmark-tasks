"""Every dimension, mass and threshold for oil-squeaky-hinge, in one place (metres, kilograms, seconds, radians).

World frame (yam_arena's): x forward from the line through the two arm bases, y to the robot's left, z up; the arm
bases are at y +0.305 and -0.305. ``TOP`` is the table top.

The cabinet's own frame (its base frame in task.py): origin on the table top, on the plane of the cabinet's front
face, at the inner face of its right-hand side board (the robot's right); x into the cabinet, y to the robot's left,
z up.
"""

from __future__ import annotations

import math

TOP = 0.7474999999999999  # yam_arena.station.TABLE_TOP
MM = 0.001
DEG = math.pi / 180.0

# -- the cabinet (fixed): a small pine wall cupboard standing on the counter against the wall --------------------------
CAB_X = 0.460  # world x of its front face
CAB_Y = 0.0  # world y of the inside face of its right-hand side board: the hinge side
BOARD = 18 * MM  # pine boards, 18 mm
CAB_DEPTH = 0.200  # front to back, outside
OPEN_W = 0.270  # the door opening, between the two side boards
OPEN_LO, OPEN_HI = BOARD, 0.340 - BOARD  # the door opening, from the table: above the bottom board, below the top
CAB_H = 0.340
BACK_T = 6 * MM  # hardboard back
REVEAL = 2 * MM  # the gap round the inset door

# The door: inset, flush with the front, 18 mm pine, hung on two 2-1/2 in butt hinges on its right-hand edge.
DOOR_T = 18 * MM
DOOR_Y0, DOOR_Y1 = REVEAL, OPEN_W - REVEAL  # cabinet frame
DOOR_Z0, DOOR_Z1 = OPEN_LO + REVEAL, OPEN_HI - REVEAL
DOOR_W = DOOR_Y1 - DOOR_Y0  # 266 mm
DOOR_H = DOOR_Z1 - DOOR_Z0  # 300 mm
DOOR_MASS = 0.62  # kg: 266 x 300 x 18 mm of pine at about 430 kg/m^3, and the knob
DOOR_OPEN_MAX = 110 * DEG  # the side board stops it a little past square
DOOR_FRICTION = 0.08  # N m at the pin: a dry, squeaky hinge pair (sim joint friction)
DOOR_DAMPING = 0.05  # N m s/rad

# The knob: a turned wooden knob, 28 mm across, on a 12 mm neck, 20 + 12 mm proud of the door, on the door's
# left-hand stile at mid height.
KNOB_Y = DOOR_Y1 - 35 * MM
KNOB_Z = (DOOR_Z0 + DOOR_Z1) / 2
KNOB_D, KNOB_L = 28 * MM, 20 * MM
NECK_D, NECK_L = 12 * MM, 12 * MM

# The butt hinges: 2-1/2 in (63.5 mm) steel, brass plated, five knuckles (12.7 mm each) on a 9 mm barrel, loose pin
# with a button head. The barrel stands just proud of the face, its axis at the joint between the door's edge and
# the side board: the door turns about it.
HINGE_LEN = 63.5 * MM
KNUCKLES = 5
KNUCKLE_L = HINGE_LEN / KNUCKLES  # 12.7 mm
BARREL_R = 4.5 * MM
PIN_HEAD_R, PIN_HEAD_H = 4.5 * MM, 2.0 * MM
AXIS_X = -BARREL_R  # cabinet frame: the pin's axis, just in front of the face
AXIS_Y = REVEAL / 2
HINGE_INSET = 50 * MM  # each hinge's centre from the door's top or bottom edge
TOP_HINGE_Z = DOOR_Z1 - HINGE_INSET  # cabinet frame (from the table top)
BOTTOM_HINGE_Z = DOOR_Z0 + HINGE_INSET
GAP_DRAWN = 0.5 * MM  # the knuckles are drawn this far apart, so each gap shows as a dark line


def knuckles(centre_z: float) -> list[tuple[float, float, bool]]:
    """(bottom z, top z, on the door) of each knuckle of one hinge, bottom up. The 1st, 3rd and 5th are the frame
    leaf's (fixed); the 2nd and 4th the door leaf's (they turn with the door)."""
    z0 = centre_z - HINGE_LEN / 2
    out = []
    for k in range(KNUCKLES):
        lo, hi = z0 + k * KNUCKLE_L, z0 + (k + 1) * KNUCKLE_L
        out.append((lo + GAP_DRAWN / 2, hi - GAP_DRAWN / 2, k % 2 == 1))
    return out


def seams(centre_z: float = TOP_HINGE_Z) -> list[float]:
    """Heights (from the table top) of the four gaps between one hinge's knuckles, bottom up."""
    z0 = centre_z - HINGE_LEN / 2
    return [z0 + k * KNUCKLE_L for k in range(1, KNUCKLES)]


SEAM_NAMES = ("gap 1 (lowest)", "gap 2", "gap 3", "gap 4 (highest)")


def axis_world() -> tuple[float, float]:
    """World (x, y) of the hinges' pin axis."""
    return CAB_X + AXIS_X, CAB_Y + AXIS_Y


def cabinet_to_world(p) -> tuple[float, float, float]:
    return (CAB_X + p[0], CAB_Y + p[1], TOP + p[2])


# -- the can: an 8 oz WD-40 Multi-Use Product can with the Smart Straw (straw flipped up: the stream setting) -----------
# Can frame: origin at the centre of its base, z up its axis, the straw along +x.
# Sizes: the maker's product data give the 8 oz can as 0.5625 x 0.21875 ft (171 x 66.7 mm); the top's proportions
# (dome, pad, nozzle, straw) are measured off the maker's product photos, scaled to that size.
CAN_R = 33.3 * MM  # 66.7 mm steel can
CAN_BODY_H = 119 * MM  # the straight body (blue, with the yellow label panel), up to the top seam
SEAM_R, SEAM_H = 34.0 * MM, 3 * MM  # the rolled top seam, a little proud of the body
DOME = ((25 * MM, CAN_BODY_H, 131 * MM), (16 * MM, 131 * MM, 139 * MM))  # the silver dome: (radius, z from, z to)
PAD_Z0 = 139 * MM  # the red Smart Straw actuator pad sits on the dome's top...
ACT_TOP = 171 * MM  # ...its flat top is the can's overall height
PAD_X = (-2 * MM, 34 * MM)  # x from-to of its flat-topped middle (rounded ends below)
PAD_W = 32 * MM  # its width across (y)
PAD_BACK_R, PAD_FRONT_R = 16 * MM, 13 * MM  # its rounded back (over the can) and front (over the nozzle) ends
PRESS_X = 6 * MM  # where a finger presses it: over the valve stem (x 0), on the flat top
NOZZLE_FROM, NOZZLE_TO = (47 * MM, 149 * MM), (81 * MM, 156.5 * MM)  # the grey nozzle arm under the pad's front
NOZZLE_R = 3.5 * MM
NOZZLE_Z = NOZZLE_TO[1]  # the straw leaves the nozzle level, this high
STRAW_R = 1.5 * MM  # 3 mm red straw
STRAW_X0 = NOZZLE_TO[0]  # it starts at the nozzle's end...
STRAW_L = 105 * MM  # ...and runs 105 mm out from it
STRAW_TIP = (STRAW_X0 + STRAW_L, 0.0, NOZZLE_Z)  # can frame, valve at rest: 186 mm out from the can's axis
CAN_MASS = 0.31  # kg: 227 g (8 oz) of product, propellant and the can
CAN_FRICTION = (1.0, 0.9)  # painted steel against the grippers' rubber pads
VALVE_TRAVEL = 3.0 * MM  # the actuator goes down this far at most
VALVE_K = 6700.0  # N/m: about 10 N at the spraying threshold, 20 N at full travel
VALVE_DAMPING = 5.0
SPRAY_AT = 1.5 * MM  # the valve is open (the straw sprays) with the actuator pressed at least this far

CAN_START = (0.280, -0.080)  # world x, y of its base's centre: standing on the table, in front of the cabinet, right of it
CAN_START_YAW = 0.0  # the straw points along +x (away from the robot)

# -- the spray (a rule: the stream is a straight line from the straw's tip along it) ------------------------------------
JET_RANGE = 0.50  # m: farther than this the rule credits nothing (a straw's stream reaches about that far)
SEAM_BAND = 3.0 * MM  # the stream lands in a gap when it meets the barrel within this of the gap's height
SEAM_DOSE = 0.10  # s of stream in a gap for it to count as oiled: a burst that lands there, not a stream passing over
FLOW_G_PER_S = 0.81  # g/s out through the straw (a published stream test of WD-40 with its straw: 0.81 g/s)
SPRAY_LIMIT_G = 4.0  # g: the can losing more than this in a trial (weighed before and after) floods the hinge: 0
SPRAY_LIMIT = SPRAY_LIMIT_G / FLOW_G_PER_S  # s of open valve that sprays that much: about 4.9 s

# -- working the door ---------------------------------------------------------------------------------------------------
SWING_OPEN = 45 * DEG  # a swing: the door opened at least this far...
SWING_SHUT = 10 * DEG  # ...and back to within this of shut
SWINGS_SOME = 1  # swings after the last gap was oiled for 0.75...
SWINGS = 3  # ...and for full score

# -- the end ---------------------------------------------------------------------------------------------------------------
DOOR_SHUT = 5 * DEG  # left shut: within this of closed
CAN_UPRIGHT = 10 * DEG  # the can left standing: its axis within this of vertical...
CAN_ON_TABLE = 5 * MM  # ...its base within this of the table top...
CAN_OFF_CABINET = 10 * MM  # ...and clear of the cabinet: not leaning on it or standing on it
CLEAR = 0.03  # both grippers at least this far from the can and the door
STILL_V = 0.02  # m/s
SUCCESS_HOLD_S = 1.0

# -- colours -------------------------------------------------------------------------------------------------------------
PINE = (0.80, 0.64, 0.42)
PINE_DARK = (0.70, 0.54, 0.34)
HARDBOARD = (0.45, 0.36, 0.26)
BRASS = (0.78, 0.64, 0.30)
KNOB_WOOD = (0.55, 0.36, 0.20)
CAN_BLUE = (0.10, 0.25, 0.60)
CAN_YELLOW = (0.95, 0.80, 0.10)
CAN_SILVER = (0.74, 0.75, 0.77)
PAD_RED = (0.86, 0.06, 0.10)
NOZZLE_GREY = (0.55, 0.56, 0.58)
STRAW_RED = (0.85, 0.10, 0.10)
