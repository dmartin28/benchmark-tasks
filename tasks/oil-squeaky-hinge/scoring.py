"""Scoring for oil-squeaky-hinge: pure functions of the can's pose and valve, the door's angle and the arms' let-go
check, and a small recorder (:class:`Spray`) that the task runs once per control step.

The stream (a rule; simulators don't spray): while the can's actuator is pressed at least 1.5 mm (the valve open),
a straight stream leaves the straw's tip along the straw. Where it first meets the top hinge's barrel (within 50 cm,
not blocked by the door), the height of that point decides which gap between knuckles, if any, it is landing in:
within 3 mm of a gap's height (the knuckles are 12.7 mm long). A gap is oiled once a burst has landed there (0.1 s of
stream in all, so a stream merely passing over it doesn't count). The recorder also counts all the time the valve is
open, wherever the straw points, and turns it into grams sprayed at 0.81 g/s.

Working the door: after the fourth gap is oiled, each time the door is opened at least 45 deg and then brought back
to within 10 deg of shut counts one swing.

The real trial is scored with a kitchen scale, the trial video and a slow-motion close-up video of the hinge:

- the can is weighed before and after the trial (0.1 g scale): the grams it lost are what was sprayed;
- a gap is oiled when a burst lands on the barrel within 3 mm of it (between the tape marks for that gap);
- swings, the door, the can and the robot's hands are judged from the trial video.

Tiers (the highest that holds at the end of the trial):
  0     the can lost more than 4 g (about 4.9 s of spraying: the hinge and cabinet flooded), whatever else holds;
  1.0   all four gaps between the top hinge's knuckles oiled, then the door swung open at least 45 deg and back to
        within 10 deg of shut at least three times, and at the end the door is within 5 deg of shut, the can stands
        upright on the table (axis within 10 deg of vertical, base within 5 mm of the table top, clear of the
        cabinet), and the robot has let go of both (in the simulation: both grippers at least 3 cm from the can and
        the door, and still);
  0.75  all four gaps oiled and the door swung at least once after that;
  0.5   all four gaps oiled;
  0.25  at least two gaps oiled;
  0     otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import layout as L

N_SEAMS = L.KNUCKLES - 1


# -- geometry ----------------------------------------------------------------------------------------------------------
def seam_heights_world() -> list[float]:
    return [L.TOP + z for z in L.seams()]


def top_hinge_span_world() -> tuple[float, float]:
    c = L.TOP + L.TOP_HINGE_Z
    return c - L.HINGE_LEN / 2, c + L.HINGE_LEN / 2


def straw_jet(can_rot, can_origin, valve: float) -> tuple[np.ndarray, np.ndarray]:
    """World (tip, unit direction) of the straw. ``valve`` is how far the actuator is pressed (m): the straw rides
    on it."""
    r = np.asarray(can_rot, float)
    tip_local = np.array(L.STRAW_TIP) - np.array([0.0, 0.0, float(valve)])
    tip = np.asarray(can_origin, float) + r @ tip_local
    d = r @ np.array([1.0, 0.0, 0.0])
    return tip, d / np.linalg.norm(d)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def door_box_world(door_angle: float):
    """The door slab (without the knob) as (centre, half sizes, rotation) in the world, turned ``door_angle`` (rad,
    positive open) about the pin."""
    ax, ay = L.axis_world()
    pivot = np.array([ax, ay, 0.0])
    c_cab = np.array([L.DOOR_T / 2, (L.DOOR_Y0 + L.DOOR_Y1) / 2, (L.DOOR_Z0 + L.DOOR_Z1) / 2])
    c_rest = np.array(L.cabinet_to_world(c_cab))
    r = _rot_z(door_angle)
    centre = pivot + r @ (c_rest - pivot)
    half = np.array([L.DOOR_T / 2, L.DOOR_W / 2, L.DOOR_H / 2])
    return centre, half, r


def segment_hits_box(p0, p1, centre, half, rot, shrink: float = 0.5 * L.MM) -> bool:
    """Whether the segment p0-p1 passes through the box (slab test), the box shrunk by ``shrink`` on every side."""
    rt = np.asarray(rot, float).T
    a = rt @ (np.asarray(p0, float) - centre)
    b = rt @ (np.asarray(p1, float) - centre)
    h = np.asarray(half, float) - shrink
    d = b - a
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if abs(a[k]) > h[k]:
                return False
            continue
        u0, u1 = (-h[k] - a[k]) / d[k], (h[k] - a[k]) / d[k]
        lo, hi = min(u0, u1), max(u0, u1)
        t0, t1 = max(t0, lo), min(t1, hi)
        if t0 > t1:
            return False
    return True


def barrel_hit(tip, direction, door_angle: float = 0.0):
    """Where the stream from ``tip`` along ``direction`` first meets the top hinge's barrel: the world point, or None
    (it misses the barrel, meets it beyond the stream's reach, or the door is in the way)."""
    p, d = np.asarray(tip, float), np.asarray(direction, float)
    ax, ay = L.axis_world()
    rx, ry = p[0] - ax, p[1] - ay
    lo, hi = top_hinge_span_world()
    if rx * rx + ry * ry <= L.BARREL_R ** 2:  # the tip is pressed into the barrel itself
        return p.copy() if lo <= p[2] <= hi else None
    a = d[0] * d[0] + d[1] * d[1]
    if a < 1e-9:  # straight up or down, along the barrel
        return None
    b = 2.0 * (rx * d[0] + ry * d[1])
    c = rx * rx + ry * ry - L.BARREL_R ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b - math.sqrt(disc)) / (2 * a)
    if t < 0 or t > L.JET_RANGE:
        return None
    hit = p + t * d
    if not lo <= hit[2] <= hi:
        return None
    centre, half, rot = door_box_world(door_angle)
    if segment_hits_box(p, hit, centre, half, rot):
        return None
    return hit


def seam_at(z: float):
    """Index of the gap (0..3, bottom up) whose height is within the band of ``z`` (world), or None."""
    for k, s in enumerate(seam_heights_world()):
        if abs(z - s) <= L.SEAM_BAND:
            return k
    return None


# -- the recorder ----------------------------------------------------------------------------------------------------------
@dataclass
class Spray:
    """What has happened so far in a trial: the stream's time in each gap, the valve's total open time, and the
    door's swings after the last gap was oiled."""

    dose: list = field(default_factory=lambda: [0.0] * N_SEAMS)
    spray_time: float = 0.0
    oiled_step: int | None = None  # the step at which the fourth gap became oiled
    swings: int = 0
    _opened: bool = False
    step_n: int = 0
    max_door: float = 0.0

    def reset(self) -> None:
        self.dose = [0.0] * N_SEAMS
        self.spray_time = 0.0
        self.oiled_step = None
        self.swings = 0
        self._opened = False
        self.step_n = 0
        self.max_door = 0.0

    @property
    def oiled(self) -> list[bool]:
        return [d >= L.SEAM_DOSE - 1e-9 for d in self.dose]

    @property
    def n_oiled(self) -> int:
        return sum(self.oiled)

    @property
    def sprayed_g(self) -> float:
        return self.spray_time * L.FLOW_G_PER_S

    @property
    def flooded(self) -> bool:
        return self.sprayed_g > L.SPRAY_LIMIT_G + 1e-9

    def step(self, valve: float, can_rot, can_origin, door_angle: float, dt: float):
        """One control step. Returns the gap index the stream landed in this step (or None)."""
        self.step_n += 1
        landed = None
        if valve >= L.SPRAY_AT:
            self.spray_time += dt
            tip, d = straw_jet(can_rot, can_origin, valve)
            hit = barrel_hit(tip, d, door_angle)
            if hit is not None:
                landed = seam_at(float(hit[2]))
                if landed is not None:
                    self.dose[landed] += dt
        if self.oiled_step is None and self.n_oiled == N_SEAMS:
            self.oiled_step = self.step_n
            self._opened = False
        if self.oiled_step is not None:
            self.max_door = max(self.max_door, door_angle)
            if door_angle >= L.SWING_OPEN:
                self._opened = True
            elif self._opened and door_angle <= L.SWING_SHUT:
                self.swings += 1
                self._opened = False
        return landed


# -- the end state ----------------------------------------------------------------------------------------------------------
def can_tilt(can_rot) -> float:
    return math.acos(max(-1.0, min(1.0, float(np.asarray(can_rot, float)[2, 2]))))


def cabinet_footprint_world() -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of the cabinet's footprint on the table (door shut)."""
    return (L.CAB_X + L.AXIS_X - L.BARREL_R, L.CAB_X + L.CAB_DEPTH, L.CAB_Y - L.BOARD, L.CAB_Y + L.OPEN_W + L.BOARD)


def clear_of_cabinet(can_origin) -> bool:
    x0, x1, y0, y1 = cabinet_footprint_world()
    x, y = float(can_origin[0]), float(can_origin[1])
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy) >= L.CAN_R + L.CAN_OFF_CABINET


def can_standing(can_rot, can_origin) -> bool:
    return (can_tilt(can_rot) <= L.CAN_UPRIGHT and abs(float(can_origin[2]) - L.TOP) <= L.CAN_ON_TABLE
            and clear_of_cabinet(can_origin))


@dataclass
class Reading:
    score: float
    why: str
    n_oiled: int
    swings: int
    spray_time: float
    door_deg: float
    can_standing: bool
    released: bool
    still: bool


def judge(spray: Spray, door_angle: float, can_rot, can_origin, released: bool, still: bool) -> Reading:
    standing = can_standing(can_rot, can_origin)
    shut = abs(door_angle) <= L.DOOR_SHUT
    n, sw = spray.n_oiled, spray.swings

    def r(score, why):
        return Reading(score, why, n, sw, round(spray.spray_time, 3), round(math.degrees(door_angle), 2), standing,
                       released, still)

    if spray.flooded:
        return r(0.0, f"flooded: {spray.sprayed_g:.1f} g sprayed ({spray.spray_time:.2f} s; limit {L.SPRAY_LIMIT_G} g)")
    if n == N_SEAMS and sw >= L.SWINGS and shut and standing and released and still:
        return r(1.0, "all four gaps oiled, the door worked, can stood on the table, door shut, let go")
    if n == N_SEAMS and sw >= L.SWINGS_SOME:
        return r(0.75, f"all four gaps oiled and the door swung {sw} times after; not full score "
                       f"(door shut {shut}, can standing {standing}, let go {released}, still {still})")
    if n == N_SEAMS:
        return r(0.5, f"all four gaps oiled; door swung {sw} times after")
    if n >= 2:
        return r(0.25, f"{n} gaps oiled")
    return r(0.0, f"{n} gaps oiled")
