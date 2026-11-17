"""The foam model, pure NumPy: how much liquid foam leaves the straw for a given lever press and can attitude, where it
goes (a ring sector of the gap round the pipe, or mess outside it), how it rises, and how it spreads along the gap.

Spreading: the ring is 36 sectors, each holding SECTOR_VOLUME. Foam that rises beyond a sector's room pushes along
the gap into its neighbours; on each round of spreading a share of the excess (SPREAD_VENT) escapes through the open
face as a bulge instead, so foam squeezed into one spot travels a limited way along the gap and the rest stands proud
of the wall. The result depends only on how much foam is in each sector, not the order it went in. Depth through the
board is not modelled: a sector's fill is averaged over the board's thickness.
"""

from __future__ import annotations

import math

import numpy as np

import geometry as g


def press_fraction(lever_angle: float) -> float:
    """How far the lever is pressed, 0 (at rest) to 1 (against its stop)."""
    return float(np.clip(lever_angle / g.LEVER_TRAVEL, 0.0, 1.0))


def valve_from_down(rot: np.ndarray) -> float:
    """Angle (rad) between the valve's direction (the can's +z, toward the adapter) and straight down."""
    v = np.asarray(rot, float) @ g.VALVE_DIR
    return math.acos(float(np.clip(-v[2], -1.0, 1.0)))


def gate(angle_from_down: float) -> float:
    """Share of the flow that is foam: 1 within GATE_FULL of straight down, fading to 0 at GATE_NONE."""
    if angle_from_down <= g.GATE_FULL:
        return 1.0
    if angle_from_down >= g.GATE_NONE:
        return 0.0
    return (g.GATE_NONE - angle_from_down) / (g.GATE_NONE - g.GATE_FULL)


def flow_rate(press: float, angle_from_down: float) -> float:
    """Liquid foam leaving the straw (m3/s): nothing in the lever's dead zone, then Q_MAX * p^1.5 past it, times
    the upside-down gate."""
    p = (press - g.DEAD_ZONE) / (1.0 - g.DEAD_ZONE)
    if p <= 0.0:
        return 0.0
    return g.Q_MAX * min(p, 1.0) ** g.FLOW_EXP * gate(angle_from_down)


# -- where the foam goes ------------------------------------------------------------------------------------------------
SECTOR_VOLUME = g.RING_VOLUME / g.SECTORS
SECTOR_FACE = math.pi * (g.RING_OUT**2 - g.RING_IN**2) / g.SECTORS     # a sector's open face on the wall (m2)


def ring_cell(point) -> tuple[int, float] | None:
    """The ring sector (0 at the robot's left, +y, counting counter-clockwise seen from the robot) and the depth into
    the gap (m) of a world point, or None if the point is not inside the gap."""
    p = np.asarray(point, float)
    depth = p[0] - g.FACE_X
    if not 0.0 <= depth <= g.RING_DEPTH:
        return None
    dy, dz = p[1] - g.HOLE_CENTRE[1], p[2] - g.HOLE_CENTRE[2]
    r = math.hypot(dy, dz)
    if not g.RING_IN <= r <= g.RING_OUT:
        return None
    angle = math.atan2(dz, dy) % (2 * math.pi)
    return int(angle / (2 * math.pi / g.SECTORS)) % g.SECTORS, depth


def risen(v0: float, age: float) -> float:
    """Volume (m3) of a deposit of v0 liquid, ``age`` seconds after it left the straw."""
    return v0 * (1.0 + (g.EXPANSION - 1.0) * (1.0 - math.exp(-max(age, 0.0) / g.TAU)))


def spread(volumes) -> tuple[np.ndarray, np.ndarray]:
    """Foam volume per sector (m3) as it lands, to (the foam held in each sector, the foam bulging out of each
    sector's face). Rounds of: every sector's excess over its room goes half to each neighbour, less SPREAD_VENT of it,
    which vents out of the face where it is. Whatever is still in excess at the end bulges too. Conserves volume."""
    v = np.asarray(volumes, float) / SECTOR_VOLUME                   # in sector-fulls
    bulge = np.zeros_like(v)
    for _ in range(g.SPREAD_ROUNDS):
        e = np.maximum(v - 1.0, 0.0)
        if e.max() < 1e-9:
            break
        v -= e
        bulge += g.SPREAD_VENT * e
        e *= 1.0 - g.SPREAD_VENT
        v += 0.5 * (np.roll(e, 1) + np.roll(e, -1))
    e = np.maximum(v - 1.0, 0.0)
    return (v - e) * SECTOR_VOLUME, (bulge + e) * SECTOR_VOLUME


class Ledger:
    """Liquid foam deposited each control step, by ring sector or as mess, with the time it was laid down."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.deposits: list[tuple[int, float, float]] = []      # (sector, liquid m3, time)
        self.mess: list[tuple[np.ndarray, float, float]] = []    # (where, liquid m3, time)
        self.liquid_total = 0.0

    def step(self, tip, flow: float, dt: float) -> int | None:
        """Lay down ``flow * dt`` of liquid at the straw tip; returns the sector it went into, if any."""
        self.t += dt
        v = flow * dt
        if v <= 0.0:
            return None
        self.liquid_total += v
        cell = ring_cell(tip)
        if cell is None:
            self.mess.append((np.asarray(tip, float).copy(), v, self.t))
            return None
        self.deposits.append((cell[0], v, self.t))
        return cell[0]

    def ring_liquid(self) -> float:
        return sum(v for _, v, _ in self.deposits)

    def landed(self, cured: bool = False) -> np.ndarray:
        """Foam volume (m3) that went into each sector, as risen by now or fully risen (cured), before spreading."""
        out = np.zeros(g.SECTORS)
        for s, v, t0 in self.deposits:
            out[s] += v * g.EXPANSION if cured else risen(v, self.t - t0)
        return out

    def fill(self, cured: bool = False) -> np.ndarray:
        """Each sector's share filled with foam, 0..1, after spreading."""
        held, _ = spread(self.landed(cured))
        return held / SECTOR_VOLUME

    def bulge_heights(self, cured: bool = True) -> np.ndarray:
        """How far the foam stands proud of the wall face at each sector (m): the sector's bulging foam spread over
        BULGE_FOOTPRINT times the sector's face."""
        _, bulge = spread(self.landed(cured))
        return bulge / (g.BULGE_FOOTPRINT * SECTOR_FACE)

    def coverage(self, cured: bool = True) -> float:
        """Share of the ring's sectors filled to at least SECTOR_FULL."""
        return float(np.mean(self.fill(cured) >= g.SECTOR_FULL))

    def mess_volume(self, cured: bool = True) -> float:
        return sum(v * g.EXPANSION if cured else risen(v, self.t - t0) for _, v, t0 in self.mess)

    def mess_area(self) -> float:
        """Wall, pipe and floor area (m2) the cured mess covers, lying MESS_THICK thick."""
        return self.mess_volume(cured=True) / g.MESS_THICK

    def rise_shown(self) -> float:
        """How far along the rise of the most recent foam is, 0..1 (1 with no foam)."""
        if not self.deposits and not self.mess:
            return 1.0
        last = max([t for _, _, t in self.deposits] + [t for _, _, t in self.mess])
        return 1.0 - math.exp(-max(self.t - last, 0.0) / g.TAU)


def score(ledger: Ledger, dropped: bool = False) -> float:
    """The task's score from the cured foam: coverage tiers, less for bulge; 0 if the can was dropped. Foam that
    lands outside the gap (mess) earns nothing but costs nothing either: the score is judged at the gap's face."""
    if dropped:
        return 0.0
    c = ledger.coverage()
    base = 0.0
    for share, value in g.COVERAGE_TIERS:
        if c >= share:
            base = value
    bulge = float(ledger.bulge_heights().max())
    s = base
    if bulge > g.BULGE_OK:
        s -= 0.25
    if bulge > g.BULGE_MAX:
        s = min(s, 0.5)
    return max(0.0, s)
