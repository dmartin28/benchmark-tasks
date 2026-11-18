"""Endpoint tiers, with physical extraction history and ordinary ruler checks."""
from dataclasses import dataclass
from collections import deque
import numpy as np


def fits_tray(vertices,size):
    """Every nail vertex must lie inside the walls and below the actual rim.

    A 0.5 mm floor tolerance accommodates rigid-contact solver penetration;
    the rim and side walls use only a 1 micrometre floating-point tolerance.
    """
    vertices=np.asarray(vertices);size=np.asarray(size)
    return bool(np.all(np.abs(vertices[:,:2])<=size[:2]/2-.003+1e-6) and
                vertices[:,2].min()>=.0010 and
                vertices[:,2].max()<=size[2]-.0015+1e-6)


class StabilityWindow:
    """Compare the entire final second, so slow drift cannot count as rest."""
    def __init__(self):
        self.frames=deque()
        self.time=0.

    def update(self,poses,dt,clear):
        self.time+=dt
        if not clear:
            self.frames.clear()
            return 0.
        self.frames.append((self.time,{n:(p.copy(),q.copy()) for n,(p,q) in poses.items()}))
        while len(self.frames)>1 and self.frames[1][0]<=self.time-1.:
            self.frames.popleft()
        for _,frame in self.frames:
            if any(np.linalg.norm(p-frame[n][0])>.002 or
                   abs(float(np.dot(q,frame[n][1])))<np.cos(np.radians(2.)/2)
                   for n,(p,q) in poses.items()):
                return 0.
        return self.time-self.frames[0][0]


@dataclass(frozen=True)
class Outcome:
    """`extraction_m` is the head's rise normal to the board for ruler tiers."""
    extraction_m: float = 0.
    claw_work_m: float = 0.
    nail_free: bool = False
    nail_in_tray: bool = False
    board_on_table: bool = False
    hammer_on_table: bool = False
    tray_on_table: bool = False
    hands_clear: bool = False
    stable_s: float = 0.
    invalid: bool = False


def score(outcome: Outcome):
    """Score only released, stable endpoints; no best-score latch.

    The claw must do at least 5 mm of withdrawal. Finishing an already loosened
    nail by hand is legitimate; pulling it entirely by hand earns no credit.
    """
    o=outcome
    if (o.invalid or not o.board_on_table or not o.hammer_on_table or
            not o.tray_on_table or not o.hands_clear or o.stable_s < 1.-1e-6 or
            o.claw_work_m < .005-1e-6):
        return 0.
    if o.nail_free:
        return 1. if o.nail_in_tray else .75
    if o.extraction_m >= .010-1e-6:
        return .50
    if o.extraction_m >= .005-1e-6:
        return .25
    return 0.
