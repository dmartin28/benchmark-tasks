"""Bead length and miter alignment, measured independently of the scripted controller."""
import math

import numpy as np

from .geometry import Pose,yaw_matrix
from .model import GlueModel
from .scene import HEAD,LEG,THICK,TOP,WIDTH

FACE_CENTER = np.array([WIDTH/2,0.,THICK/2])
ASSEMBLY_CENTER = np.array([.42,0.,TOP+THICK/2])
ASSEMBLY_HEAD_YAW = HEAD.yaw-math.pi/2


class ContactDwell:
    """Require a bounded pose over the entire cure interval, not a noisy substep velocity."""
    def __init__(self):
        self.anchor = None
        self.time = 0.

    def advance(self,dt,qualified,poses):
        if not qualified:
            self.anchor,self.time = None,0.
            return self.time
        moved = self.anchor is None
        if self.anchor is not None:
            for name,(p,r) in self.anchor.items():
                angle = np.arccos(np.clip((np.trace(r.T@poses[name].rot)-1)/2,-1.,1.))
                moved |= np.linalg.norm(poses[name].pos-p) > .0005 or angle > math.radians(.5)
        if moved:
            self.anchor = {n:(poses[n].pos.copy(),poses[n].rot.copy()) for n in (HEAD.name,LEG.name)}
            self.time = 0.
        self.time += dt
        return self.time


def bead_status(film):
    h = film.thickness
    wet = (h >= .00005) & film.mask
    columns = wet.any(axis=1)
    indices = np.flatnonzero(columns)
    longest = count = 0
    if len(indices):
        for value in columns[indices[0]:indices[-1]+1]:
            count = 0 if value else count+1
            longest = max(longest,count)
    length = float(columns.sum()*film.pitch[0])
    gap = float(longest*film.pitch[0])
    volume = float(film.volume.sum()*1e6)
    peak = float(h.max()*1000)
    return dict(qualified=bool(length >= .065 and gap <= .008 and .03 <= volume <= .15 and peak <= 1.),
                wet_length_mm=length*1000,largest_gap_mm=gap*1000,volume_ml=volume,max_mm=peak)


def mating_leg(head,gap=.0005):
    """Complementary cuts meet with opposed normals; both long axes then form 90 degrees."""
    rot = head.rot@yaw_matrix(-math.pi/2)
    normal = head.rot@next(s for s in HEAD.mesh()[2] if s.target).normal
    return Pose(head.world(FACE_CENTER)+normal*gap-rot@FACE_CENTER,rot)


def planar_mating_leg(head,gap=.025):
    """Prepare the mating pose on the table while the faces are still safely separated."""
    flat = Pose(np.array([head.pos[0],head.pos[1],TOP]),yaw_matrix(np.arctan2(head.rot[1,0],head.rot[0,0])))
    return mating_leg(flat,gap)


def wet_contact_length(model,poses):
    """Length where actual deposited glue can bridge to the opposing cut (50 um tolerance)."""
    best = 0.
    for name,film in model.films.items():
        owner = film.surface.owner
        other = LEG if owner == HEAD.name else HEAD
        surface = next(s for s in other.mesh()[2] if s.target)
        pose,op = poses[owner],poses[other.name]
        normal = op.rot@surface.normal
        world = pose.world(film.points)
        distance = (world-op.world(surface.origin))@normal
        projected = op.local(world-distance[...,None]*normal)
        height = film.thickness*max(0.,-(pose.rot@film.surface.normal)@normal)
        touches = (film.thickness >= .00005) & (np.abs(distance) <= height+.00005) & surface.contains(projected)
        best = max(best,float(touches.any(axis=1).sum()*film.pitch[0]))
    return best


def joint_geometry(poses):
    head,leg = poses[HEAD.name],poses[LEG.name]
    sh = next(s for s in HEAD.mesh()[2] if s.target)
    sl = next(s for s in LEG.mesh()[2] if s.target)
    a,b = head.world(sh.corners),leg.world(sl.corners)
    delta = a-b[np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2).argmin(axis=1)]
    n = head.rot@sh.normal
    normal_gap = np.abs(delta@n)
    tangent = delta-(delta@n)[:,None]*n
    angle = math.degrees(math.acos(float(np.clip(head.rot[:,0]@leg.rot[:,0],-1.,1.))))
    opposed = -(head.rot@sh.normal)@(leg.rot@sl.normal)
    flush = float(np.max(np.abs(delta@head.rot[:,2])))
    gap,tang = float(normal_gap.max()),float(np.linalg.norm(tangent,axis=1).max())
    flat = head.rot[2,2] >= math.cos(math.radians(5)) and leg.rot[2,2] >= math.cos(math.radians(5))
    seated = abs(angle-90) <= 3. and opposed >= math.cos(math.radians(3)) and gap <= .002 and tang <= .002 and flush <= .001 and flat
    bottom_error = max(float(np.abs(poses[spec.name].world(spec.mesh()[0][:4])[:,2]-TOP).max()) for spec in (HEAD,LEG))
    precise = (seated and abs(angle-90)<=1. and gap<=.0002 and tang<=.0005 and flush<=.0005 and bottom_error<=.001)
    return dict(seated=bool(seated),precision_seated=bool(precise),bottom_error_mm=bottom_error*1000,angle_deg=angle,gap_mm=gap*1000,
                tangent_error_mm=tang*1000,flush_mm=flush*1000,flat=bool(flat))


class AssemblyModel(GlueModel):
    def __init__(self,surfaces,require_bond=True,completion="released"):
        if completion not in ("released","held"): raise ValueError("Unknown completion criterion")
        self.completion = completion
        self.require_bond = require_bond
        super().__init__(surfaces)

    def reset(self):
        super().reset()
        self.joint = dict(seated=False)
        self.release_s = self.press_s = self.held_s = 0.
        self.held_dwell = ContactDwell()
        self.held_corner = None
        self.bonded = False
        self.last_joint_tick = None
        self.hand_wood_clearance_mm = None

    def coating_score(self):
        if self.failed:
            return 0.
        statuses = [bead_status(f) for f in self.films.values()]
        if any(s['qualified'] for s in statuses):
            return .5
        return .25 if any(s['wet_length_mm'] >= 10. and s['volume_ml'] >= .01 and s['max_mm'] <= 1. for s in statuses) else 0.

    def joint_step(self,tick,dt,poses,released,parked,bonded=False,held=False):
        if tick == self.last_joint_tick:
            return
        self.last_joint_tick = tick
        self.joint = joint_geometry(poses)
        self.bonded = bool(bonded)
        contact = self.coating_score() == .5 and self.joint['seated']
        self.press_s = self.press_s+dt if contact else 0.
        stable = poses[HEAD.name].still() and poses[LEG.name].still()
        self.ready = bool(contact and released and parked and stable and (bonded or not self.require_bond))
        self.release_s = self.release_s+dt if self.ready else 0.
        if self.completion == 'held':
            qualified = (contact and self.joint['precision_seated'] and held and parked and wet_contact_length(self,poses)>=.040
                         and (bonded or not self.require_bond))
            self.held_s = self.held_dwell.advance(dt,qualified,poses)
            self.ready = bool(qualified)

    def score(self):
        s = self.coating_score()
        if s == .5 and self.joint['seated']:
            return 1. if (self.held_s if self.completion == "held" else self.release_s) >= 1.-1e-9 else .75
        return s

    def metrics(self):
        out = super().metrics()
        out.update(beads={n:bead_status(f) for n,f in self.films.items()},joint=self.joint,
                   release_hold_s=self.release_s,held_corner_hold_s=self.held_s,held_corner=self.held_corner,
                   completion=self.completion,press_s=self.press_s,bonded=self.bonded,
                   hand_wood_clearance_mm=self.hand_wood_clearance_mm,
                   accelerated_bond_required=self.require_bond)
        return out
