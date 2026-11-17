"""Primary frame rubric: visible bead, bottle weight loss, gauges and a held ending.

The fluid grid remains a material approximation. No internal wet-contact length,
per-face microlitre dose, prescribed TCP, or chemical bond determines this score.
"""
from dataclasses import replace
import math
import numpy as np

from .assembly import AssemblyModel, joint_geometry
from .scene import HEAD, LEG, TOP, THICK, WIDTH, BOTTLE_BASE, BOTTLE_RADII, NOZZLE_DIAMETER, GLUE_DENSITY_G_ML


def bead_status(film):
    wet = (film.volume > 1e-16) & film.mask  # Same footprint drawn by FilmRenderer.
    widths = wet.sum(axis=1)*film.pitch[1]
    columns = (widths >= .002-1e-9) & (widths <= .008+1e-9)
    occupied = np.flatnonzero(columns)
    longest = current = 0
    if len(occupied):
        for value in columns[occupied[0]:occupied[-1]+1]:
            current = 0 if value else current+1
            longest = max(longest,current)
    length = float(columns.sum()*film.pitch[0])
    gap = float(longest*film.pitch[0])
    peak = float(film.thickness.max())
    return dict(qualified=bool(length >= .065-1e-9 and gap <= .005+1e-9 and
                widths.max() <= .008+1e-9 and peak <= .002+1e-9),
                wet_length_mm=length*1000,visible_length_mm=float(wet.any(axis=1).sum()*film.pitch[0]*1000),
                largest_gap_mm=gap*1000,max_width_mm=float(widths.max()*1000),max_mm=peak*1000,
                volume_ml=float(film.volume.sum()*1e6))  # Volume is a diagnostic, not a per-face criterion.


class CornerHold:
    """A ruler-visible 2 mm bound on each rail's corners for two seconds."""
    def __init__(self):
        self.anchor = None
        self.time = 0.

    def advance(self,dt,qualified,poses):
        if not qualified:
            self.anchor,self.time = None,0.
            return self.time
        points = {s.name:poses[s.name].world(s.mesh()[0][:4]) for s in (HEAD,LEG)}
        if self.anchor is None or any(np.linalg.norm(points[n]-self.anchor[n],axis=1).max() > .002+1e-9 for n in points):
            self.anchor,self.time = points,0.
        self.time += dt
        return self.time


def joint_status(poses):
    result = joint_geometry(poses)
    # Match a top-view ruler reading: lateral mismatch along the seam is
    # separate from the front-face height step, which has its own gauge test.
    cut = next(s for s in HEAD.mesh()[2] if s.target)
    other = next(s for s in LEG.mesh()[2] if s.target)
    a,b = poses[HEAD.name].world(cut.corners),poses[LEG.name].world(other.corners)
    delta = a-b[np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2).argmin(axis=1)]
    result['tangent_error_mm'] = float(np.abs(delta@(poses[HEAD.name].rot@cut.u)).max()*1000)
    # Go/no-go physical checks: a 2 mm leaf for coarse seating; a 0.5 mm
    # leaf at both seam ends and under a bridging straightedge for full credit.
    g,t,f,b,a = (result[k] for k in ('gap_mm','tangent_error_mm','flush_mm','bottom_error_mm','angle_deg'))
    result['seated'] = bool(abs(a-90)<=3 and g<=2 and t<=2 and f<=2 and b<1)
    result['precision_seated'] = bool(abs(a-90)<=2 and g<.5 and t<=1 and f<.5 and b<1)
    return result


def _directions(vertices):
    from scipy.spatial import ConvexHull
    triangles = ConvexHull(vertices).simplices
    d = (vertices[triangles[:,[1,2,0]]]-vertices[triangles]).reshape(-1,3)
    d = d[np.linalg.norm(d,axis=1)>1e-10]
    d /= np.linalg.norm(d,axis=1,keepdims=True)
    return np.unique(np.round(d,6),axis=0)


def finger_near_wood(hull,vertices,normals):
    """Full convex separating-axis contact envelope, 2 mm for the visible pad skin."""
    a = hull.vertices
    box_gap = np.maximum(np.maximum(vertices.min(0)-a.max(0),a.min(0)-vertices.max(0)),0.)
    if np.linalg.norm(box_gap) > .002:
        return False
    faces = np.r_[hull.planes[:,:3],normals]
    faces /= np.linalg.norm(faces,axis=1,keepdims=True)
    pa,pb = a@faces.T,vertices@faces.T
    if np.maximum(pb.min(0)-pa.max(0),pa.min(0)-pb.max(0)).max()>.002+1e-9:
        return False
    cross = np.cross(_directions(a)[:,None,:],_directions(vertices)[None,:,:]).reshape(-1,3)
    axes = np.r_[hull.planes[:,:3],normals,cross]
    axes = axes[np.linalg.norm(axes,axis=1)>1e-8]
    axes /= np.linalg.norm(axes,axis=1,keepdims=True)
    pa,pb = a@axes.T,vertices@axes.T
    gap = np.maximum(pb.min(0)-pa.max(0),pa.min(0)-pb.max(0)).max()
    return bool(gap <= .002+1e-9)


def hand_status(poses,hulls,gaps):
    """Either hand assignment and any grasp; both opposing finger pads must touch."""
    from .clearance import wood_geometry
    candidates = {}
    for side in ('left','right'):
        fingers = {name:[h for link,h in hulls[side] if link==name] for name in ('tip_left','tip_right')}
        candidates[side] = {}
        for spec in (HEAD,LEG):
            local,normals,_ = wood_geometry(spec.name)
            vertices = poses[spec.name].world(local)
            contacts = {name:any(finger_near_wood(h,vertices,normals@poses[spec.name].rot.T) for h in parts)
                        for name,parts in fingers.items()}
            holding = len(contacts)==2 and all(contacts.values()) and gaps[side]<.09
            candidates[side][spec.name] = dict(piece=spec.name,holding=bool(holding),finger_contacts=contacts)
    assignments = [(LEG.name,HEAD.name),(HEAD.name,LEG.name)]
    pair = max(assignments,key=lambda p:sum(candidates[s][n]['holding'] for s,n in zip(('left','right'),p)))
    return {s:candidates[s][n] for s,n in zip(('left','right'),pair)}


def bottle_parked(poses,states,hands):
    from .runtime import aabb_distance
    p = poses['glue_bottle']
    extent = (states['glue_bottle'].low,states['glue_bottle'].high)
    # The whole oval base must fit in its 100 mm home square. A 10 degree
    # lean tolerance is checked with a protractor in the physical protocol.
    radii = np.sqrt(np.sum((p.rot[:,:2]*BOTTLE_RADII)**2,axis=1))
    return bool(np.max(np.abs(p.pos[:2]-BOTTLE_BASE[:2])+radii[:2])<=.05 and
                abs(p.pos[2]-TOP)+radii[2]<.002 and p.rot[2,2]>=math.cos(math.radians(10)) and
                all(aabb_distance(h,extent)>=.002 for h in hands))


def evaluate_hold(emb,poses,states,model):
    from .clearance import measured_joints
    from yam_arena.analysis import world_hulls,fk_link_poses_world
    from yam_arena import station
    from yam_arena.constants import FINGER_JOINTS
    joints = measured_joints(emb)
    hulls = {s:world_hulls(fk_link_poses_world(q,station.BASES[s]),['gripper','tip_left','tip_right'])
             for s,q in joints.items()}
    gaps = {s:-sum(joints[s][n] for n in FINGER_JOINTS) for s in joints}
    hands = hand_status(poses,hulls,gaps)
    joint = joint_status(poses)
    bounds=[]
    for parts in hulls.values():
        v=np.concatenate([h.vertices for _,h in parts])
        bounds.append((v.min(0),v.max(0)))
    parked = bottle_parked(poses,states,bounds)
    qualified = all(h['holding'] for h in hands.values()) and joint['precision_seated'] and parked
    return dict(qualified=bool(qualified),hands=hands,joint=joint,bottle_parked=parked,bonded=False)


class ObservableAssemblyModel(AssemblyModel):
    def __init__(self,surfaces,require_bond=False,completion='held'):
        super().__init__(surfaces,require_bond=False,completion='held')
        self.settings = replace(self.settings,fatal_contamination=False,
                                tip_width=NOZZLE_DIAMETER,tip_depth=NOZZLE_DIAMETER,
                                wetting_radius=.0015)

    def reset(self):
        super().reset()
        self.held_dwell = CornerHold()
        self.escaped_glue = False
        self.cleanliness_mm2 = 0.

    def invalidate(self,reason):
        if reason == 'glue_left_work_area':
            self.escaped_glue = True
            return
        super().invalidate(reason)

    @property
    def dose_g(self):
        return self.emitted*1e6*GLUE_DENSITY_G_ML

    @property
    def scale_dose_g(self):
        return math.floor(self.dose_g*100+.5)/100

    def coating_score(self):
        if self.failed:
            return 0.
        status = [bead_status(f) for f in self.films.values()]
        if any(s['qualified'] for s in status) and .05<=self.scale_dose_g<=.30:
            return .5
        return .25 if any(s['visible_length_mm']>=20-1e-9 for s in status) else 0.

    def stray_area(self):
        area = 0.
        for key,(owner,point) in self.marks.items():
            allowed = False
            if self.joint.get('seated') and key[0].endswith('/front'):
                film = self.films.get(owner+'/miter')
                if film is not None:
                    cut = film.surface
                    distance = abs(float((point-cut.origin)@cut.normal))
                    along = float((point-cut.origin)@cut.u)
                    allowed = distance<=.002 and 0<=along<=np.linalg.norm(cut.corners[1]-cut.origin)
            if not allowed:
                area += self.settings.cell**2*1e6
        return area

    def joint_step(self,tick,dt,poses,released,parked,bonded=False,held=False):
        if tick == self.last_joint_tick:
            return
        self.last_joint_tick = tick
        self.joint = joint_status(poses)
        self.cleanliness_mm2 = self.stray_area()
        qualified = (self.coating_score()==.5 and self.joint['precision_seated'] and held and parked and
                     self.cleanliness_mm2<=25+1e-9 and not self.escaped_glue)
        self.held_s = self.held_dwell.advance(dt,qualified,poses)
        self.ready = bool(qualified)

    def score(self):
        score = self.coating_score()
        if score==.5 and self.joint.get('seated'):
            score = 1. if self.held_s>=2.-1e-9 else .75
        if self.cleanliness_mm2>25+1e-9 or self.escaped_glue:
            score = min(score,.5)
        return score

    def metrics(self):
        out = super().metrics()
        out.update(beads={n:bead_status(f) for n,f in self.films.items()},rubric='observable-frame-v1',
                   dispensed_mass_g=self.dose_g,scale_dose_g=self.scale_dose_g,stray_glue_mm2=self.cleanliness_mm2,
                   escaped_glue=self.escaped_glue,hold_required_s=2.,accelerated_bond_required=False)
        return out
