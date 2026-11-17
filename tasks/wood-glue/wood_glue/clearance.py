"""Conservative whole-arm separation certificates from the URDF convex collision hulls.

Projection gaps are lower bounds on Euclidean clearance. A missing separating axis
can only underestimate clearance, never certify an intersecting pair as separated.
Values are capped at 50 mm, so a reported 50 mm means 'at least 50 mm'.
"""
from functools import lru_cache
import numpy as np
from yam_arena import station
from yam_arena.analysis import world_hulls, fk_link_poses_world
from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS


def projection_clearance(a,b):
    gap = np.maximum(b.min(0)-a.max(0),a.min(0)-b.max(0))
    return float(np.linalg.norm(np.maximum(gap,0.)))


def robot_hulls(joints):
    return {side:world_hulls(fk_link_poses_world(q,station.BASES[side])) for side,q in joints.items()}


def arm_clearance(joints,hulls=None):
    hulls = robot_hulls(joints) if hulls is None else hulls
    best,pair = .050,None
    left,right = hulls['left'],hulls['right']
    alo,ahi = np.array([h.vertices.min(0) for _,h in left]),np.array([h.vertices.max(0) for _,h in left])
    blo,bhi = np.array([h.vertices.min(0) for _,h in right]),np.array([h.vertices.max(0) for _,h in right])
    gaps = np.maximum(blo[None]-ahi[:,None],alo[:,None]-bhi[None])
    bounds = np.linalg.norm(np.maximum(gaps,0.),axis=2)
    for i,j in np.argwhere(bounds < best):
        lower = bounds[i,j]
        if lower >= best: continue
        na,a = left[i]
        nb,b = right[j]
        axes = np.unique(np.round(np.r_[a.planes[:,:3],b.planes[:,:3]],7),axis=0)
        # Rounding must not overstate a projection gap: renormalize the axes.
        axes /= np.linalg.norm(axes,axis=1,keepdims=True)
        pa,pb = a.vertices@axes.T,b.vertices@axes.T
        lower = max(lower,float(np.maximum(pb.min(0)-pa.max(0),pa.min(0)-pb.max(0)).max()))
        if lower < best: best,pair = max(0.,lower),(na,nb)
    return best*1000,pair


def commanded_joints(controller):
    out = {}
    for side in ('left','right'):
        arm = getattr(controller,side)
        q = dict(zip(ARM_JOINTS,arm.q))
        q.update(dict.fromkeys(FINGER_JOINTS,-.0939*arm.grip/2))
        out[side] = q
    return out


def measured_joints(emb):
    from .runtime import arr
    return {side:dict(zip(emb._base.scene[side+'_robot'].joint_names,
                          arr(emb._base.scene[side+'_robot'].data.joint_pos)[0].astype(float)))
            for side in ('left','right')}


@lru_cache(maxsize=2)
def wood_geometry(name):
    from scipy.spatial import ConvexHull
    from .scene import HEAD,LEG
    spec = next(s for s in (HEAD,LEG) if s.name == name)
    vertices = spec.mesh()[0]
    planes = ConvexHull(vertices).equations
    return vertices,planes[:,:3],-planes[:,3]


def wood_clearance(joints,poses,contacts=None,hulls=None):
    """Robot-to-wood separation; only named grasping fingertips may touch their own piece.

    Face projection gaps certify a conservative lower bound. Failure to certify separation is
    reported as zero, so edge configurations can never hide an actual collision.
    """
    from .scene import HEAD,LEG
    hulls = robot_hulls(joints) if hulls is None else hulls
    contacts = {} if contacts is None else contacts
    best,pair = .050,None
    for spec in (HEAD,LEG):
        pose = poses[spec.name]
        local,normals,_ = wood_geometry(spec.name)
        vertices = pose.world(local)
        normals = normals@pose.rot.T
        lo,hi = vertices.min(0),vertices.max(0)
        for side,parts in hulls.items():
            lows = np.array([h.vertices.min(0) for _,h in parts])
            highs = np.array([h.vertices.max(0) for _,h in parts])
            bounds = np.linalg.norm(np.maximum(np.maximum(lo-highs,lows-hi),0.),axis=1)
            for i in np.flatnonzero(bounds < best):
                link,hull = parts[i]
                if contacts.get(side) == spec.name and link in ('tip_left','tip_right'):
                    continue
                axes = np.r_[hull.planes[:,:3],normals]
                a,b = hull.vertices@axes.T,vertices@axes.T
                lower = max(float(bounds[i]),float(np.maximum(b.min(0)-a.max(0),a.min(0)-b.max(0)).max()))
                if lower < best:
                    best,pair = max(0.,lower),(side,link,spec.name)
    return best*1000,pair
