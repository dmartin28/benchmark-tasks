"""End-score contract for clearing combustibles before a cabinet-light solder repair."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Outcome:
    unsafe_heat: bool=False
    damaged: bool=False
    cardboard_clear: bool=False
    cardboard_supported: bool=False
    mat_unobstructed: bool=False
    mat_under_joint: bool=False
    joint_repaired: bool=False
    iron_off: bool=True
    objects_released: bool=False
    iron_parked: bool=False
    light_supported: bool=False
    workpiece_supported: bool=False
    hands_clear_mm: float=0.
    stable_seconds: float=0.


def score(o: Outcome) -> float:
    """Endpoint contract connected to the native heat/contact surrogate.

    Repair is an observed solder bridge between the red conductor and its pad,
    with neither a gap nor a bridge to the black-return pad. A cooled physical
    joint must also survive a 1 N pull for one second; thermal reflow alone is
    not a real-world electrical or mechanical joint-quality measurement.
    """
    if not np.isfinite([o.hands_clear_mm,o.stable_seconds]).all():return 0.
    if min(o.hands_clear_mm,o.stable_seconds)<0:return 0.
    if o.unsafe_heat or o.damaged:return 0.
    if not o.iron_off or not o.objects_released or not o.workpiece_supported or o.stable_seconds<1.:return 0.
    if not o.cardboard_clear or not o.cardboard_supported or not o.mat_unobstructed:return 0.
    if not o.mat_under_joint:return .25
    if not o.joint_repaired:return .5
    if not o.iron_parked or not o.light_supported or o.hands_clear_mm<50.:return .75
    return 1.


def _footprint(points):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]<2 or len(p)<3 or not np.isfinite(p).all():return None
    ordered=sorted(set(map(tuple,p[:,:2])))
    if len(ordered)<3:return None
    lo=[]
    for v in ordered:
        while len(lo)>=2 and _cross(lo[-2],lo[-1],v)<=0:lo.pop()
        lo.append(v)
    hi=[]
    for v in reversed(ordered):
        while len(hi)>=2 and _cross(hi[-2],hi[-1],v)<=0:hi.pop()
        hi.append(v)
    hull=np.array(lo[:-1]+hi[:-1])
    return hull if len(hull)>=3 else None


def _cross(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def cardboard_distance_xy(points,joint):
    """Minimum plan-view distance to the full convex footprint, including edges."""
    hull=_footprint(points)
    q=np.asarray(joint,float)[:2]
    if hull is None or not np.isfinite(q).all():return 0.
    if all(_cross(a,b,q)>=-1e-12 for a,b in zip(hull,np.roll(hull,-1,axis=0))):return 0.
    distances=[]
    for a,b in zip(hull,np.roll(hull,-1,axis=0)):
        d=b-a
        t=np.clip((q-a)@d/(d@d),0.,1.)
        distances.append(np.linalg.norm(q-(a+t*d)))
    return float(min(distances))


def mat_is_unobstructed(center,rotation,size,carton_vertices):
    """Packaging must be at least 10 mm outside the mat's footprint.

    A carton entirely below an elevated mat does not obstruct its upper surface.
    Separating-axis detection also catches crossed rectangles with no corners inside.
    """
    c,R,s,p=np.asarray(center,float),np.asarray(rotation,float),np.asarray(size,float),np.asarray(carton_vertices,float)
    if c.shape!=(3,) or R.shape!=(3,3) or s.shape!=(3,) or p.ndim!=2 or p.shape[1]!=3:return False
    if not all(np.isfinite(a).all() for a in (c,R,s,p)) or len(p)<3:return False
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*s/2
    world=corners@R.T+c
    if p[:,2].max()<world[:,2].min()-.005:return True
    mat,box=_footprint(world),_footprint(p)
    if mat is None or box is None:return False
    separated=False
    for hull in (mat,box):
        for a,b in zip(hull,np.roll(hull,-1,axis=0)):
            d=b-a;normal=np.array([-d[1],d[0]])
            ma,bo=mat@normal,box@normal
            if ma.max()<bo.min() or bo.max()<ma.min():separated=True
    if not separated:return False
    gap=min([cardboard_distance_xy(box,v) for v in mat]+
            [cardboard_distance_xy(mat,v) for v in box])
    return gap>=.010-1e-9


def mat_supports_joint(center,rotation,size,joint,surface_z):
    """A flat mat directly on the wooden work surface must cover a 40 mm square
    beneath the measured joint, with another 5 mm of felt beyond each edge.
    A mat resting on cardboard, hanging vertically, or lying elsewhere fails.
    """
    c,R,s,j=np.asarray(center,float),np.asarray(rotation,float),np.asarray(size,float),np.asarray(joint,float)
    if c.shape!=(3,) or R.shape!=(3,3) or s.shape!=(3,) or j.shape!=(3,):return False
    if not all(np.isfinite(a).all() for a in (c,R,s,j)) or not np.isfinite(surface_z):return False
    if min(s)<=0 or not np.allclose(R.T@R,np.eye(3),atol=1e-5) or abs(np.linalg.det(R)-1)>1e-5:return False
    if R[2,2]<math.cos(math.radians(10)):return False
    bottom=c[2]-s[2]/2
    if abs(bottom-surface_z)>.002:return False
    if not c[2]+s[2]/2-.001<=j[2]<=c[2]+s[2]/2+.030:return False
    normal=R[:,2]
    for dx in (-.020,.020):
        for dy in (-.020,.020):
            target=j+np.array([dx,dy,0.])
            target[2]=c[2]-(normal[0]*(target[0]-c[0])+normal[1]*(target[1]-c[1]))/normal[2]
            local=R.T@(target-c)
            if np.any(np.abs(local[:2])>s[:2]/2-.005):return False
    return True
