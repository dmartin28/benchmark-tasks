"""Metre-based rigid geometry shared by the scorer, renderer and scene builder."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import math

import numpy as np


def rotation(quat):
    q = np.asarray(quat, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("expected a finite nonzero XYZW quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


@dataclass
class Pose:
    pos: np.ndarray
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    linear: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float)
        self.rot = np.asarray(self.rot, float)

    def world(self, points):
        return np.asarray(points) @ self.rot.T + self.pos

    def local(self, points):
        return (np.asarray(points) - self.pos) @ self.rot

    def still(self):
        return bool(np.linalg.norm(self.linear) < .01 and np.linalg.norm(self.angular) < .1)


@dataclass(frozen=True)
class Surface:
    name: str
    owner: str
    corners: np.ndarray
    target: bool = False

    @cached_property
    def origin(self):
        return self.corners[0]

    @cached_property
    def u(self):
        d = self.corners[1] - self.origin
        return d / np.linalg.norm(d)

    @cached_property
    def normal(self):
        d = np.cross(self.u, self.corners[-1] - self.origin)
        return d / np.linalg.norm(d)

    @cached_property
    def v(self):
        return np.cross(self.normal, self.u)

    def uv(self, points):
        p = np.asarray(points) - self.origin
        return np.stack((p @ self.u, p @ self.v), axis=-1)

    def contains(self, points, tolerance=1e-9):
        p = self.uv(points)
        outline = self.uv(self.corners)
        e = np.roll(outline, -1, axis=0) - outline
        d = p[..., None, :] - outline
        cross = e[:, 0]*d[..., 1] - e[:, 1]*d[..., 0]
        return np.all(cross >= -tolerance, axis=-1)

    def intersect(self, start, end, pose):
        """Segment hit, ignoring departures from a surface and back-face hits."""
        a, b = pose.local(start), pose.local(end)
        n = self.normal
        da, db = (a-self.origin) @ n, (b-self.origin) @ n
        if da < -1e-6 or db >= da - 1e-12 or db > 1e-9:
            return None
        t = max(0., float(da / (da-db)))
        p = a + t*(b-a)
        return (t, p) if self.contains(p) else None


def prism(name, outline, thickness, target_edge=None):
    """CCW xy outline extruded in z; identical mesh and surface boundaries."""
    xy = np.asarray(outline, float)
    count = len(xy)
    vertices = np.concatenate([np.c_[xy, np.full(count, z)] for z in (0., thickness)])
    sides = []
    for i in range(count):
        k = (i+1) % count
        label = "miter" if i == target_edge else f"edge_{i}"
        sides.append(Surface(f"{name}/{label}", name, vertices[[i, k, k+count, i+count]], i == target_edge))
    sides += [Surface(f"{name}/back", name, vertices[np.arange(count)[::-1]]),
              Surface(f"{name}/front", name, vertices[np.arange(count)+count])]
    faces = []
    for s in sides:
        ids = [int(np.flatnonzero(np.all(vertices == p, axis=1))[0]) for p in s.corners]
        faces.extend((ids[0], ids[i], ids[i+1]) for i in range(1, len(ids)-1))
    return vertices, np.asarray(faces, int), sides


def box_surfaces(name, size, center=(0., 0., 0.)):
    x, y, z = np.asarray(size)/2
    _, _, surfaces = prism(name, [(-x,-y), (x,-y), (x,y), (-x,y)], 2*z)
    shift = np.asarray(center)-[0.,0.,z]
    return [Surface(s.name, name, s.corners+shift) for s in surfaces]


def first_hit(start, end, surfaces, poses):
    best = None
    for surface in surfaces:
        hit = surface.intersect(start, end, poses[surface.owner])
        if hit is not None and (best is None or hit[0] < best[0]):
            best = (hit[0], surface, hit[1])
    return best


def first_hits(starts, ends, surfaces, poses):
    """Vectorized first intersections of independent emission rays."""
    n = len(starts)
    times = np.full(n,np.inf)
    indices = np.full(n,-1,int)
    points = np.zeros((n,3))
    for i,surface in enumerate(surfaces):
        pose = poses[surface.owner]
        a,b = pose.local(starts),pose.local(ends)
        da,db = (a-surface.origin)@surface.normal,(b-surface.origin)@surface.normal
        valid = (da >= -1e-6) & (db < da-1e-12) & (db <= 1e-9)
        if not valid.any(): continue
        t = np.maximum(0.,np.divide(da,da-db,out=np.full(n,np.inf),where=np.abs(da-db)>1e-12))
        candidate = a+t[:,None]*(b-a)
        valid &= (t < times) & surface.contains(candidate)
        times[valid],indices[valid],points[valid] = t[valid],i,candidate[valid]
    return indices,points
