"""Conserved-volume surface glue approximation. No simulator or scoring side effects."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import first_hit, first_hits


@dataclass(frozen=True)
class Settings:
    cell: float = .001
    min_film: float = .00010
    max_film: float = .00025
    puddle: float = .00050
    contamination_area: float = 1e-6
    reservoir_ml: float = 20.
    squeeze_deadband: float = .004
    squeeze_max: float = .012
    ml_per_m: float = 100.
    tip_width: float = .008
    tip_depth: float = .004
    contact_gap: float = .0015
    hold_s: float = 1.
    fatal_contamination: bool = True
    wetting_radius: float = 0.  # Optional local wetting footprint after a round jet lands.


class Film:
    def __init__(self, surface, settings):
        self.surface = surface
        uv = surface.uv(surface.corners)
        lo, hi = uv.min(0), uv.max(0)
        self.shape = np.maximum(1, np.ceil((hi-lo)/settings.cell-1e-10).astype(int))
        self.pitch = (hi-lo)/self.shape
        self.area = float(np.prod(self.pitch))
        self.lo = lo
        u, v = np.meshgrid(*(lo[i]+(np.arange(self.shape[i])+.5)*self.pitch[i] for i in range(2)), indexing="ij")
        self.uv = np.stack((u,v), axis=-1)
        self.points = surface.origin + u[...,None]*surface.u + v[...,None]*surface.v
        self.mask = surface.contains(self.points)
        self.volume = np.zeros(tuple(self.shape))

    @property
    def thickness(self):
        return self.volume/self.area

    def deposit(self, local, amount):
        ij = np.floor((self.surface.uv(local)-self.lo)/self.pitch).astype(int)
        ij = np.minimum(np.maximum(ij, 0), self.shape-1)
        if not self.mask[tuple(ij)]:
            raise ValueError("deposit outside film polygon")
        self.volume[tuple(ij)] += amount

    def coverage(self, cfg):
        h = self.thickness
        valid = (h >= cfg.min_film-1e-12) & (h <= cfg.max_film+1e-12)
        return float(valid[self.mask].mean())

    def spread(self, local, u_local, v_local, width, depth):
        """Equalize a contacted footprint; all redistributed volume is preserved."""
        d = self.points-local
        hit = self.mask & (np.abs(d @ u_local) <= width/2+1e-10) & (np.abs(d @ v_local) <= depth/2+1e-10)
        if hit.any():
            self.volume[hit] = self.volume[hit].sum()/hit.sum()


class GlueModel:
    def __init__(self, surfaces, settings=None):
        self.settings = settings or Settings()
        self.surfaces = tuple(surfaces)
        self.films = {s.name: Film(s, self.settings) for s in surfaces if s.target}
        self.reset()

    def reset(self):
        for film in self.films.values():
            film.volume.fill(0.)
        self.remaining = self.settings.reservoir_ml*1e-6
        self.emitted = 0.
        self.contamination_volume = 0.
        self.contamination = {}  # (surface, cell_i, cell_j) -> volume in m^3
        self.marks = {}  # same keys -> (owner, local landing point)
        self.drops = []  # [position, velocity, volume]
        self.failed = False
        self.previous = None
        self.last_tick = None
        self.time = 0.
        self.held = 0.
        self.ready = False
        self.violations = set()

    def _land(self, surface, point, amount):
        if surface.target:
            film = self.films[surface.name]
            if self.settings.wetting_radius:
                # Isotropic local spreading on the actual hit surface, independent of
                # desired path/score. Conserves volume; no widening of airborne jets.
                wet = film.mask & (np.linalg.norm(film.points-point,axis=-1)<=self.settings.wetting_radius)
                if wet.any():
                    film.volume[wet] += amount/wet.sum()
                    return
            film.deposit(point, amount)
            return
        uv = surface.uv(point)
        ij = np.floor(uv/self.settings.cell).astype(int)
        key = (surface.name, int(ij[0]), int(ij[1]))
        self.contamination[key] = self.contamination.get(key, 0.)+amount
        self.marks[key] = (surface.owner, np.asarray(point).copy())
        self.contamination_volume += amount
        # Count wet area, not thickness: arbitrarily thin contamination is still contamination.
        if self.settings.fatal_contamination and len(self.contamination)*self.settings.cell**2 >= self.settings.contamination_area-1e-12:
            self.failed = True
            self.violations.add("off_target_glue")

    def invalidate(self, reason):
        self.failed = True
        self.violations.add(reason)

    def _emit(self, tip, rot, amount, poses):
        if amount <= 0:
            return
        cfg = self.settings
        # Uniform flow over a finite spreader lip; every ray independently lands or remains in flight.
        nu, nv = max(2, math.ceil(cfg.tip_width/cfg.cell)), max(2, math.ceil(cfg.tip_depth/cfg.cell))
        u,v = np.meshgrid((np.arange(nu)+.5)/nu-.5,(np.arange(nv)+.5)/nv-.5,indexing="ij")
        p = tip + u.ravel()[:,None]*rot[:,0]*cfg.tip_width + v.ravel()[:,None]*rot[:,1]*cfg.tip_depth
        indices,points = first_hits(p-rot[:,2]*cfg.contact_gap,p+rot[:,2]*cfg.contact_gap,self.surfaces,poses)
        dv = amount/(nu*nv)
        for j,idx in enumerate(indices):
            if idx < 0:
                self.drops.append([p[j].copy(),.05*rot[:,2],dv])
            else:
                self._land(self.surfaces[idx],points[j],dv)

    def _spread(self, tip, rot, poses):
        cfg = self.settings
        if cfg.wetting_radius:
            return  # Round nozzle has no spreading blade.
        hit = first_hit(tip-rot[:,2]*cfg.contact_gap, tip+rot[:,2]*cfg.contact_gap, self.surfaces, poses)
        if hit is None or not hit[1].target:
            return
        surface, local = hit[1], hit[2]
        pose = poses[surface.owner]
        # A spreader must face the wood; a distant or edge-on tool cannot spread it.
        if -(pose.rot @ surface.normal) @ rot[:,2] < .65:
            return
        self.films[surface.name].spread(local, pose.rot.T @ rot[:,0], pose.rot.T @ rot[:,1], cfg.tip_width, cfg.tip_depth)

    def _fall(self, dt, poses):
        kept = []
        for p, vel, amount in self.drops:
            end = p+vel*dt+[0.,0.,-4.905*dt*dt]
            hit = first_hit(p, end, self.surfaces, poses)
            if hit is None:
                if end[2] < -.1:
                    # Escaping liquid remains in the conservation ledger and invalidates the run.
                    self.invalidate("glue_left_work_area")
                kept.append([end, vel+[0.,0.,-9.81*dt], amount])
            else:
                self._land(hit[1], hit[2], amount)
        self.drops = kept

    def advance(self, tick, dt, tip, rot, compression, poses, ready=False):
        """Advance once per measured control interval; scores never call this method.

        Releasing the soft bottle restores its shape without aspirating glue in this approximation.
        Only *additional* inward travel dispenses, so holding full squeeze cannot create infinite glue.
        """
        if tick == self.last_tick:
            return
        if self.last_tick is not None and tick < self.last_tick:
            raise ValueError("ticks must increase; call reset between trials")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        tip, rot = np.asarray(tip,float), np.asarray(rot,float)
        if not np.isfinite(tip).all() or not np.isfinite(rot).all() or not math.isfinite(compression):
            raise ValueError("nonfinite measured state")
        c = float(np.clip(compression, 0., self.settings.squeeze_max))
        previous = self.previous
        self.last_tick = tick
        if previous is None:
            previous = (tip.copy(), rot.copy(), c)
        p0, r0, c0 = previous
        active = max(0., c-self.settings.squeeze_deadband)
        active0 = max(0., c0-self.settings.squeeze_deadband)
        amount = min(self.remaining, max(0., active-active0)*self.settings.ml_per_m*1e-6)
        self.remaining -= amount
        self.emitted += amount
        # Spatial subdivision prevents skipped grid cells; temporal subdivision resolves falling drops.
        angular = np.arccos(np.clip((np.trace(r0.T@rot)-1)/2, -1, 1))
        travel = np.linalg.norm(tip-p0)+angular*self.settings.tip_width
        n = max(1, math.ceil(travel/(self.settings.cell/2)), math.ceil(dt/(1/240)))
        for i in range(n):
            s = (i+.5)/n
            p = p0+s*(tip-p0)
            # Orthonormal interpolation without a simulator dependency.
            u, _, vh = np.linalg.svd((1-s)*r0+s*rot)
            r = u @ np.diag([1.,1.,np.linalg.det(u@vh)]) @ vh
            self._emit(p, r, amount/n, poses)
            self._spread(p, r, poses)
            self._fall(dt/n, poses)
        self.previous = (tip.copy(), rot.copy(), c)
        self.time += dt
        self.ready = bool(ready)
        self.held = self.held+dt if ready and self.coating_score() == .75 else 0.

    def coating_score(self):
        if self.failed:
            return 0.
        qualified = []
        for film in self.films.values():
            qualified.append(0. if film.thickness.max() > self.settings.puddle+1e-12 else film.coverage(self.settings))
        if len(qualified) != 2:
            raise RuntimeError("expected exactly two miter faces")
        if min(qualified) >= .95-1e-12:
            return .75
        if min(qualified) >= .8-1e-12:
            return .5
        return .25 if max(qualified) >= .8-1e-12 else 0.

    def score(self):
        s = self.coating_score()
        return 1. if s == .75 and self.held >= self.settings.hold_s-1e-9 else s

    def metrics(self):
        deposited = sum(float(f.volume.sum()) for f in self.films.values())
        flying = sum(d[2] for d in self.drops)
        return dict(score=self.score(), success=self.score() == 1., time_s=self.time,
                    ready=self.ready, release_hold_s=self.held, violations=sorted(self.violations),
                    emitted_ml=self.emitted*1e6, remaining_ml=self.remaining*1e6,
                    target_ml=deposited*1e6, contaminated_ml=self.contamination_volume*1e6,
                    airborne_ml=flying*1e6, contamination_mm2=len(self.contamination)*self.settings.cell**2*1e6,
                    conservation_error_ml=(self.emitted-deposited-self.contamination_volume-flying)*1e6,
                    faces={name:dict(coverage=f.coverage(self.settings), max_mm=float(f.thickness.max()*1000),
                                     volume_ml=float(f.volume.sum()*1e6)) for name,f in self.films.items()})
