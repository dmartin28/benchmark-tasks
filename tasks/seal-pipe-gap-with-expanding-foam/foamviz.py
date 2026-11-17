"""Draws the foam so the cameras see it: one pale-yellow blob per ring sector at the gap's mouth, growing as the foam
in that sector rises and spreads (a full sector's blob spans the gap), pushed out of the face where the foam bulges,
and blobs where foam landed outside the gap. Isaac Lab VisualizationMarkers, imported only once
the simulator runs (as the accepted painting task does). Cosmetic: any error turns the drawing off with one warning;
the foam's bookkeeping (foam.Ledger) is unaffected.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

import foam
import geometry as g

FOAM_RGB = (0.93, 0.86, 0.55)       # cured gap-filler foam: pale yellow
CELL_R = 0.0065                      # a full sector's blob radius (m): about the gap's half-width (7 mm)
MAX_MESS = 200


def _linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


class FoamViz:
    def __init__(self) -> None:
        self._markers = None
        self._off = False
        self._shown = -1

    def hide(self) -> None:
        if self._markers is not None and not self._off:
            try:
                self._markers.set_visibility(False)
                self._shown = 0
            except Exception as error:  # cosmetic only
                self._disable(error)

    def draw(self, environment, ledger: foam.Ledger) -> None:
        if self._off:
            return
        try:
            held, bulge = foam.spread(ledger.landed())
            fill = held / foam.SECTOR_VOLUME
            proud = bulge / (g.BULGE_FOOTPRINT * foam.SECTOR_FACE)
            sectors = np.nonzero(fill > 1e-3)[0]
            points, scales = [], []
            rm = (g.RING_IN + g.RING_OUT) / 2
            for s in sectors:
                a = (s + 0.5) * 2 * math.pi / g.SECTORS
                x = g.FACE_X + CELL_R - min(float(proud[s]), 0.03)           # flush with the face when full
                points.append((x, g.HOLE_CENTRE[1] + rm * math.cos(a), g.HOLE_CENTRE[2] + rm * math.sin(a)))
                k = float(fill[s]) ** (1 / 3)
                scales.append((k, k, k))
            blobs: dict[tuple, float] = {}
            for where, v, t0 in ledger.mess:
                key = tuple(np.round(where / 0.005).astype(int))
                blobs[key] = blobs.get(key, 0.0) + foam.risen(v, ledger.t - t0)
            for key, vol in list(blobs.items())[:MAX_MESS]:
                r = (3 * vol / (4 * math.pi)) ** (1 / 3)
                points.append(tuple(np.asarray(key) * 0.005))
                k = r / CELL_R
                scales.append((k, k, k))
            if len(points) == self._shown == 0:
                return
            if self._markers is None:
                self._markers = self._make()
            self._shown = len(points)
            if not points:
                self._markers.set_visibility(False)
                return
            origin = environment.unwrapped.scene.env_origins[0].detach().cpu().numpy()
            pts = np.asarray(points, float) + origin
            self._markers.set_visibility(True)
            self._markers.visualize(translations=pts.astype(np.float32),
                                    orientations=np.tile([0.0, 0.0, 0.0, 1.0], (len(pts), 1)).astype(np.float32),
                                    scales=np.asarray(scales, np.float32))
        except ImportError:
            self._off = True                      # no simulator (unit tests)
        except Exception as error:                # cosmetic only: never abort a trial over the picture
            self._disable(error)

    def _disable(self, error: Exception) -> None:
        self._off = True
        warnings.warn(f"foam drawing turned off ({type(error).__name__}: {error}); the foam ledger is unaffected")

    @staticmethod
    def _make():
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        blob = sim_utils.SphereCfg(radius=CELL_R, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(_linear(c) for c in FOAM_RGB), roughness=0.8))
        return VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/foam", markers={"foam": blob}))
