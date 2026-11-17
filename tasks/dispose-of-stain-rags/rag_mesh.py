"""Cotton-rag triangle mesh: a flat square, and the same square crumpled. No simulation.

The `Mesh.audit` integrity checks are adapted from the accepted task
`peg-out-a-tshirt` (cloth_mesh.py, MIT licence, see LICENSE in this folder).
The rag geometry is original. Units are metres.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    rest: np.ndarray
    faces: np.ndarray
    uv: np.ndarray = None      # per-vertex texture coordinates (0..1), from the flat square

    def audit(self):
        p, f = self.rest, self.faces
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError("invalid points")
        if f.ndim != 2 or f.shape[1] != 3 or f.min() < 0 or f.max() >= len(p):
            raise ValueError("invalid topology")
        area = np.linalg.norm(np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]]), axis=1) / 2
        if np.any(area < 1e-12):
            raise ValueError("degenerate triangle")
        edges = Counter(tuple(sorted((int(a), int(b)))) for tri in f for a, b in zip(tri, np.roll(tri, -1)))
        if max(edges.values()) > 2:
            raise ValueError("nonmanifold edge")
        adjacency = defaultdict(set)
        for a, b in edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
        todo, visited = [0], {0}
        while todo:
            for b in adjacency[todo.pop()]:
                if b not in visited:
                    visited.add(b)
                    todo.append(b)
        if len(visited) != len(p):
            raise ValueError("disconnected or unused vertices")
        lens = np.array([np.linalg.norm(p[a] - p[b]) for a, b in edges])
        return dict(vertices=len(p), triangles=len(f), surface_area_m2=float(area.sum()),
                    min_edge_m=float(lens.min()), max_edge_m=float(lens.max()))


def square(side=0.30, spacing=0.010):
    """Flat square sheet in the XY plane, centred on the origin, alternating diagonals."""
    n = int(round(side / spacing)) + 1
    u = np.linspace(-side / 2, side / 2, n)
    xx, yy = np.meshgrid(u, u, indexing="ij")
    rest = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n * n)])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            if (i + j) % 2:
                faces += [(a, b, c), (a, c, d)]
            else:
                faces += [(a, b, d), (b, c, d)]
    return Mesh(rest, np.asarray(faces, int), uv=(rest[:, :2] + side / 2) / side)


def edges_of(faces):
    return np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)


def crumple(flat, seed=0, height=0.065, gather=0.70):
    """The rag as it lies after being wiped and dropped: a rumpled heap of folds and creases.

    A height field over the flat square (so the sheet never passes through itself): five to seven
    creased folds along gently curved lines (sharp crests, 2-5 cm wide, crossing one another), small
    rumples, the outline gathered in to `gather` of its width and made irregular, the edges lower.
    This shape becomes the rag's rest shape, so the simulated cloth keeps its creases, as a
    stain-stiffened cotton rag does, instead of relaxing flat. Returns a Mesh (centred, lowest point at
    z 0) with the flat square's texture coordinates.
    """
    rng = np.random.default_rng(seed)
    p0 = flat.rest
    side = np.ptp(p0[:, 0])
    u = p0[:, :2] / side                     # -0.5 .. 0.5
    z = np.zeros(len(p0))
    for _ in range(rng.integers(5, 8)):     # creased folds
        c = rng.uniform(-0.3, 0.3, 2)
        a = rng.uniform(0, np.pi)
        n = np.array((-np.sin(a), np.cos(a)))
        t = (u - c) @ np.array((np.cos(a), np.sin(a)))
        dist = (u - c) @ n - rng.uniform(-1.0, 1.0) * t ** 2
        w = rng.uniform(0.06, 0.14)
        crest = np.clip(1.0 - np.abs(dist) / w, 0.0, None) ** 1.3
        z = np.maximum(z, rng.uniform(0.35, 1.0) * crest * np.exp(-(t / rng.uniform(0.25, 0.5)) ** 2)) \
            + 0.25 * rng.uniform(0.35, 1.0) * crest
    for _ in range(14):                      # small rumples
        c, w = rng.uniform(-0.45, 0.45, 2), rng.uniform(0.03, 0.07)
        z += rng.uniform(-0.15, 0.25) * np.exp(-np.sum((u - c) ** 2, axis=1) / w ** 2)
    edge = np.clip((0.5 - np.abs(u)).min(axis=1) / 0.10, 0.0, 1.0)  # the hem lies lower
    z = np.clip(z, 0.0, None) * (0.45 + 0.55 * edge)
    # Round the crests over a few centimetres: a fold in cotton is a curve, not a knife edge.
    n = int(round(np.sqrt(len(p0))))
    g = z.reshape(n, n)
    for _ in range(3):
        pad = np.pad(g, 1, mode="edge")
        g = 0.4 * g + 0.15 * (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:])
    z = g.ravel()
    z *= height / z.max()
    ang = np.arctan2(u[:, 1], u[:, 0])
    r = np.clip(np.linalg.norm(u, axis=1) / 0.5, 0, 1)
    wav = 1.0 + r * (0.08 * np.sin(2 * ang + rng.uniform(0, 6.3)) + 0.06 * np.sin(3 * ang + rng.uniform(0, 6.3)))
    xy = p0[:, :2] * gather * wav[:, None]
    p = np.column_stack([xy - xy.mean(0), z - z.min()])
    return Mesh(p, flat.faces.copy(), uv=flat.uv)


def edge_strain(rest, current, faces):
    edges = edges_of(faces)
    r = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    c = np.linalg.norm(current[edges[:, 1]] - current[edges[:, 0]], axis=1)
    return c / r - 1.0

