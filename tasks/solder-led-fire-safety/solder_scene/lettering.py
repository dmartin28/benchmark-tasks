"""Vector lettering for product markings; no downloaded photos or bitmap edits.

DejaVu Sans outlines are triangulated into small, non-colliding native meshes.
"""
from functools import lru_cache
import os
from pathlib import Path
import numpy as np
from . import mechkit as m


@lru_cache(maxsize=64)
def _outline(text):
    cache=Path(os.environ.get('ROBOCURVE_HOME','/workspace/davidrm3/robocurve'))/'playground/solder-led-fire-safety/matplotlib'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    from shapely.geometry import Polygon,GeometryCollection
    from shapely import constrained_delaunay_triangles
    path=TextPath((0,0),text,size=1.,prop=FontProperties(family='DejaVu Sans',weight='bold'))
    region=GeometryCollection()
    for ring in path.to_polygons():
        if len(ring)>2:
            poly=Polygon(ring)
            if not poly.is_valid:poly=poly.buffer(0)
            region=region.symmetric_difference(poly)
    # Constrain triangle edges to glyph boundaries, including counters in A/B/O.
    # Unconstrained triangulation discarded boundary-crossing faces and lost strokes.
    triangles=list(constrained_delaunay_triangles(region).geoms)
    verts=np.vstack([np.array(t.exterior.coords)[:3] for t in triangles])
    faces=np.arange(len(verts)).reshape(-1,3)
    low,high=verts.min(0),verts.max(0)
    return verts-(low+high)/2,faces,high-low


def text_mesh(text,width,height,center,angle_deg=-90.):
    """Fit text uniformly to a width/height box, centered in the local xy plane."""
    v,f,extent=_outline(text)
    scale=min(width/extent[0],height/extent[1])
    mesh=m.Mesh(np.c_[v*scale,np.zeros(len(v))],f)
    return mesh.turned(angle_deg,'z').moved(center)
