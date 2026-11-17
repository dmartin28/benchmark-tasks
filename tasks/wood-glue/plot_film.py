#!/usr/bin/env python3
"""Replay measured trace values into a diagnostic film map; does not launch the simulator."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from wood_glue.geometry import Pose
from wood_glue.model import GlueModel
from wood_glue.scene import LEG,TIP_LOCAL,surfaces


def plot(root):
    receipt = json.loads((root/"receipt.json").read_text())
    model = GlueModel(surfaces())
    with (root/"trace.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            poses = {name:Pose(np.array(p["pos"]),np.array(p["rotation"])) for name,p in row["poses"].items()}
            bottle = poses["glue_bottle"]
            compression = -row["sliders"]["panel_left"]+row["sliders"]["panel_right"]
            model.advance(row["step"],1/receipt["control_hz"],bottle.world(TIP_LOCAL),bottle.rot,compression,poses)
    film = model.films[LEG.name+"/miter"]
    metrics = receipt['final_metrics']
    expected = metrics.get('held_edge_glue')
    if expected is None:
        face = metrics['faces'][LEG.name+'/miter']
        expected = dict(film_coverage=face['coverage'],glue_on_held_face_ml=face['volume_ml'])
    if abs(film.coverage(model.settings)-expected["film_coverage"]) > 1e-9:
        raise RuntimeError("replayed coverage differs from the native receipt; use the run's original source")
    if abs(film.volume.sum()*1e6-expected["glue_on_held_face_ml"]) > 1e-8:
        raise RuntimeError("replayed volume differs from the native receipt; use the run's original source")

    os.environ.setdefault("MPLCONFIGDIR","/workspace/davidrm3/robocurve/playground/wood_glue/matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig,ax = plt.subplots(figsize=(12,4.4),layout="constrained")
    fig.set_facecolor("#f6f4ee")
    cmap = LinearSegmentedColormap.from_list("wet_glue",["#ca9a64","#e8d6a0","#fff5cf","#d7ac44"])
    height = film.thickness.T*1000
    extent = [0.,film.pitch[0]*film.shape[0]*1000,0.,film.pitch[1]*film.shape[1]*1000]
    im = ax.imshow(height,origin="lower",extent=extent,aspect="equal",vmin=0.,vmax=.30,cmap=cmap,interpolation="nearest")
    ax.set_xlabel("Along the 45-degree cut (mm)")
    ax.set_ylabel("Across cut (mm)")
    ax.set_title("Glue on the held miter — reconstructed from the native run",loc="left",fontweight="bold",pad=15)
    cb = fig.colorbar(im,ax=ax,shrink=.8,pad=.025)
    cb.set_label("Film thickness (mm)")
    cb.set_ticks([0.,.1,.15,.2,.25,.3])
    spill = receipt["final_metrics"]["contaminated_ml"]
    text = (f"{100*expected['film_coverage']:.1f}% in the 0.10–0.25 mm target band  •  "
            f"{expected['glue_on_held_face_ml']:.3f} mL on the cut face  •  {spill:.3f} mL off target\n"
            "Diagnostic of the surface-glue approximation; this is not a camera image.")
    if 'beads' in metrics:
        bead = metrics['beads'][LEG.name+'/miter']
        text = (f"{bead['wet_length_mm']:.1f} mm of cut length with glue  •  "
                f"largest gap {bead['largest_gap_mm']:.1f} mm  •  {bead['volume_ml']:.3f} mL on the cut face\n"
                "Diagnostic of the surface-glue approximation; this is not a camera image.")
    fig.supxlabel(text,fontsize=10,x=.46,y=.015)
    output = root/"glue-thickness.png"
    fig.savefig(output,dpi=170)
    plt.close(fig)
    np.savez_compressed(root/"reconstructed-film.npz",thickness_m=film.thickness,cell_area_m2=film.area,
                        source_run=root.name,coverage=film.coverage(model.settings))
    print(json.dumps(dict(output=str(output),reproduced_coverage=film.coverage(model.settings),
                         reproduced_volume_ml=float(film.volume.sum()*1e6))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory",type=Path)
    plot(parser.parse_args().run_directory.resolve())
