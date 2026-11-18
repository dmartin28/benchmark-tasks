"""Seat an electrical box on the flat web of a vertical U-channel and precrimp
its two steel mounting flaps around the channel flanges. One hand may support
the box body; both hands must clear the flaps for the final two-second hold.
This preparatory operation ends before final crimping or unsupported mounting.

The box must be seated within 5 mm horizontally and 5 degrees of the channel,
with its center height within 20 mm of the blue mark. Return-tail bends of
150--180 degrees and tips at least 3 mm inside their own flange and 5 mm behind
its free edge earn full credit. Keep all fingertips at least 10 mm from the
flaps and hold the seated result steady for two seconds; body support is allowed.
"""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from yam_arena.task_file import Box
import iberville_assets as a
a.configure_flap_sheet(.0003)
import work_layout as layout
from place_box import INITIAL_POSITION,INITIAL_ROTATION
from yam_arena.ik import matrix_to_quat_xyzw
layout.BOX_INITIAL_POSITION=tuple(float(v) for v in INITIAL_POSITION)
layout.BOX_INITIAL_ROTATION=INITIAL_ROTATION.copy()
layout.BOX_INITIAL_ROT_XYZW=tuple(float(v) for v in matrix_to_quat_xyzw(INITIAL_ROTATION))
import iberville_runtime as runtime
from precrimp_scoring import Scorer
import precrimp_native

SCENE='station';SCORE_OVER='end';SUCCESS_HOLD_S=2.
class DeviceBox(Box):pass
class StudFrame(Box):pass
MASS=json.loads((Path(__file__).parent/'geometry.json').read_text())['total_mass_kg']
OBJECTS=[StudFrame('stud_frame',pos=layout.CHANNEL_POSITION,size=(a.DEPTH+.004,.180,.501)),
    DeviceBox('device_box',pos=layout.BOX_INITIAL_POSITION,size=(a.DEPTH+.0026,a.WIDTH+.054+.067,a.HEIGHT+.018),mass=MASS)]
_scorer=None
def reset_score():
    global _scorer
    _scorer=Scorer()
def score(snapshot):
    global _scorer
    if _scorer is None:reset_score()
    return _scorer.update(snapshot)['score']
def success(snapshot):return score(snapshot)>=1.-1e-9
def scoring_report():return {} if _scorer is None else _scorer.last_report

runtime.FORMING_YIELD_SCALE=1.;runtime.FORMING_DAMPING=.03;runtime.PLACEMENT_RUN=True
runtime.CUSTOM_SCORING=True
runtime.install(Path(__file__).parent,DeviceBox,StudFrame,frame_asset='stud_frame_u.usda')
precrimp_native.install(Path(__file__).parent)
