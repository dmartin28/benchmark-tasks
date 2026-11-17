"""Two loose poplar frame rails with 45-degree ends rest on a thin blue bench cover.

The side rail is on the left and the head rail on the right. A natural LDPE oval squeeze bottle with a
0.762 mm round outlet references U.S. Plastic 66063 and stands directly on the table on the right. Only compression of its spring-loaded
sides dispenses glue. Apply one long bead to a cut face, park the bottle and press the miters together
to form a flush 90-degree corner. Keep both pieces held in place. Scoring uses visible bead
dimensions, total bottle weight loss and gauge checks; no hidden wet-contact requirement.
Real-time wood-glue curing is outside this prototype.
"""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from wood_glue.scene import INSTRUCTION, OBJECTS  # noqa: E402,F401
from wood_glue.runtime import install, current  # noqa: E402

SCENE = "station"
WOOD_GLUE_TASK = True
SCORE_OVER = "end"
# The model includes the complete 2-second stable two-hand hold itself.
SUCCESS_HOLD_S = 0.
install()


def score(objects):
    return current().score() if current() is not None else 0.


def success(objects):
    return score(objects) == 1.


def distance(objects):
    model = current()
    return 1. if model is None else 1.-model.score()
