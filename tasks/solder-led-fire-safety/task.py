"""A full-size wall cabinet lies upside down while its under-cabinet lighting is
prepared. A disconnected Armacost RibbonFlex Pro 142210 strip lies in a matching
cut-to-length channel, with its solder end exposed toward the robot. Its empty
open cardboard shipping carton sits beside the loose light on the cabinet.
A clear soldering mat lies on the bench in front. A powered soldering iron,
preheated to 350 C, is ready on its steel rest. The black return is attached;
the red lead is pre-tinned and aligned with the positive pad but not soldered.
The LED supply is disconnected. The trial has no iron startup or warm-up step.
"""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from solder_scene.scene import INSTRUCTION, OBJECTS  # noqa: E402,F401
from solder_scene.runtime import TRIAL  # noqa: E402
from solder_scene.scoring import score as endpoint_score  # noqa: E402

SCENE='station'
SCORE_OVER='end'
SUCCESS_HOLD_S=0.  # Runtime scorer counts the full stable second.


def score(objects):
    return endpoint_score(TRIAL.outcome)


def success(objects):
    return score(objects)>=1.


def demo_script(ctx):
    from solder_scene.controller import demo_script as motions
    yield from motions(ctx)
