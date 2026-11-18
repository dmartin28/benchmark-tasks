"""Pull a partly driven common nail out of a loose cedar board using a claw hammer."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from nail_scene.scene import INSTRUCTION,OBJECTS
from nail_scene.runtime import TRIAL
from nail_scene.scoring import score as endpoint_score

SCENE='station'
SCORE_OVER='end'
SUCCESS_HOLD_S=0.


def score(objects):
    return endpoint_score(TRIAL.outcome)


def success(objects):
    return score(objects)==1.


def demo_script(ctx):
    from nail_scene.controller import demo_script as motions
    yield from motions(ctx)
