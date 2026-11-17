# Seal a pipe penetration with a can of expanding spray foam

A copper pipe passes through a 50 mm hole in a timber wall section, leaving a 14 mm gap all round it. A 12 oz can
of straw-applicator gap-filler foam (similar to GREAT STUFF Gaps & Cracks; the model isn't an exact replica)
stands upright on the table. The robot has to turn the can upside down (it only dispenses that way),
steer the straw's tip into the narrow ring around the pipe, and work the trigger under the straw while holding the
can still. It meters each dose, because the foam doubles as it cures: too little leaves the ring open, too much bulges
out of the wall's face.

## Requirements

- `yam-isaac-lab-arena` with Isaac Lab-Arena 0.3.1 (Isaac Lab 3.0, Isaac Sim 6.1); start shells with its `yam.env`.
- Nothing else. Every asset is generated from this folder's code; there are no third-party meshes, scans or scenes.

## Files

| File | What it is |
|---|---|
| `seal_pipe_gap_with_expanding_foam.py` | The task file: scene, instruction, the per-step foam monitor, `score()` and `success()`. |
| `geometry.py` | Every dimension and mechanism number (wall, pipe, can, trigger, flow, foam, scoring), in one place. |
| `foam.py` | The flow law (trigger travel and valve attitude to flow), the ring's 36 sectors, the cure, spreading and scoring. |
| `foamviz.py` | Markers that show the foam in the ring and any mess. |
| `build_assets.py` | Writes `assets/can.usda` (the can as an articulation, the trigger on a revolute joint) and `assets/wall.usda`. |
| `scenekit.py`, `demokit.py` | Helpers copied unchanged from the accepted rack-blanking-panel task: USD articulations in a task file, and scripted demos. |
| `demo.py` | The scripted full-score demo. |
| `tests/test_offline.py` | Offline checks: the task file loads, the flow law, the foam model and scoring, and a kinematic dry run of the demo. |

## Run a trial

From this folder. `assets/can.usda` and `assets/wall.usda` are included; to regenerate them from the code first, run
`python build_assets.py`.

```bash
inspect-robots run --embodiment yam_arena -E task=seal_pipe_gap_with_expanding_foam.py -E object_jitter=0 -E control_hz=20 \
    -E cam_width=320 -E cam_height=176 --policy dream -P rows_per_request=20 --scorer task_score --max-steps 2400 \
    --instruction "Seal the gap around the pipe where it passes through the wall with the spray foam. The can only dispenses upside down. Fill the gap only about halfway so the foam can rise." \
    --store-frames --log-dir out
```

`inspect-robots video <log>.json` writes one MP4 per camera. The scene has no randomness (`-E object_jitter=0`), so no
seed is needed. A trial at 20 Hz control (12 physics steps of 1/240 s each) takes about 8 minutes of wall time on one
RTX PRO 6000 with the three cameras recording.

## Scripted demo

```bash
python demo.py seal_pipe_gap_with_expanding_foam.py OUT_DIR --stage ring --max-steps 2400
```

It writes one MP4 per camera, `setup.png` and `result.json` to `OUT_DIR`, and stops when the task succeeds.

1. **Turn the can over.** The left hand picks the can up and turns it about 97 deg. The right hand takes the can's
   lower body and finishes the turn, upside down.
2. **Left, top and right of the gap.** The right hand holds the can. The left hand comes in from the side with its jaw
   shut, the two fingers one thin block between the trigger and the straw. At each point the straw goes in along its own
   line, 7 mm into the gap, and the left pushes the trigger toward the can until about 9 cm3 of liquid is out (it
   watches the dispensed volume, as a person watches the bead). Then the straw comes back out and the can turns to the
   next point.
3. **Swap hands for the bottom.** Near the bottom of the gap the straw must come in nearly level (otherwise it hits
   the pipe stub). With the can tipped that far, the right hand's grip leaves its wrist at its joint limit, so the
   left takes the can's body and the right lets go and presses the trigger instead.
4. **Clear the view.** The can is carried back and down, out of the top camera's view of the pipe, while the foam
   finishes rising.

The demo reads the simulator's state to plan: the can's pose (closed-loop tracking, because the arms sag under
load), the trigger's angle (a closed loop on the press) and the liquid dispensed (to end each dose). It never moves
objects directly or edits the simulation. The left, top and right points are aimed 6 to 12 deg along the ring from their
nominal angles (`RING_ANGLE_TRIM` in `geometry.py`), because each press pushes the held can and its tip drifts along
the gap the same way every time.

## Scoring

`--scorer task_score` reports `score()` at the end of the run; `success()` is a full score with the foam's visible
rise at least 95% complete, held for 1 s.

- **The foam.** Liquid leaves the straw's tip at up to 6 cm3/s, set by how far the trigger is pressed (none in the
  first 25% of its travel), and only while the valve points within 60 deg of straight down (fading to none at 90 deg).
  Liquid that leaves the tip inside the gap is booked to its 10 deg sector. It doubles in volume as it cures (3 s time
  constant). Excess over a sector's volume spreads to its neighbours along the gap, 2% of it per round venting out of
  the face as a bulge. Liquid that leaves the tip anywhere else is tracked as mess but not scored.
- **Coverage.** A sector counts as sealed when its cured foam fills at least 90% of its volume. The score is 0 below 9
  of the 36 sectors, 0.25 from 9, 0.5 from 18, 0.75 from 27 and 1.0 from 35.
- **Bulge.** -0.25 for foam standing more than 10 mm proud of the wall face (a sector's bulging foam spread over three
  times its face area). The score is capped at 0.5 past 20 mm. Never below 0.

**The real trial and the stand-in.** A real trial is scored at the wall face 30 minutes after it ends, from a
straight-on photo with a printed 36-sector ring template over it. A sector is sealed when cured foam spans the gap from
the pipe to the hole's edge across the whole sector, with no opening wider than 2 mm. A ruler across the face measures
the bulge (see `task.json`). The simulation doesn't model depth, so its 90%-of-volume rule stands in for the photo's
face check. The stand-in is conservative: the straw puts the foam in 7 mm from the face of the 38 mm gap, so a real gap
closes at the face before the back.

**Evidence from each demo run:** `closeup.png` (a camera aimed at the hole's face near the end, the foam drawn as
markers) and `score_report.json` (sectors sealed, the fill and bulge per sector, foam in the gap and outside it, and the
step) are written to the output folder alongside the videos. The close-up is an extra camera outside the robot's
observations, and its foam markers are drawn from the scoring ledger: it illustrates the score, it doesn't measure it.

## Results

| Run | Foam in the gap | Outside the gap | Ring sealed | Bulge | Score | Steps |
|---|---|---|---|---|---|---|
| Scripted demo | 38.0 cm3 (9.8, 9.4, 9.4, 9.4 at the left, top, right, bottom) | 0 | 100% | 8.0 mm | 1.0 | 2,260 (success) |
| Dream-1, 2,400 steps | (picks the can up with its right hand and holds it, upright, by the wall) | | | | 0.0 | 2,400 |

The Dream-1 run used an earlier wording of the instruction that also asked to keep foam off the exposed pipe and the
wall face. The can stays upright in its video (checked every 10 s), so it never dispenses, and the run scores 0.0
under the current scoring too.

## Offline checks

```bash
python -m pytest -q tests/
```

No simulator or GPU is needed. The checks cover:
- the task file loads;
- the flow law's dead zone and upside-down gate;
- the foam's spreading and the score tiers;
- a kinematic dry run of the whole demo through yam_arena's IK and URDF collision hulls: every pose reached, every
  joint inside its limits, and the straw inside the gap at each point.
