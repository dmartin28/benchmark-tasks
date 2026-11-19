# Oil a squeaky cabinet hinge with WD-40

Instruction given to the policy: `The cabinet door's top hinge squeaks. Give each gap between its knuckles a short
burst of WD-40 through the straw, swing the door open and shut three times to work it in, then stand the can on the
table and leave the door shut.`

A small pine wall cupboard stands on the table, its inset door hung on two 2-1/2 in butt hinges at the robot's
midline. The top hinge squeaks. A can of WD-40 with its Smart Straw flipped up stands in front of the cabinet. The
robot picks the can up, holds the straw's tip about a centimetre from the hinge's barrel, and gives each of the four
gaps between the knuckles a short burst, pressing the can's red actuator pad with the other hand; then it stands the can back on
the table and swings the door open and shut three times by its knob.

## Why it matters

Spraying WD-40 into a squeaky hinge is one of the most recognisable household fixes there is: most people have done
it or watched it done, and anyone seeing a clip knows at once what the robot is doing and why. Doing it well is a
fine bimanual skill: a 3 mm stream has to land in gaps 12.7 mm apart on a 9 mm barrel, from a can held in one hand
while the other presses the actuator, and in short bursts rather than a flood. It is the usual quick fix, done
without taking the door off: spray into the gaps between the knuckles through the straw, then swing the door back
and forth to work the oil in.

## Why it is hard

1. **Aiming a stream at a gap, not a surface.** Each gap is a joint line between two knuckles; the stream counts only
   where it lands within 3 mm of one. The straw has to be held level, pointing at the barrel, its tip about a
   centimetre away, at four heights 12.7 mm apart.
2. **Pressing pushes the aim off.** Opening the valve takes about 10 N on the pad. That push sinks the held can
   and tries to tip it in the grip, and the straw's tip is 186 mm out from the can's axis, so a one-degree tip moves
   the stream 3 mm (in the simulation the arms are soft as well), so the holding hand has to hold the can so it
   cannot pitch, lift against the push and re-aim while the other presses, and the valve has to stay shut until the
   aim is back on the gap.
3. **Short bursts.** More than 4 g sprayed in all (about 5 s) floods the hinge and the cabinet and scores 0, so the valve
   must be opened only while the stream is on a gap.
4. **Then the door.** The can goes back on the table upright, and the door is swung by its knob (the grip turning
   with the door about the hinge pin) at least 45 deg open and back to shut, three times.

## Files

| File | What |
|---|---|
| `task.py` | The task file: the objects, the spray rule (`_spray`), the scorer hooks, the demo entry point |
| `layout.py` | Every dimension, mass and threshold |
| `scoring.py` | The scorer and the spray recorder, pure Python |
| `demo.py` | The scripted demo (closed loop on the can's pose, its actuator and the door) |
| `tests/test_scoring.py` | Offline tests (35), no simulator |
| `setup.png` | The setup image (top camera, arms at their ready pose) |
| `mechkit.py`, `rigstate.py`, `demokit.py` | Shared helpers, copied unchanged (see below) |

## The scene

| Object | What |
|---|---|
| Cabinet (fixed) | A small wall cupboard of 18 mm pine with a 6 mm hardboard back, 306 x 340 x 200 mm (wide, tall, deep), standing on the table with its front face 460 mm in front of the arm bases; in a real trial it is screwed to a wall panel or clamped so it cannot move |
| Door | Inset, 266 x 300 x 18 mm pine, 0.62 kg, 2 mm gap all round, shut; a turned wooden knob (28 mm across, 20 mm long, on a 12 mm neck) on its left-hand stile |
| Hinges | Two 2-1/2 in (63.5 mm) brass-plated steel butt hinges on the door's right-hand edge (the robot's right), five 12.7 mm knuckles each on a 9 mm barrel, loose pin with a button head; the barrels stand just proud of the face. The top one is the squeaky one: centred 270 mm above the table, its four gaps 251, 264, 276 and 289 mm above the table |
| Can (loose) | An 8 oz WD-40 Multi-Use Product can with the Smart Straw: steel, 66.7 mm across, 171 mm tall to the top of its red actuator pad, 0.31 kg (the maker's product data give 171 x 66.7 mm). A blue body with the yellow label panel and a silver domed top; the red pad (about 64 x 32 mm, flat on top over the valve) sits on the dome, and a grey nozzle arm under its front carries the red straw (3 mm), flipped up into the stream position: level, 105 mm beyond the nozzle, its tip 186 mm in front of the can's axis, pointing away from the robot. It stands upright 280 mm out, 80 mm to the right of the midline |

Coordinates for every object are in `task.json` (setup) and `layout.py`. Nothing else is on the table.

## Scoring (scoring.py)

Three simple tools: a kitchen or jewellery scale reading to 0.1 g, the trial video, and a phone filming the top hinge
in slow motion (60 fps, on a stand 15 cm to the right of the barrel, level with it, looking across it, its view
checked in a dry run so the arms don't block it; a dark card behind the barrel so the stream shows; a strip of white
masking tape on the cabinet's side board beside the barrel, marked with each gap's height and lines 3 mm above and
below it).

- Weigh the can just before and just after the trial: the difference is what was sprayed.
- A gap is **oiled** when the stream's first point of impact on the barrel (not splash) stays between that gap's two
  marks (within 3 mm of the joint line between two knuckles) for at least 6 frames of the 60 fps clip (0.1 s); a sweep
  up the barrel counts for each gap it holds on that long.
- A **swing** is the door opened past 45 deg and brought back to within 10 deg of shut. Only swings after the fourth
  gap is oiled count, so working the door first earns nothing.

| Score | When |
|---|---|
| **0** | the can lost more than 4 g (about 5 s of spraying: the hinge and cabinet flooded), whatever else holds |
| **1.0** | all four gaps of the top hinge oiled, then at least three swings, and at the end the door within 5 deg of shut, the can standing upright on its base on the table (not leaning on the cabinet), and the robot touching neither the can nor the door |
| **0.75** | all four gaps oiled, then at least one swing |
| **0.5** | all four gaps oiled |
| **0.25** | at least two gaps oiled |
| **0** | otherwise |

The 4 g limit: a published stream test of WD-40 through its straw measured 0.81 g/s, so 4 g is about 5 s of spraying
(to check a particular can, weigh it, spray 5 s onto a rag, and weigh it again). The number of swings follows WD-40's
own squeaky-hinge guide, which has the door opened and closed "a few times" after spraying to work the oil in: one
swing earns 0.75, three earn full score. In the simulation, too, a gap needs 0.1 s of stream on it to count; the
grams are the valve's open time at 0.81 g/s; "touching neither" is both grippers at least 3 cm from the can and the
door; and the end state must hold, with the arms still, for 1 s.

## Real trials: keeping the rig clean

Cover the gripper pads with replaceable tape, fit clear covers over the wrist cameras' lenses, and keep the close-up
phone out of the stream's line. The simulated valve opens at about 10 N; measure the real can's opening force with a
spring scale and note it (the arm presses with it, and the holding hand's grip resists it). After each trial wipe the hinge, the door's edge, the cabinet's face and the pad covers with a paper towel; a
paper towel on the table under the hinge and kraft paper taped over the door and cabinet face round the hinge (cut
away round the barrel) catch overspray. Oil left inside the hinge does not affect scoring (the squeak is not scored);
every 20 trials, or once it drips, take the pin out, wipe it and the knuckles dry and refit it. WD-40 is a flammable
aerosol: ventilate the room and keep flames away.

## The spray (a declared rule, `task.py` `_spray`)

Simulators do not spray liquids, so a rule stands in for the stream, and the robot still has to do the real work to
trigger it: hold the can, aim the straw and press the pad. The red pad, with the nozzle and straw, is a sprung slider on the can (6.7 N/mm:
the valve opens at 1.5 mm, about 10 N; 3 mm of travel). Every control step that it is pressed at least 1.5 mm, a
straight stream leaves the straw's tip along the straw; the rule finds where the stream first meets the top hinge's
barrel (a 4.5 mm cylinder about the pin), within 50 cm and not blocked by the door's slab, and credits the step to
the gap within 3 mm of that point's height, if any. It also counts every step the valve is open, wherever the straw
points. Nothing is moved or edited by the rule. No liquid, drips or squeak are simulated, and the door's hinge
friction stays the same after oiling.

## Run it

From this folder, with the benchmark's environment (`source ~/IsaacLab-Arena/yam.env`):

    inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \
        --scorer task_score ...

The layout is fixed (`object_jitter=0`); there are no seeds to set. The scorer reads the end state (`SCORE_OVER =
"end"`, 1 s hold) and the spray recorder's record of the trial.

The scripted demo:

    python demokit.py task.py <out_dir>                   # final: records <out_dir>/full/demo.mp4 and setup.png
    python demokit.py task.py <out_dir> --no-record       # GPU physics, cameras rendered 1 step in 20
    python demokit.py task.py <out_dir> --fast            # physics on the CPU, no cameras (differs a little)

Other demo modes: `idle` (the setup image), `spray` (the four gaps only), `flood` (holds the valve open over 5 s at one
gap: scores 0), `swing` (works the door without oiling: scores 0).

Offline tests (no simulator; `yam-isaac-lab-arena` on `PYTHONPATH` for `yam_arena.task_file`):

    python3 -m unittest discover -s tests -v

The Dream-1 run:

    inspect-robots run --embodiment yam_arena --policy dream \
      -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
      --instruction "<the instruction above>" \
      --scorer task_score --max-steps 3200 --seed 0 --no-prompt --store-frames --log-dir <out_dir>

## Demo (demo.py)

Closed loop: every stage reads the can's pose, its actuator's travel and the door's angle; it never reads the scorer
or the spray rule's record, and nothing is teleported or edited.

1. The right hand picks the can up round its body from the can's right-hand side (fingers level, one pad on the
   can's front and one on its back, so a press on the pad cannot pitch it in the grip), lifts it, and turns it so the
   straw is level and points straight at the hinge.
2. It holds the straw's tip 12 mm in front of the barrel, aimed so the stream would land at the lowest gap. The left
   hand comes down from behind onto the red pad with the underside of its closed fingers, which lie along the pad over
   the valve, and presses it slowly to just short of opening (most of the force) while the right hand lifts against
   the push and re-aims; it stays on the pad for all four gaps. At each gap, once the aim has settled, it presses the
   last fraction of a millimetre for a burst of about 0.3 s, holding the valve open only while the stream lands on
   the gap, then lets back to just shut while the right hand steps the straw up 12.7 mm to the next gap. A gap the
   demo is unsure it hit gets a second burst if its spray budget (4.3 s) allows. Then the left hand eases off slowly,
   the right hand letting its lift go with it.
3. The right hand stands the can back on its spot and lets go.
4. The left hand takes the knob, fingers either side of it, and swings the door about its pin, open about 60 deg and
   shut, three times; then both arms move clear.

## Results

| Run | Score | Steps |
|---|---|---|
| Demo, GPU, recorded | 1.0 (one burst per gap: gaps oiled at steps 513, 602, 719 and 852; the valve open 1.55 s in all, about 1.3 g; door swings at 1333, 1425, 1517) | 1,592 |
| Dream-1 (`--policy dream`, seed 0, full limit) | 0.0: the right hand reached in over the can and touched round its top, and the left hand reached toward the cabinet's front, but neither grasped anything; the can moved about 5 mm, its valve never opened (0.06 mm of travel at most), the door stayed shut, and in the last third both arms drew back toward their bases | 3,200 |

Runtime per trial: the recorded demo trial took about 4 minutes of wall time on one RTX 4080 (16 GB), 6.4 steps
per second with three cameras; a Dream-1 trial to the 3,200-step limit took about
5.5 minutes.

## Third-party code and assets

No meshes, scans or other assets: every object is built from boxes and cylinders here. WD-40 and Smart Straw are
trademarks of the WD-40 Company, named only to identify the product used in real trials. `mechkit.py`, `rigstate.py`
and `demokit.py` are shared helpers copied unchanged from accepted tasks in this repository
(`tasks/shim-sleeper-to-level/mechkit.py`, `tasks/casement-stay-last-hole/rigstate.py`,
`tasks/alaris-set-load/demokit.py`, the last under the MIT licence as that task states).
