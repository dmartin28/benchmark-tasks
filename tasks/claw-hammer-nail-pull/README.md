# Pull a protruding nail with a claw hammer

**Status: scripted native GPU demo and Dream-1 trial complete.** Demo score
**1.0**, seed 0, **1,930 controls / 96.5 simulated seconds**. Dream-1 score
**0.0** after **2,400 controls / 120 simulated seconds**, seed 0. All 16 CPU
tests pass. The local media package is prepared for review.

Instruction: **Remove the protruding nail from the board with the claw hammer
and put the nail in the tray.** Category: **Societal relatability**.

A real-sized curved-claw hammer lies beside a loose cedar offcut. A short common
nail is partly driven into its broad face. A blue plastic parts tray is nearby.
The intended action is to steady the board, catch the shank below the nail's
head with the claw and rock the hammer against the board to withdraw it. The
other hand may finish pulling a nail the hammer has loosened, then collect it.
No clamp, locating funnel, enlarged nail head, tool stand or extra handle is used.

## Reproduce the setup

See `PRODUCT_REFERENCE.md` for the physical hammer and material sources. Units
below are mm from the midpoint between the two arm bases: x forward, y left,
z above the native station tabletop. The layout and seed are fixed, without jitter.

| Object | Type, size and material | Starting layout |
|---|---|---|
| Bench | Native station wooden tabletop, 1100 x 1300 x 60 | Native station; top at world z=747.5 |
| Board | Western red cedar offcut, 600 x 139.7 x 38.1; approximately 1.02 kg | Loose, broad face up; long axis along x; center (240,100,19.55), settles on table |
| Nail | Grip-Rite 4C1 reference: bright smooth common 4d nail, 38.1 under head, nominal 2.515 shank, 6.35 head, estimated 1.0 head thickness and 1.55 g | Point 20 into the board, perpendicular to its broad face; head underside 18.1 above wood; entry at board-local (85,-20), world/table x=325,y=80 |
| Hammer | Stanley 51-613 reference, 7 oz/198 g steel head, black-painted wooden handle; approximately 288 overall x 111 head length x 23 wide; estimated 253 g total | On its side on the table, head-eye origin (315,-190,14); handle toward -y, butt about y=-463; settles naturally |
| Tray | Ordinary blue polypropylene parts tray, 140 x 110 x 20, 3 walls/floor, approximately 70 g | Floor center (140,320,1.5), opening up, long side along x |

The nail starts driven into intact wood, without a drilled or reused hole. In a
real reset, use a fresh equivalent offcut or a fresh location of identical grain
orientation. Prepare that site at the same position by repositioning the board;
keep earlier holes at least 30 mm away. Use a fresh nail each trial. Avoid knots,
splits, ring-shank nails, coated nails, rotten wood and green or soaked wood.
Use dry stock equilibrated indoors, nominal 10-14% moisture, and record the batch.
The board remains loose, resting directly on the table.

## Scoring

All credit requires the hammer, tray and board supported on the tabletop,
objects released, both hands at least 50 mm from all four objects, and a final
stable second (each object changes position by at most 2 mm and orientation by
at most 2 degrees over that whole second). No credit if an object ends off the
bench, the board is split, or the claw did less than 5 mm of the withdrawal.
Ordinary compression marks from the hammer are allowed; a bent removed nail is
allowed. Inspect the video to verify use of the claw. These are endpoint tiers,
not the highest score ever seen during the run:

- **0.25:** nail head raised at least 5 mm from its starting height relative to
  the board, with the nail still embedded.
- **0.50:** raised at least 10 mm, still embedded.
- **0.75:** nail fully withdrawn, released and stably supported outside the tray.
- **1.00:** nail fully withdrawn and entirely inside the tray, below its rim.
- **0:** every other outcome.

The hammer must do at least 5 mm of withdrawal; finishing a loosened nail by
hand is allowed. Moving or lifting the whole board does not count as extraction.
Measure the increase in head-underside height normal to the board with a
ruler and the setup photo. A freed nail in the
tray is unambiguous in an endpoint photo. Whole-trial tool use is checked from
the video, including the claw engaged under the nail head. Simulated contacts
and relative body poses approximate that check. Tray containment checks every
vertex of the nail, including its head, against the actual inner walls and rim.
A 0.5 mm floor-contact tolerance accommodates numerical contact penetration.
No force sensor is needed for scoring a physical trial.

## Material response and limits

`nail_scene/material.py` implements the USDA Wood Handbook average peak
withdrawal estimate, **p = 54.12 G^2.5 D L** N, with diameter and embedment in mm.
The first reference setting is G=0.32, D=2.5146 mm, L=20 mm: about **158 N at the
nail**. This is an estimate for smooth nails in side grain, not a measured force
or a structural allowable load. Actual stock, moisture and nail coatings matter.

The nail is a separate dynamic rigid body. A native D6 joint couples its buried
shaft to the loose board and transmits reaction forces to the board. A
force-limited implicit spring approximates static/yield resistance, with 1e6 N/m
numerical stiffness (about 0.16 mm elastic travel at initial yield). Its plastic
rest position follows yielded extraction, and resistance decreases with the
remaining embedment. This does not write a body pose or move it for the robot.
Once the point clears the surface by 1 mm, the material constraint is disabled
and nail/board collisions are enabled. The grippers still must handle the nail.
The constraint is restored before native reset settling.

The wood does not split or crush, and the nail is rigid rather than elastically
bending. Lateral support is a narrow joint limit, widening near complete
extraction. Estimated angular restoring stiffness starts at 15 Nm/rad with
a 0.5 Nm cap and decreases quadratically with remaining embedment; angular
damping is 0.02 Nms/rad. These support parameters also need physical calibration.
Both grippers are capped at 40 N per finger in every trial. Hammer–nail and
hammer–board contact use physical colliders, including
separate curved claw prongs that retain their open V-slot.

The native demo establishes completion in the specified simulation. Withdrawal
resistance, detailed tool dimensions and the 253 g total mass remain estimates;
physical batch measurements and grip-slip calibration have not been performed.
The full-size geometry, material formula and force caps are recorded so those
measurements can replace estimates without changing the task or its scoring.

## Run the scene and CPU checks

Use the program's pinned Isaac Sim 6.1 / Isaac Lab 3.0 / Arena 0.3.1 / yam_arena
environment. From this directory:

```bash
python3 -m unittest discover -s tests -v
```

Check GPU load, choose the least utilized card with at least 15 GB free (more
free memory on a utilization tie), then pin it:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
CUDA_VISIBLE_DEVICES=<chosen-index> python3 render_scene.py --output <new-output-directory>
```

This runs a stationary seed-0 reset and 40 control steps at 20 Hz, writes the
native cameras, three development close/angled stills and `receipt.json`, and
checks that the nail remains 20 mm embedded and the board lies flat. It is a
scene preview, not a task-completion demonstration. Runtime depends on GPU load;
the measured receipt records wall time. Planned task limit: 2400 controls/120 s.

## Run and score the reference motion

From this task directory in the pinned environment, check GPU usage and pin
one GPU as above. The full native top-camera recording uses GPU physics:

```bash
CUDA_VISIBLE_DEVICES=<chosen-index> python3 demokit.py task.py <run-directory> --seed 0 --mode full
```

The runner writes `full/demo.mp4`, `full/setup.png`, `full/trace.jsonl` and
`full/result.json`. The result contains the actual endpoint score, control
count, runtime and source hashes. The task's `score()` uses the same endpoint
scorer during model runs. The reference controller reads live poses and joints;
it commands only the robot, without moving props or editing material parameters.

For motion debugging without cameras, use native CPU PhysX:

```bash
CUDA_VISIBLE_DEVICES=<chosen-index> python3 demokit.py task.py <debug-directory> --seed 0 --mode full --fast
```

Diagnostic stages are `pickup`, `engage`, `lever`, `raise`, `collect` and `full`;
`none` leaves the robot idle. An incomplete stage intentionally scores zero
when it ends holding an object. All trials have zero layout jitter. The
recorded native GPU trial used 96.5 simulated seconds and 1012.52 s of
wall time excluding startup on the shared GPU. Runtime is recorded per trial.
The limit remains 2400 controls / 120 s. The H.264 top-camera video is 640 x
360 at 20 fps, 4.33 MB, with zero blank frames. Key pickup, levering, collection,
placement and final frames were inspected. See `verification.json` for hashes,
endpoint conditions and the sampled collision audit.

## Run Dream-1

Configure the Dream client's private credentials file before running; a trial
uses paid API credits. From this task directory in the pinned environment,
choose and pin a GPU as above, then run the stock Inspect Robots policy:

```bash
CUDA_VISIBLE_DEVICES=<chosen-index> inspect-robots run \
  --embodiment yam_arena --policy dream -P model=dreamlabs/dream-1-beta \
  -E task="$PWD/task.py" -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction 'Remove the protruding nail from the board with the claw hammer and put the nail in the tray.' \
  --scorer task_score --max-steps 2400 --seed 0 --store-frames \
  --no-prompt --grader none --no-rerun --no-rerun-save --log-dir <new-run-directory>
```

The model receives the native top and both wrist cameras. The completed JSON
log reports the endpoint `task_score`; the task's `score()` supplies that value.
For export, `inspect-robots video <completed-log.json> --out <video-directory>`
encodes the stored camera streams. Use only its native top-camera stream for
the model-run submission video, without overlays or composite views. Both
submitted videos use H.264/yuv420p at 20 fps and must each be under 50 MB.

The recorded trial used `dreamlabs/dream-1-beta`, seed 0, zero jitter and the
same scene, scoring and physics sources as the full-score demo. It ended at
the 2,400-step limit without a trial error; both the official Inspect Robots
metric and native endpoint scorer reported **0.0**. The nail remained embedded
and no claw withdrawal was recorded. Wall time excluding startup was
**1450.84 seconds** on the shared GPU. A local launcher limited application
threads to four and saved a native verification receipt; policy actions came
from the stock Dream client. Recoverable expired-chunk requests were retried
by that client. The model-run video contains all 2,401 native top frames,
320 x 176 at 20 fps, 120.05 s, approximately 0.49 MB, with no empty or all-black
frames. The complete log and per-step records are retained with the run.

## Assets and dependencies

All task prop meshes and colliders are generated from code in `nail_scene/scene.py`.
No purchased mesh, product photograph or downloaded texture is distributed.
The photograph used to match the hammer remains outside the task package.
Product names and yellow product-label text identify the reference tool.

`mechkit.py` derives from the helper distributed with accepted benchmark tasks;
this copy adds task-local scene-layer joint binding and a pre-reset
material-constraint hook. It generates
plain USD and hooks scene loading in memory, without editing the installed sim.
`preview_camera.py` and `lettering.py` derive from the preceding task's local
helpers. Lettering uses installed DejaVu Sans outlines; its license is included
as `assets/LICENSE-DejaVu.txt`. YAM assets and the station environment come from
the pinned simulation installation and are not bundled.
