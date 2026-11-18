# Seat and prebend an electrical box

Position the empty box on the channel web and wrap its two steel mounting flaps
around the channel flanges. This is a **supported precrimp** operation: either
hand may continue holding the box body. Both hands must clear the flap material
for the final two-second hold. Final crimping, fasteners and unsupported retention
are outside the task.

## Real-world basis and ability tested

The box is modeled on the nominal 76.2 × 50.8 × 63.5 mm envelope of the
ABB Iberville CI3104-KSSX in the
[ABB device-box catalog](https://tnb.ca.abb.com/en/pdf-catalogues/boxes-and-covers-meter-sockets/boxes-and-covers/device-boxes.pdf).
The simulation represents steel sheet with geometry-dependent elastic resistance,
permanent bending and elastic recovery. Its mounting-flap outline and placement
are estimated from product imagery. Flap thickness is assumed to be 0.3 mm for
the simulation: no physical factory box was available to measure, and the
reviewed manufacturer documents do not specify the bracket's sheet thickness.
The assumed thickness has therefore not been verified against the factory
part. These assets are a procedural approximation, not manufacturer CAD.

The operation itself has a manufacturer reference: page 2 of the
[ABB steel-stud-box installation leaflet](https://tnb.ca.abb.com/en/fichier/whatsnewsousmenu/CI3004STUDBOX-E.pdf)
shows wrapping the brackets around the stud before a separate plier crimp of
bracket and stud. That leaflet describes CI3004-KSSX-1.25, a different box in the
same wraparound family. It supplies evidence for the operation sequence, not
dimensions or material properties for this model. The task ends at the supported
wrapping stage, before the plier crimp. Its three-sided channel omits the return
lips of a conventional C-stud, so the modeled access and capture geometry also
remain approximations.

[O*NET lists fastening small metal or plastic boxes to walls](https://www.onetonline.org/link/summary/47-2111.00)
among electricians' tasks. The ability tested here is maintaining the alignment
of a loose box while forming two separate slotted brackets around an anchored
channel and leaving their return bends set after the forming hand withdraws.
The support hand may continue holding the box body. The score measures the
preparation step, not a completed, self-supporting electrical installation.

| Feature | Basis in this package |
| --- | --- |
| Box envelope | Nominal CI3104-KSSX catalog dimensions |
| Wrap, then crimp sequence | ABB leaflet for CI3004-KSSX-1.25 |
| Flap outline and placement | Estimated from product imagery; specified in `geometry.json` and `iberville_assets.py` |
| Flap thickness | Assumed 0.3 mm for simulation; no physical factory sample available for verification |
| Steel stiffness and plastic-moment scale | Calculated from the modeled geometry and assumed steel properties; equations below |
| Forming force, springback and attachment equivalence to the factory box | Not physically validated |

## Geometry and reproduction

The submitted geometry can be rebuilt without changing its dimensions:

```bash
python tasks/precrimp-electrical-box/iberville_assets.py --output /tmp/precrimp-rebuild
```

This generates the same two USD files byte for byte and a dimensional manifest.
`strap_polygon()` defines each flat blank in millimeters as a 54 × 44 rectangle,
with two radius-6 slots between longitudinal coordinates 20 and 43 at transverse
coordinates ±10, and radius-2 holes at longitudinal coordinates 6 and 50 and
transverse coordinates −19, 0 and 19. The supplied polygonal outlines, including
their hole discretization, are authoritative. Active sheet thickness is 0.3 mm;
the box-relative flap root planes are −0.65 and 64.15 mm. These are specifications
of the prototype, not measurements of the commercial product. A physical build
would require fabrication and characterization of the root attachment and fresh
flaps; no validated commercial replacement part is claimed.

## Setup

Use the pinned Robocurve YAM environment: Isaac Sim 6.1, Isaac Lab 3.0,
Isaac Lab-Arena 0.3.1 and `yam_arena`. Extra Python dependencies are listed in
`requirements.txt`; do not replace the pinned simulation stack.

`task.py` is the Inspect Robots entry point. The supplied USD assets contain the
complete box and anchored three-sided U-channel. `task.json` gives every object,
dimension, material and starting pose. The box starts on its closed back with
both flaps straight; there is no fixture holding the box. Set object jitter to
zero and use seed 0 for the fixed layout. One trial permits 4,800 actions at 20 Hz.

The stage adapter in `iberville_runtime.py` loads the authored articulation and
passive material actuator. `precrimp_native.py` reads its measured state for the
SDK score hook, since the stock rigid-object bounding-box reader does not support
this articulated object. It does not provide state to the model policy. Policies
receive the ordinary YAM RGB images and joint observations and send ordinary
14-dimensional joint/gripper actions.

## Run and score

After activating the pinned environment, inspect GPU utilization and select the
least busy GPU with at least 15 GB free. Pin its physical index in
`CUDA_VISIBLE_DEVICES` and pass the same index as `--renderer-gpu`. `RUN_DIR` must
be a new directory outside the repository.

```bash
CUDA_VISIBLE_DEVICES="$GPU" OPENBLAS_NUM_THREADS=1 \
python tasks/precrimp-electrical-box/demo.py \
  --renderer-gpu "$GPU" --record --output "$RUN_DIR"

python tasks/precrimp-electrical-box/precrimp_scoring.py \
  --run "$RUN_DIR" --output "$RUN_DIR/scoring.json"
```

The demo uses CPU PhysX with GPU rendering, matching the verified controller.
It starts from the loose box and uses only bounded robot actions. It has no
object attachments, external object forces, commanded flap angles or object pose
edits during the trial. The scripted controller may use measured simulator state.
`result.json`, `trace.json`, `actions.json` and execution-source snapshots make the
run auditable. `work_area.mp4` and `top_cam.mp4` are original sensor recordings.
Use the unmodified `top_cam.mp4` for the submission demo: calibrated overhead
perspective, steel-colored flaps, no overlays or visualization colors.
Native runtime depends on the host; the demonstrated motion takes 156.05 simulated
seconds / 3,121 control actions. Record wall time from each run's `result.json`.
The verified top-camera trial took 2,194.3 wall seconds (36.6 minutes) on the
shared host with CPU PhysX and GPU rendering.

For a Dream-1 model run:

```bash
CUDA_VISIBLE_DEVICES="$GPU" OPENBLAS_NUM_THREADS=1 \
python tasks/precrimp-electrical-box/model_run.py \
  --renderer-gpu "$GPU" --output "$MODEL_RUN_DIR"

python tasks/precrimp-electrical-box/precrimp_scoring.py \
  --run "$MODEL_RUN_DIR" --output "$MODEL_RUN_DIR/scoring.json"
```

The launcher invokes the stock Inspect Robots CLI with policy `dream`, the
instruction in `task.json`, scorer `task_score`, seed 0, 4,800 steps, and stored
320 × 176 policy-camera frames. Its observer writes the original 640 × 360 native
top-camera stream to `model_run.mp4`, plus a state/action trace and
`native_result.json`. Observation logging does not change policy inputs or
termination. Offline scoring exits 1 when the final score is below 1.0; the JSON
still contains the measured partial score.
The recorded seed-0 Dream-1 trial used the full 4,800-action budget, took 59.6
wall minutes on the shared host and ended at 0.0; live and offline scores agree.

## Scoring

`precrimp_scoring.py` is shared by live trials and offline trace evaluation. It
uses geometry and the final two seconds of state, never demo stage names, control
targets, contact forces, plastic memory or precomputed success flags.

For any credit, the box must be seated within 5 mm horizontally and 5° of the
channel, its center height within 20 mm of the blue mark, and all fingertips at
least 10 mm from all flap material. Over two seconds its translation range must
be at most 5 mm and its rotation change at most 2°. Body support is permitted.

Each flap contributes 0.25 for an inward return-tail angle of 60–180°, plus 0.25
for an angle of at least 150° with its tip captured inside its own channel half:
at least 3 mm inside the flange inner face and at least 5 mm behind its free edge.
Sum both flap scores. All conditions are defined numerically in `task.json` and
can be checked with an end-on photograph, a ruler and a protractor. Use fresh
straight flaps after each physical trial. Damaged or detached parts score zero;
the simulator does not model fracture.

```bash
python tasks/precrimp-electrical-box/test_scoring.py
python tasks/precrimp-electrical-box/test_force_monitor.py
```

The scoring tests include the supported successful endpoint, partial completion,
wrong fold direction, an unseated box, finger-supported flaps, an insufficient
hold, a moving box and a result beyond the action limit.

## Material and actuator limits

The flaps use an engineering model of low-carbon steel sheet, rather than
freely rotating hinges. Each 0.3 mm flap has 15 passive elastic-plastic joints.
The assumed material properties are Young's modulus E=200 GPa, yield stress
sigma_y=250 MPa and density 7,850 kg/m³. These are generic steel assumptions,
not a material certificate for an Iberville bracket.

At each fold station, the code obtains the net sheet width `b` from the actual
slotted polygon and uses sheet thickness `t` and the station's representative
length `L` to calculate:

- Elastic rotational stiffness: `K = E * b * t³ / (12 * L)` (`EI/L`).
- Plastic-moment threshold: `M_p = sigma_y * b * t² / 4`.
- Linear isotropic hardening: 0.05 Nm/rad; rotational damping: 0.03 Nm·s/rad.

`iberville_assets.fold_stiffness()` supplies the elastic stiffness and
`install_physics.fold_moments()` supplies the moment thresholds.
`flap_material_native.return_map()` updates the plastic rest angles from the
current deformation; the remaining elastic deformation produces springback.
The native task uses a yield multiplier of 1.0. Neither a desired bend angle,
the controller's stage nor the scoring goal enters that material update.
The robot must make contact and do the forming work; it does not command the
flap joints to a target angle.

This is a reduced sheet-bending model, not a finite-element model or a measured
force-displacement curve. It omits fracture, local buckling and a deformable root
attachment. No physical bracket test has calibrated its forming force,
hardening or springback. Geometry and mass are generated from the supplied
solids; total modeled mass is 0.203 kg. The U-channel has no return lips.

For a plausibility estimate only, the rectangular-section plastic moment
`yield_stress × net_width × thickness² / 4` gives 0.248 Nm for a solid 44 mm
section and 0.113 Nm through the 20 mm net width at the slots. At a 10 mm lever
arm these correspond to 24.8 N and 11.3 N, respectively. Contact position,
hardening, springback, root attachment and friction can increase the force.
This calculation is not a measured forming-force bound or a physical success
demonstration.

Thickness is a material part of the task specification: at fixed geometry and
yield stress, the plastic-moment threshold scales with `t²`, and elastic
stiffness with `t³`. A 1.0 mm flap would therefore have about 11.1 times the
plastic moment and 37.0 times the elastic stiffness of this 0.3 mm flap.
The existing demo establishes success only for the specified simulation;
it does not establish that the robot can form a thicker factory bracket.

Standard motor effort limits remain 27 Nm for shoulder joints, 7 Nm for wrist
joints and 40 N for gripper joints. The scripted controller logs brief contact
spikes and stops if a finger remains above 40 N for 20 ms continuously. Native
contact telemetry measures net normal force, not total force including friction.
Brief peaks are retained in the run record. This monitor is a controller check,
not the primary task score. No unsupported retention is claimed.
The demonstrated run contains a 181.7 N normal-contact peak lasting 1 ms during
the rear flap's horizontal insertion. No finger exceeded 40 N continuously for
20 ms. This is simulated contact telemetry, not a physical force measurement or
the gripper motor's commanded force. An isolated numerical contact transient
is a possible explanation; its cause has not been established by a timestep or
contact-solver sensitivity study. The peak alone does not establish the force
needed to bend a real bracket. It is retained in the record as an unresolved
simulation observation, separate from the sustained-overload check above.

## Assets and licenses

All box, flap, stud and track meshes are authored procedurally in
`iberville_assets.py`; no manufacturer CAD, scans, textures or photographs are
redistributed. The product catalog is a dimensional reference only. The standard
YAM arm meshes/URDF are loaded from the installed `yam_arena` package, derived
from [I2RT YAM](https://github.com/i2rt-robotics/i2rt), under the MIT license in
`yam_arena/assets/yam/LICENSE.i2rt`. Dream-look split arm meshes are likewise MIT,
as documented by that package's `assets/yam/dream/SOURCE`. The calibrated station
and cameras are supplied by the installed
[yam-arena](https://github.com/kairosiann/yam-isaac-lab-arena) environment; no copies
of its station code are included here. Its floor, table and camera stand are
procedural primitives; this scene imports no external table textures, scans or
object-library meshes. The installed yam-arena README reports that the project
itself has no declared license; it remains an installed environment dependency,
not redistributed task code. The robot mesh license is separately MIT as above.
Isaac Lab/Arena's installed components retain their upstream licenses.
