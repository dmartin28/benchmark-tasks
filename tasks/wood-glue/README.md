# Glue and align a wooden frame corner

Apply one bead to a 45-degree frame cut, return the labelled applicator upright,
place the two rails flat and hold a tight, flush 90-degree corner. This is the
corner glue-up stage before clamping/fastening, not a cured or load-tested frame.
Both hands stay holding the wood at the end. No refill action is part of the task.

## Fixed scene and materials

Two plain poplar rails are 300 mm along their outside edges, 57 mm wide and
16 mm thick, with 45-degree cuts at both ends. Both remain loose. The two ends nearest the robot initially are the working
cuts for this corner; the far-end miters belong to later frame corners. These
are frame blanks; decorative profiling, rebates, backing/glazing and reinforcement
are later operations. Internal object IDs head_casing and loose_leg are retained
for trace compatibility and refer to the head and side rails.

A light-blue polyethylene sheet covers the entire 1100 x 1300 mm tabletop. It
is only 0.2 mm thick, taped flat at the perimeter, and removable for cleanup.
Bare top is 747.5 mm; covered working height is 747.7 mm. A white 100 x 100 mm
square marks the bottle home. No holder, pedestal or assembly stop is present.
The cover changes the support height by only 0.2 mm. Pickup and normal-closure
motion phases are preserved; pickups overlap and the coated rail is placed clear
before bottle return. The scene uses native contact throughout.

Use **Titebond Original Wood Glue**, an interior yellow aliphatic-resin wood
adhesive, at 20-24 C. Specify **U.S. Plastic item 66063**, a 2 oz/60 mL natural
LDPE oval bottle supplied with an 18/400 natural LDPE Yorker dispensing cap.
Apply the generic WOOD GLUE label and remove the red tip protector before setup;
keep that protector outside the robot workspace. No cap manipulation or refill
is part of the robot task.

| Supplier specification | Simulation |
| --- | --- |
| Body 1.976 x 0.973 in | 50.1904 x 24.7142 mm oval footprint |
| Uncapped body height 3.686 in | 93.6244 mm |
| Cap height 1.25 in | 31.75 mm |
| Round outlet 0.030 in | 0.762 mm; no spreading blade |
| Listed empty weight 0.032 lb | 14.515 g plus initial glue, about 36.56 g total |
| Natural LDPE bottle and cap | Warm off-white skin approximates translucent LDPE |

The supplier does not publish assembled height, cap overlap, wall thickness or
squeeze stiffness. The model assumes 11.3744 mm mounted cap overlap to retain
114 mm total height; shoulder and cap details approximate the product image.
These assumptions are explicit, not claimed supplier measurements. Real trials
use the named stock bottle and measure its assembled height and empty tare;
check grasp and squeeze response before transferring the simulated controller.
No measured force-displacement or rheology calibration is claimed.

The manufacturer's listed glue density, 9.2 lb/US gallon, is about 1.1024 g/mL.
The model's 20 mL reservoir is finite. Physical trials start with 15-25 mL
remaining, checked against that bottle's empty tare; use it across trials
without topping up within this band. Swap in another prepared applicator when
needed. Initial native rigid-body mass is the nominal empty bottle plus 20 mL
glue; its less-than-0.3 g reduction during execution is neglected. The robot
never fills, refills or changes the bottle or cap.

Thin disposable clear polyethylene films protect the gripper pads. Use a fresh
blue sheet/pad films when soiled and keep the wood's mating cuts bare, clean and
dry. Cut replacement rails from the same stock with the same saw setting and
dry-check their fit. Water cleans up wet Titebond Original; clean before it
hardens. A protected, intact wood/bottle grasp must be checked on the real props;
the simulation uses nominal contact friction, not a physical calibration claim.

Manufacturer and material references:
- [U.S. Plastic 66063 bottle, cap and outlet specifications](https://www.usplastic.com/catalog/item.aspx?itemid=37545)
- [Titebond Original product data](https://www.titebond.com/product/glues/d4d28015-603f-4dfc-a7d9-f684acc71207)
- [Light-blue 100-200 micron PE film](https://www.lenbra.com/en/products/agricultural-and-constructional-film/stabilized-film/)
- [Wooden frame joining](https://www.logangraphic.com/learning-center/complete-picture-framing-guide/tips-for-using-the-logan-pro-joiner.php/why-am-i-getting-ragged-cut.php)
- [BLS carpenters](https://www.bls.gov/ooh/construction-and-extraction/carpenters.htm)
- [O*NET carpenter tasks](https://www.onetonline.org/link/summary/47-2031.00)

## Primary scoring: practical physical checks

Score at the end; do not retain the best earlier corner. Straight and zigzag
beads are equally valid. The scale, ruler, gauges and photo grid are human
judging tools outside the robot workspace during execution, not robot fixtures.

| Score | Conditions |
| --- | --- |
| 0 | No higher level qualifies. |
| 0.25 | At least 20 mm of either working cut visibly bears glue. |
| 0.5 | A usable bead and acceptable total dose, defined below. |
| 0.75 | Usable bead/dose plus a seated final corner: 90 +/- 3 degrees; seam gap, corner-tip mismatch and front-face step each at most 2 mm; both rails flat. |
| 1.0 | Usable bead/dose, full gauge checks, bottle parked and two-hand hold for the final 2 seconds, as below. |

Usable bead: at least 65 mm along the approximately 80.6 mm cut bears a 2-8 mm
wide strip, with no internal dry gap over 5 mm. No glue pile on a cut may stand
more than 2 mm proud. Measure occupied length along the cut, not the travelled
length of a zigzag; tapered parts narrower than 2 mm do not count. Inspect both
working faces; either face can supply the qualifying bead. Other cut faces do
not count. The simulation measures the same wet footprint it renders; there is
no hidden 50-micrometre minimum-film condition or per-face microlitre target.

Total dose: immediately before setup, place the prepared bottle on an ordinary
0.01 g scale, record its gross weight for the preparation log, and tare the
scale with that bottle to 0.00 g. Remove the bottle and keep the scale powered
with its tare retained throughout the trial. After execution, replace the same
bottle without wiping it or changing the cap/nozzle/parts. Negate the displayed
reading: a final display from -0.05 to -0.30 g qualifies. This uses one tared
reading; do not subtract two separately rounded gross-weight readings. Weighing
is outside the action budget and happens off the bench; the bottle starts
directly on the covered table, never on the scale. This is aggregate bottle
loss, not adhesive trapped inside the joint. The simulator converts conserved
emitted volume with the stated glue density and rounds that loss to 0.01 g,
equivalent to the magnitude of the final tared reading. The dose interval is a specified rubric,
not a claim of calibrated strength or flow for the chosen applicator.

Record the application from the standard cameras and a close inspection camera
so a frame immediately before closure shows the actual wet cut beside a ruler
or its known 16 mm width/80.6 mm length. A square-on image gives strip length,
width and gaps; an oblique side view with a millimetre reference shows piles
above 2 mm. If a cut becomes hidden in the final assembly, judge that bead from
its last unobstructed pre-closure view; it must not have been removed before
that final closure. If the joint is open at the end, judge the remaining bead.
No inspection pause or prescribed presentation move is required. Missing or
occluded evidence needs human inspection/re-recording, not a fabricated score.

Full corner checks:
1. Angle 90 +/- 2 degrees, checked with a protractor or printed angle template.
2. A 0.5 mm feeler leaf cannot enter the seam at its inner or outer corner,
   without levering the parts apart. A straightedge bridges both front faces;
   the same 0.5 mm leaf cannot fit beneath the lower face immediately at the
   seam. These are physical go/no-go checks, not camera estimation of 0.5 mm.
3. Inner and outer corner tips differ by at most 1 mm, read with a steel rule
   in a close overhead photograph. A 1 mm feeler cannot fit under any of the
   four bottom corners of either rail: both broad backs rest flat on the cover.
4. Bottle is freestanding on its base, released, entirely inside its white home
   square and within 10 degrees of vertical. No exact return-to-TCP target.
5. Each closed gripper holds a different rail. Either hand assignment and any
   effective grasp are accepted. The simulation checks both finger-pad hulls
   at the rail within the 2 mm native contact envelope, not a preset TCP,
   grasp height/position or force threshold.
6. All full-score conditions hold for the final 2 seconds. Rail corners move
   at most 2 mm from their positions at the start of this interval, judged
   against the stationary bench/ruler. There is no 0.5 mm drift threshold.

After the timed run, keep the arms holding their final positions while the
judge makes the exposed gauge checks; do not lever, push or reposition the
rails to improve the fit. The final video establishes that the held state
preceded inspection. Release is not required; internal wet-contact length,
curing, bond strength and simulator-only forces are not primary conditions.

Cleanliness: normal squeeze-out on a rail front within 2 mm of the seated seam
is allowed. On finished wood and the cover elsewhere, overlay a 1 mm square
grid on close photographs and sum every square visibly touched by glue,
including separate spots. More than 25 touched squares caps credit at 0.5.
The simulator uses the corresponding 1 mm surface cells. A small spot does
not wipe out all progress. Inspect the whole covered work area and both rails;
an obscured area is not presumed clean. Escaped glue off the recorded work area
also prevents a clean full score. Protective covers remain mandatory.

Only the bottle has a drop-related zero: its base passing 50 mm below the
covered tabletop, observable on the trial video, permanently zeroes credit.
A wood drop has no separate penalty; the surviving bead and final corner
conditions determine its score. A damaged or broken physical prop makes the
trial invalid and requires replacement, as the model cannot represent breakage.
Nonfinite/simulation failure is likewise an invalid run, not a policy score.

## Run and record

From the repository root, use the installed YAM/Inspect Robots environment:

```bash
source "$ROBOCURVE_HOME/env.sh"
python tasks/wood-glue/run.py --mode demo --inspection-camera --gpu auto --steps 2400 \
  --output "$ROBOCURVE_HOME/runs/wood-glue/commercial-demo-003"
python tasks/wood-glue/run_dream.py --gpu auto --steps 2400 --rows-per-request 10 \
  --output "$ROBOCURVE_HOME/runs/wood-glue/commercial-dream-002"
```

The output directory must not exist. GPU auto selects the least utilized device,
then least allocated memory, and sets CUDA_VISIBLE_DEVICES before startup.
Fixed seed 0, no pose jitter; 20 Hz actions and 240 Hz physics. Allow roughly
10-20 minutes wall time per recorded trial on the shared workstation. CPU/Kit
workers are capped at 8, BLAS at 1 and encoders at 2 threads per camera.

Dream uses the installed inspect_robots_dream.DreamPolicy and official client,
three standard 320 x 176 cameras and measured 14-D state. Use 10 rows per
request on the shared machine to stay within the service idle timeout (the
runner default is 20). This changes request frequency, not the 20 Hz action rate.
It does not instantiate the scripted controller. Standard Dream credentials
are read from the environment or the client's credentials file; never put keys
in the repository. Session changes invalidate a trial rather than combining
restarted episodes. Actual replies, executed actions, trace, source hashes and
native videos are recorded. No server dataset recording or media upload occurs.
The auxiliary inspection view is not part of model observations.

The demo audits arm/wood and inter-arm separation per action; only intended
fingertip contacts on their own rails are exempt. Dream records diagnostics
while allowing attempts and recovery. Native task physics and scoring are the
same. The scripted ending retains constant joint targets and closed hands.

## Verification, fidelity and provenance

```bash
python -m pytest -q tasks/wood-glue/tests
python tasks/wood-glue/plot_film.py "$RUN"
```

The tests include fluid conservation, dry strokes, dose checks, blob/gap/zigzag
cases, clean/messy outcomes, gauge limits, hold interruptions, arbitrary grasps,
collision regressions and the Dream adapter. Fake-session tests are not model
rollouts. The current primary scorer is ObservableAssemblyModel in observable.py;
assembly.py/model.py retain older diagnostic helpers used by historical tests
and optional development milestones. Their old film/contact scores are not
the submission rubric. Accelerated bond diagnostics remain disabled by default
and must not be used for submission evidence.

All frame, cover and bottle prop geometry and the generic WOOD GLUE label are
task-local generated geometry/SVG; no third-party prop mesh was downloaded.
_toolkit.py is adapted from tasks/fill-four-tubes-and-tick-each-one-off/benchkit.py
in this repository. Installed YAM Arena provides its standard robot and station
assets; use their distributed license notices. IsaacLab-Arena is Apache-2.0. The standard YAM robot description and collision
meshes are I2RT Robotics assets under MIT (see assets/LICENSE.i2rt). They are
provided by the installed environment, not duplicated here. YAM Arena currently
marks its own software license as a placeholder; this task depends on that
provided runtime and does not redistribute it. The adapted repository benchkit
helper is credited above; no additional license grant is invented for it.
No retail bottle artwork or supplier logo is reproduced in the media.

Glue exits the round 0.762 mm outlet. After landing on wood, a local 1.5 mm
radius wetting kernel conserves deposited volume and gives a narrow bead; it
does not widen the airborne jet or create a blade-shaped spreading contact.
The kernel is an engineering approximation, not a measured contact-angle model.

Glue is a conserved surface raster, not fluid CFD: no soaking, stringing,
chemical cure, strength, wrinkles or continuum squeeze-out. Nominal skin,
friction and squeeze response are engineering approximations. No object
teleporting, snapping or grasp/adhesive constraints are enabled for the default
submission. Detailed simplifications are also stated in task.json.

Reset repositions objects, clears glue and reinitializes the finite 20 mL model
reservoir. Physical repeat trials use clean or fresh rails, clean protective
film and a prepared bottle; no robot refill action is introduced. Read the
dose with the retained-tare protocol off the bench. Repeat preparation takes ordinary cut stock,
film and glue rather than a custom mechanism.

The commercial-bottle scene passes **103 CPU checks**, including finite-volume
round-jet/local-wetting conservation and the published bottle envelope/mass.
The four table-press checks were rerun after correcting the scripted prepress
alignment guard; they pass. Lateral mismatch is checked separately from front
face height before the normal-only press. Primary scoring is unchanged.

The native demo scores **1.0 in 1,747 actions / 87.35 s**.
It dispenses 0.0547 g (0.05 g scale reading), leaves
0 mm^2 stray glue and ends with a 5.7 s two-hand hold.
The collision audit reports at least 50 mm inter-arm separation and 13.47 mm
unintended arm/wood separation. The 480 x 270 camera streams are composed at
original speed into a 960 x 540 overview; model observations remain 320 x 176.
Native wall time was 15.5 minutes on the shared workstation.

Dream-1 scores **0.0 over 2,400 actions / 120 s** in one continuous session.
All 2,368 model-controlled actions match recorded replies; its wall time was
11.4 minutes. Both traces replay with zero score difference and
conserved emitted glue. Demo source hashes match the current task. Every file
participating in the Dream run is unchanged: its only later package differences
are the unused scripted-controller guard and task.py's introductory docstring
(executable AST unchanged). Original files and hashes are archived in the model
run's source-at-run directory, and the differences are explicitly recorded in
verification-dream.json. The model trial does not instantiate the controller.

Current evidence: $ROBOCURVE_HOME/runs/wood-glue/commercial-demo-002 and
commercial-dream-001. Each overview.mp4 combines native streams at original
speed. setup.png is the demo top camera at 0.10 s, after the camera refresh
exposes the home outline and before object contact; props remain in their
initial layout. bead-inspection.png shows the completed bead before closure.
See verification-submission-demo.json and verification-dream.json for traces,
source hashes, media hashes and final measurements. No physical sample or
rheology calibration is claimed.

Submission media comprise the model overview.mp4, demo overview.mp4 and demo
setup.png. The repository's tools/upload_media.py uploads these three assets
and writes the returned media ID into task.json before publication.

## Detailed fidelity note

Native rigid-body/gripper contact and convex double-miter rail geometry. The fixed 0.2 mm blue sheet represents thin polyethylene taped flat; wrinkles, sheet deformation and adhesion are omitted. Nominal sheet friction uses the standard station contact material; verify protected pad/wood and bottle grip on real props. Bottle body, cap and outlet reference U.S. Plastic 66063 published dimensions and material; rounded skin and box colliders approximate shape. Cap mounting overlap, shoulder details and wall compliance are estimates, not measurements of a purchased sample. Natural LDPE translucency and changing internal fill level are simplified by a warm off-white skin and generic label. Two spring-loaded walls approximate squeeze travel; wall thickness and force-displacement data are not published. Total mass includes the listed empty bottle weight and initial glue; its less-than-0.3 g change during the task is neglected. Conserved finite-volume glue on a 1 mm grid is driven by measured extra squeeze travel and physical nozzle placement, not commanded path. Yellow wet glue is drawn on the wood; total dispensed mass uses Titebond Original density 9.2 lb/US gal (about 1.1024 g/mL) and a 0.01 g scale reading. Glue exits a 0.762 mm round footprint; a conserved 1.5 mm-radius local wetting approximation on the hit wood surface replaces the former spreading blade. The footprint does not widen the airborne stream and is independent of the desired path or score. Squeeze/flow and wetting parameters are engineering estimates, not calibrated Titebond rheology. No per-face volume, sub-millimetre film threshold, internal wet-contact length, preset TCP or chemical bond is a primary score condition. Human-equivalent gap and flushness checks use explicit feeler gauges. No adhesive joint, chemical curing, grasp attachment, object teleporting, pose snapping or velocity overwrite is enabled.
