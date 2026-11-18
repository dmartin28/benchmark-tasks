# Nail extraction: workflow Step 1

The user approved starting this idea after the initial novelty/feasibility
discussion. Category: **Societal relatability**. A ten-second view of a claw
catching and levering a nail out of a wooden board shows a familiar home-repair
action and its purpose, without occupational population claims.

## Live comparison evidence

- Canonical fetch succeeded at 2026-11-18T01:13 UTC: **4d6191a406a2aa7f97e34a432282a91a96bd5045**. This task branch starts from that exact commit.
- Task-catalog snapshot: **2026-11-18T01:13:22Z**, 216 accepted and 329 pending tasks. Comparisons exclude revising and lapsed tasks, as requested by the user.
- Full paginated open-PR inventory: **2026-11-18T01:18:27Z**, 474 open PRs, including bodies, titles, URLs and labels. Full-manifest keyword searches use the catalog's own canonical/main and PR-head mirror, not the checked-out fork task list.
- Search terms: claw hammer, nail pull, pulling/removing/extracting nails, crowbar, pry bar, levering, hammer. Prior broader brainstorm searches also inspected tool-use equivalents rather than matching names alone.
- Closest accepted: **Start a nail and drive it home**, TK-W437WY, originally [PR #2206](https://github.com/robocurve/benchmark-tasks/pull/2206). Same ordinary hammer family, but repeated impacts drive a nail; this task captures an existing head with a claw and transmits continuous leverage against withdrawal resistance.
- Closest other manipulation: [paint-tin lid prying, PR #922](https://github.com/robocurve/benchmark-tasks/pull/922), where a screwdriver successively lifts a press-fit rim. It lacks a split claw, shaft/head capture and extraction of a buried fastener. Its current catalog status must be refreshed before review/submission.
- Also inspected the accepted spackling, pegboard-tool retrieval and board-breaking tasks, and toolbox opening. None performs nail extraction.
- Other agents' local task branches/folders were read without edits. Their current work concerns electrical box/support scenes and other safety tasks; no nail-extraction scene was found.

No equivalent nail-extraction operation was found among the accepted and pending
tasks checked. This is a dated overlap check, not a guarantee against later entries.
The task board requires sign-in; canonical merged folders and open PRs supply
the public accepted/pending inventory.

Public-suite coverage: cached definitions for LIBERO, BEHAVIOR-1K and RoboCasa;
live RoboDojo task index and DataCenterBench overview from the preceding idea
check. Meta-World/Adroit hammering references concern impacts, not extraction.
Web research found a proposed robot-mounted dedicated nail extractor in a patent,
not a demonstrated ordinary claw-hammer benchmark. This is not an exhaustive
audit of all public robotics suites. Repeat the live comparison before independent
review and submission.

## Five-criterion self-review

| Criterion | Initial assessment |
|---|---|
| Importance | Pass at concept level: recognizable home repair, distinct lever/head-capture manipulation; no fabricated event or rule-card reasoning |
| Construct validity | At risk until real hammer dimensions and material withdrawal resistance are checked; no grip aids or enlarged target; physics must require loaded contact |
| Scoring | Pass at design level: exposed length, complete removal, tray collection, hands clear and released endpoints, measured by ruler/photos/video |
| Clarity/reproducibility | Named hammer/nail/wood and numeric layout provided; actual batch measurements remain outstanding |
| Feasibility | Not yet demonstrated: native stationary scene first, then scripted pickup, engagement, extraction and collection within 2400 steps |

There is no miniaturization: ordinary short nail, full-size hammer and real
lumber cross section. Initial material force is estimated at 158 N at the nail;
lever amplification is expected to reduce handle load, but it is not a substitute
for measuring the grip and arm loads. Keep sharp fractured edges out of gripper
pads; the intended transfer grasps the smooth shank, leaving the point clear.
Wood fracture and nail bending are not simulated. Nothing is ready for submission
until the later workflow stages establish complete feasibility and review it.

## Refresh before model/review preparation

Canonical fetch succeeded; the verified commit is
**cae5a1b8e1961077cc750c3ecd9790baadb2de1a**. Catalog snapshot
**2026-11-18T05:06:46Z** covers **238 accepted and 329 pending** tasks. The
complete paginated open-PR inventory contains **460** PRs; comparisons here
exclude revising and lapsed tasks, following the user's requested scope.

The accepted nail-driving task TK-W437WY remains the closest object family.
Its core is supporting a loose nail while applying repeated face impacts. This
task requires a split claw to capture the underside of an existing head and
transmit a supported rolling lever motion against axial withdrawal resistance.

Pending [#922](https://github.com/robocurve/benchmark-tasks/pull/922) uses a
screwdriver in a rim groove to release a lid's distributed press fit. Pending
[#1628](https://github.com/robocurve/benchmark-tasks/pull/1628) closes sprung
tong jaws around a fuse and rocks its two clip contacts out in sequence.
Pending [#685](https://github.com/robocurve/benchmark-tasks/pull/685) couples a
worm to a cork and uses rack/pinion wings. Their full setup/scoring/fidelity
fields were read from their current PR heads. They share leverage or extraction,
but none performs open-claw capture of a buried headed fastener. An independent
review should assess these ability comparisons; no uniqueness guarantee is made.

The final native GPU demonstration now establishes simulated feasibility:
**1.0 in 1,930 controls / 96.5 s**, seed 0, within the 2,400-control limit. It uses
ordinary dynamic contacts and the unchanged estimated 158 N initial resistance,
with both finger drives capped at 40 N. Physical material calibration remains
outstanding and is disclosed in the fidelity note.

## Refresh immediately before independent review

Catalog snapshot **2026-11-18T05:42:30Z**, canonical main
**cae5a1b8e1961077cc750c3ecd9790baadb2de1a**: **238 accepted and 340 pending** tasks;
complete open-PR inventory **466**, checked 2026-11-18T05:42:35Z.
Comparison remains restricted to accepted and pending tasks. Full manifests
were searched with claw/hammer/nail/extraction/withdrawal/pry/lever synonyms.

New pending [#2594](https://github.com/robocurve/benchmark-tasks/pull/2594)
asks for scrap-lumber sorting around a protruding-nail hazard. Its nail is
rigidly part of a board and cannot pull out, and no hammer is provided. It
tests hazard segregation, rather than capturing and levering a fastener.
The previous nail-driving and lever-tool comparisons still apply.

Dream-1 completed its user-approved seed-0 trial: **0.0 after 2,400 controls**,
with no trial error and no nail withdrawal. The scripted demo remains **1.0
in 1,930 controls**. Both videos and the setup image were verified before review.
