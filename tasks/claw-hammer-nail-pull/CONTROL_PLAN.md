# Contact-driven motion sequence

The seed-0 continuous native GPU demonstration reached **1.0 in 1,930 controls
(96.5 s)**, with no script errors or blank frames. Physical specimens have not
been tested; withdrawal resistance remains a labeled estimate.

1. The right hand grasps the ordinary wooden handle 180 mm below the head eye.
   It rocks the handle upward while the head remains table-supported, then
   lifts diagonally into the clear right workspace. The left hand approaches
   the long end of the 600 mm cedar board concurrently.
2. The right hand turns the hammer away from the board, carries it over the
   nail and inserts the open V-slot below the head. The left closed pads press
   the board at table x=60, y=145 mm, clear of the right forearm's sweep.
3. With the curved crown supported on the wood, the right hand follows the
   grasp point through the lever arc. Actual claw contact supplies withdrawal
   against the native material constraint, initially about 158 N. Both grippers
   remain capped at 40 N per finger. No task body poses or velocities are set.
4. Once the nail point clears the wood, the right command unloads its accumulated
   tracking error and lifts the hammer with the freed nail retained in its V.
   The left hand releases board pressure, approaches the smooth shaft from
   behind and draws the nail through the open end of the claw.
5. The left hand carries the nail over the tray, lowers it, releases it and
   retreats through its own workspace. The right hand keeps the hammer clear.
6. The right hand carries the hammer to the clear right side of the table,
   turns it close above the tabletop and seats its head. It lowers the handle
   with the head supported, releases it close to horizontal and retreats.
7. All props remain released for a stable second, with both hands 50 mm clear.

The controller reads object poses and robot joints, never the score or material
release flag. Only robot joint targets are commanded. Pose feedback corrects
arm compliance; there are no artificial grasp joints or magnetic attachments.

Offline reach checks preceded native tests. The final GPU trace was checked
at 96 sampled poses, every 20 controls. Minimum measured convex-hull vertex
distances were 31.1 mm between arms, 40.0 mm from non-finger arm parts to the
board, and 3.3 mm from arm parts to the table. This is not a continuous collision
proof. Intentional finger/board, claw/nail and crown/wood contacts are part of
the task. The video was also inspected at pickup, insertion, loaded levering,
release, collection, hammer placement and the clear-handed endpoint.

Precision tool-pose feedback is used for claw engagement and extraction.
Parking follows reachable nominal arm poses and lets table contact support the
head. The empty right arm then raises its shoulder and returns to READY without
holding the wrist in the constrained tool attitude.
