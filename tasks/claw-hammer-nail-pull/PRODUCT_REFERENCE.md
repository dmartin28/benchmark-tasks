# Physical hammer, nail and wood references

Checked 18 November 2026. Product photographs are inspection references only;
the simulation uses original procedural geometry.

## Hammer

**Stanley 51-613, 7 oz curved-claw wood-handle nailing hammer.**

- [Manufacturer product listing](https://www.stanleytools.ca/product/51-613/7-oz-curved-claw-wood-handle-nailing-hammer): forged high-carbon steel, curved claw, smooth face, 198 g head, 111 mm head length, 11-1/4 in overall in the indexed specification. The live regional page may redirect to its home page.
- [Purchasable Toolbox Supply listing](https://www.toolboxsupply.com/products/stanley-51-613-curved-claw-hammer-7-oz-wood-handle): SKU 51-613, UPC 076174516135, 7 oz, 11-1/2 in overall; stock/price are checked when buying.
- [Second purchase reference](https://www.highlandwoodworking.com/collections/drilling-driving-tools/products/stanley7ozclawhammer): Stanley 51-613, 7 oz, 11-1/2 in overall.

The detailed product photograph shows a black-painted wooden handle with a
flared butt and waist, a yellow label, grey forged steel head, curved split claw
and polished round striking face. Those features are modeled. The open V-slot
is made from two separate tapered blade meshes and segment colliders, not a
solid head with a painted slot.

Published overall lengths differ by 6 mm (286 vs 292). The first mesh is about
288 mm overall, 111 mm along the head and 23 mm across the head. Unpublished
details—face diameter 23 mm, claw curvature, slot taper, handle cross sections,
head center of mass and total mass—are photo/density estimates. They must be
checked against an actual 51-613 before physical validation. The first mass is
198 g steel plus 55 g wood, **253 g total**, not the shipping/package weight.

## Nail

**Grip-Rite 4C1 bright smooth common 4d, 1-1/2 in**, or the same ASTM F1667
dimensions from an identified batch.

- [Manufacturer common-nail table](https://grip-rite.com/product/common-nail/): steel, diamond point, bright smooth shank, 4d length 1-1/2 in, 12-1/2 gauge, head 1/4 in.
- [4C1 purchase/specification reference](https://www.drikes.com/2852545/product/grip-rite-4c1): same length, gauge and 1/4 in flat round head.
- [Manufacturer nominal dimension comparison](https://www.magnumfasteners.com/product/construction/hand-drive-bulk/common-nail): common 4d 1-1/2 x .099 in, 12-1/2 gauge, 1/4 in head.

Modeled dimensions: 38.1 mm from point to head underside, 2.5146 mm nominal
shank, 6.35 mm head, estimated 1.0 mm head thickness. The head is not enlarged.
Use a fresh smooth bright nail, with 20 mm penetration in a fresh side-grain
location. The full 18.1 mm exposed shaft makes the head accessible to a normal
claw; no nail is loosened beforehand and no starter hole is drilled.

## Board

Use a dry **western red cedar** 2x6 offcut, actual 38.1 x 139.7 mm section,
600 mm long, cut from ordinary stock. For example the [cedar lumber listing](https://www.lowes.com/pd/2-in-x-6-in-x-10-ft-Cedar-Lumber-Common-1-5-in-x-5-5-in-x-10-ft-Actual/1002985702)
identifies western red cedar and actual 1.5 x 5.5 in dimensions. If supplied
green, condition it indoors to the stated moisture range before trials.

Plywood is not used as a synonym for soft wood. Its veneer species, density,
adhesive and thickness determine the withdrawal behavior, so naming plywood
alone would not justify a low force. The cedar reference makes the material
assumption explicit and can be replaced after actual batch measurements.

[USDA Wood Handbook](https://www.fpl.fs.usda.gov/documnts/fplgtr/fplgtr113/fplgtr113.pdf),
fastenings chapter, eq. 7-1 (eq. 8-1 in the centennial edition), gives the
smooth-nail withdrawal estimate. The initial G=0.32 setting predicts about
158 N with this nail and embedment. This is a starting estimate; a measured
withdrawal/lever test is still needed.
