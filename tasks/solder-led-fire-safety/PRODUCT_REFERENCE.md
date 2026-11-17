# Purchasable light and physical replication

Checked 17 November 2026. The scene represents a cut-to-length under-cabinet light
being prepared before installation, using these purchased components:

| Part | Exact selection and purchase reference |
|---|---|
| LED tape | **Armacost RibbonFlex Pro 142210**: 12 V DC, 3000 K, 60 LEDs/m, 2.5 m reel. [Manufacturer product selector](https://www.armacostlighting.com/products/12v-white-led-strip-light-60), [specific model listing](https://www.homedepot.com/p/304474307). The manufacturer's selector defaults to a different variant: choose 60 LEDs/m, 3000 K, 8 ft / 2.5 m and verify SKU 142210. |
| Channel | **Armacost 960050N**, Surface / Silver, supplied as five 1 m channels with diffusers and endcaps. [Manufacturer selector](https://www.armacostlighting.com/products/led-channel-strip-light-diffuser). One channel is cut for this trial. |

There is no purchase or checkout in the simulation. The LED supply remains
disconnected; the simulated soldering iron starts powered ON at 350 C in its holder.
The power supply and the rest of the reel/accessories stay outside the work cell.
The depicted 203.2 mm assembly is prepared from these products; it is not a claim
that Armacost sells a factory-made 203.2 mm rigid fixture.

## Dimensions and visible features

| Feature | Simulation and evidence |
|---|---|
| Tape width | **8 mm**, from the manufacturer's tape datasheet. |
| Tape section | Four nominal **2-inch cut sections**, **203.2 mm** modeled length; twelve LEDs, three per section. Physical prep follows four actual marked sections, not an arbitrary ruler cut. “2 in.” is a nominal published spacing; actual printed-pitch variation must be measured on the purchased reel. |
| LEDs | Twelve white SMD **2835** packages, 2.8 x 3.5 mm nominal package footprint; yellow phosphor. Four resistors, copper cut lands every section, white flexible PCB. Package height and individual resistor details are approximations. |
| Tape height | Manufacturer lists 1 mm; model uses 0.25 mm PCB plus approximately 0.75 mm LED package. |
| Copper pads | Two narrow rounded-land-style copper lanes at cut positions, halves at the cut ends. Modeled as simple rectangles, 1.7 mm wide and 3.6 mm long at each cut end; these land dimensions are photo estimates, not published tolerances. The pads are not enlarged for the robot. |
| Channel | Silver U-profile, 203.2 mm cut length, nominal outside 17.526 x 7.112 mm (0.69 x 0.28 in), following the older 960050 supplier listing. Manufacturer confirms 960050N surface U-profile and 12 mm maximum tape width, but does not publish detailed extrusion dimensions in its current text datasheet. Wall thickness, groove and lip geometry are estimated. Check the physical N-version profile before final validation. |
| Connection access | The tape is temporarily indexed 10 mm forward of the open channel end. This exposes the pads during preparation without increasing their size. The diffuser and front endcap are not fitted yet. The red positive lead faces the robot/top camera; the black return is already attached. |
| Leads | 20 AWG red/black pair, trimmed to approximately 75 mm for the prepared segment. The tape datasheet identifies 20 AWG supplied leads. Insulation diameter 2.2 mm is estimated. Red lead is pre-tinned and positioned for soldering. |
| Packaging | The scene uses a **representative corrugated shipping carton** for the lighting parts: 220 x 200 x 65 mm outside, 3 mm walls/floor, open top, lid offstage, estimated 45 g. Its front product label names RibbonFlex PRO 142210. Any plain carton of these declared dimensions can reproduce it. This is **not** a claim about the manufacturer's factory retail package, which the product photo shows as a printed sleeve around the reel. No product photograph is included as an asset. |

## Preparing the real counterpart

1. Obtain SKU 142210 and surface/silver channel 960050N. Keep the light disconnected.
2. Cut a four-section strip through its printed copper cut marks. Cut one channel
   to the same measured length. Remove burrs before it enters the task layout.
3. Remove the diffuser and front cap. Leave the tape temporarily 10 mm forward
   of the channel while preparing its wire connection. Final seating and mounting
   are outside this soldering task.
4. Prepare a 75 mm red/black lead pair, tin the working ends and the cut pads,
   attach the black return, and position the red lead at the positive pad. The
   current simulator treats leads/strip/channel as one rigid prepared assembly;
   physical wire positioning and the final bond need separate validation.
5. Lay the mat clear on the bench at its documented starting position. Put the
   empty 220 x 200 x 65 mm carton beside the light on the cabinet, center
   (580,-120) mm in plan, bottom at cabinet work-surface height. Leave its top open
   and lid offstage, with a product label naming the light. Keep the unused reel
   and accessories outside the work cell. The light starts on the cabinet and
   is brought down onto the clear bench mat during the intended demo. The carton
   can remain on the cabinet throughout.
6. Use a 450 W x 350 D x 300 H mm cabinet carcass, 18 mm panels, upside down on
   the station bench. This exposes its exterior bottom for pre-installation work.
   The front is open and doors are not fitted. A simple carcass drawing is implicit
   in the five measured box parts in `scene.py`; no custom mechanism is needed.

This is preparation for scene reproduction, not authorization for a live hot-work
trial. A cold layout rehearsal does not validate electrical soldering.

## Primary documentation

- [RibbonFlex tape datasheet](https://plytix-files.com/api/v1.1/file/public_files/pim/assets/4f/83/3a/5f/5f3a834f4ec13a1b7c75a3f8/texts/96/04/a6/68/68a60496c8a2bf7e805fa96c/RibbonFlex-Pro-12V-White-SMD-Data-Sheet.pdf), revision 260802: SKU, voltage, density, width/height, LED type, cut pitch and lead gauge.
- [Manufacturer installation guide](https://plytix-files.com/api/v1.1/file/public_files/pim/assets/4f/83/3a/5f/5f3a834f4ec13a1b7c75a3f8/texts/83/c4/11/60/6011c4838569c6c5764d14a7/RFP-White-inst-ENG-FR-Sept2023.pdf): cut at marked lands, polarity, supported leads and installation preparation.
- [Manufacturer FAQ](https://www.armacostlighting.com/en-ca/pages/faq): direct wire-to-copper-pad soldering is supported.
- [Channel datasheet](https://plytix-files.com/api/v1.1/file/public_files/pim/assets/4f/83/3a/5f/5f3a834f4ec13a1b7c75a3f8/texts/9e/51/fa/69/69fa519e89f54bc88476cc90/Tape-Light-Mounting-Channels-Data-Sheet-Apr2026.pdf): surface channel 960050N, material/profile, compatible tape width and supplied parts.
- [Manufacturer channel instructions](https://plytix-files.com/api/v1.1/file/public_files/pim/assets/4f/83/3a/5f/5f3a834f4ec13a1b7c75a3f8/texts/84/c4/11/60/6011c484b02a99ea4c7ac9ca/LED-Tape-Light-Channel-Installation.pdf): cut-to-length preparation.
- [Manufacturer packaging photograph](https://www.armacostlighting.com/cdn/shop/files/RibbonFlex-Pro-12V-White-60-AL.jpg?v=1741967378&width=1600) and [unpacked product photograph](https://www.armacostlighting.com/cdn/shop/files/RibbonFlex-Pro-12V-White-60-contents.jpg?v=1741967378&width=1600): appearance reference only, retained outside the repository for inspection.
- [Older channel supplier listing](https://www.parkrosehardware.com/2783254/product/armacost-lighting-960050): rounded outer cross-section dimensions. These are explicitly distinguished from the current manufacturer's published N-version dimensions.
