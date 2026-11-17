"""Every object's size, material and fixed position. Shared by the scene, the scorer, the demo and the tests.

World frame of the YAM station: x forward from the arm bases, y to the robot's left, z up, metres.
The arm bases stand at x 0, y +-0.305, z 0.75; the bench top is at z 0.7475. Nothing is randomized.
No Isaac Sim imports here.

Plan view: the stained board and its rag lie straight ahead, centred between the arms; the red
oily-waste can stands to their left, the black wastebasket to their right, mirror images of each other;
the stain can, its lid and the brush stand behind.
"""
TOP = 0.7475                      # station bench top

# Pine 1x10 shelf board (19 x 235 mm, 440 mm long), just wiped with oil-based stain, lying flat with
# its length pointing away from the robot, centred between the arms, on two 2x4 offcuts set on edge
# (38 x 89 mm, 140 mm long, across the board under its ends) so it is raised 89 mm off the bench.
BOARD_T, BOARD_W, BOARD_L = 0.019, 0.235, 0.44
BLOCK = (0.038, 0.140, 0.089)     # x, y (across the board), z
BOARD_X0 = 0.09                   # the board's near (robot-side) end
BOARD_Y = 0.0                     # centred between the arms: it spans y -0.1175 .. 0.1175
BOARD_TOP = TOP + BLOCK[2] + BOARD_T
BOARD_CENTRE = (BOARD_X0 + BOARD_L / 2, BOARD_Y, TOP + BLOCK[2] + BOARD_T / 2)
BLOCK_CENTRES = [(BOARD_X0 + 0.03 + BLOCK[0] / 2, BOARD_Y, TOP + BLOCK[2] / 2),
                 (BOARD_X0 + BOARD_L - 0.03 - BLOCK[0] / 2, BOARD_Y, TOP + BLOCK[2] / 2)]
STAINED_PINE = (0.22, 0.11, 0.05)     # dark walnut stain
RAW_PINE = (0.86, 0.73, 0.52)

# One used cotton rag (about 200 g/m2), dropped crumpled on the board after wiping, blotched dark with the
# stain (texture rag.png). Its crumpled shape is fixed (rag_mesh.crumple with RAG_SEED, gathered from a
# 250 mm flat grid): about 180 x 190 mm across and 65 mm high, 0.042 m2 of cloth (a 200 mm square rag);
# it keeps its creases on the board. Mesh spacing 10 mm on the flat cloth.
RAG_SIDE, RAG_SPACING, RAG_SEED = 0.25, 0.010, 2
RAG_XY = (0.29, BOARD_Y)
RAG_COLOUR = (0.26, 0.11, 0.035)                        # linear RGB, used only where no texture is drawn
RAG_BEND = 1.8e5          # cloth bending modulus (the base fabric's value)

# Justrite 09200 countertop oily-waste can (2 gal / 8 L), red powder-coated galvanized steel, 232 mm
# tall, 244 mm outer diameter, hand-operated self-closing lid that opens at most 60 degrees.
# Hinge at the back (+x), lift tab at the front, facing the robot. To the left of the board.
CAN_XY = (0.33, 0.25)

# Open-top round plastic wastebasket (about 10 L), black polypropylene, 230 mm across, 260 mm tall,
# empty. The ordinary trash bin, to the right of the board, mirroring the can.
BIN_XY = (0.33, -0.25)
BIN_R, BIN_H, BIN_WALL = 0.115, 0.26, 0.004
BIN_COLOUR = (0.10, 0.10, 0.11)

# Gallon (3.78 L) steel can of oil-based wood stain, lid off, 168 mm across, 190 mm tall (a standard
# gallon paint can), on the bench 10 cm past the board's far end, in the middle, its label turned toward
# the overhead camera. Its lid rests upside down behind the wastebasket, the 50 mm foam brush across it.
# All fixed (the task never needs them moved), all clear of the arms' paths.
STAIN_XY = (0.71, 0.0)
STAIN_R, STAIN_H = 0.084, 0.190
STAIN_LID_XY = (0.66, -0.20)
STAIN_LID_R = 0.080
BRUSH_POS = (0.66, -0.20, TOP + 0.006 + 0.010)   # lying across the upturned lid; handle along y, head toward +y


def facing_camera(xy):
    """Angle (rad, about z) of the direction from a point on the bench toward the overhead camera."""
    import math
    return math.atan2(0.0 - xy[1], -0.19 - xy[0])
