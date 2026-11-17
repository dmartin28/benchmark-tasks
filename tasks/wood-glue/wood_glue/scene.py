"""Fixed station layout. Every dimension is in metres, every mass in kg."""
from dataclasses import dataclass
import math

import numpy as np
from yam_arena.task_file import Box, TABLE_TOP

from ._toolkit import Slider, tool
from .geometry import Pose, Surface, box_surfaces, prism, yaw_matrix

BARE_TOP = TABLE_TOP["station"]
COVER_THICKNESS = .0002
TOP = BARE_TOP+COVER_THICKNESS
WOOD = (.76, .57, .34)
CREAM = (.92, .88, .76)
GREY = (.28, .31, .34)
YELLOW = (.98, .73, .08)
LENGTH, WIDTH, THICK = .300, .057, .016
RISER = 0.
FACE_LENGTH = WIDTH*math.sqrt(2)
FACE_X = .265
HOLDER_XY = (.62, -.38)
HOLDER_ELEVATION = 0.
BOTTLE_YAW = 0.
BOTTLE_PANEL_RATE = 3000.
BOTTLE_BODY_EXTENSION = 0.
# U.S. Plastic 66063: 1.976 x 0.973 x 3.686 in uncapped body,
# 1.25 in cap and 0.030 in round outlet. Mounted cap overlap is estimated;
# overall height stays 114 mm until a physical sample can be measured.
BOTTLE_RADII = np.array([.973,1.976])*.0254/2
BOTTLE_UNCAPPED_HEIGHT = 3.686*.0254
BOTTLE_CAP_HEIGHT = 1.25*.0254
NOZZLE_DIAMETER = .030*.0254
BOTTLE_EMPTY_MASS = .032*.45359237
GRASP_LOCAL = np.array([0., 0., .045+BOTTLE_BODY_EXTENSION])
TIP_LOCAL = np.array([0., 0., .114+BOTTLE_BODY_EXTENSION])
GLUE_DENSITY_G_ML = 9.2*453.59237/3785.411784  # Manufacturer: 9.2 lb per US gallon.
INSTRUCTION = ("Apply a narrow bead of wood glue from the labelled bottle along a frame rail's miter. "
               "Keep the rail fronts and bench cover clean apart from a small line of squeeze-out at the joint. "
               "Return the bottle upright to its starting area, press the two rails together into a tight, flush "
               "90-degree corner on the covered bench, and finish holding both rails in place.")


@dataclass(frozen=True)
class Miter(Box):
    mirror: bool = False

    def mesh(self):
        # Rectangle with a 45-degree corner cut off, not a diagonal bounding box.
        outline = [(0., -WIDTH/2), (LENGTH, -WIDTH/2), (LENGTH-WIDTH, WIDTH/2), (WIDTH, WIDTH/2)]
        if self.mirror:
            outline = [(x,-y) for x,y in outline][::-1]
            target = 3
        else:
            target = 3
        return prism(self.name, outline, THICK, target_edge=target)


def casing(name, y, mirror, fixed):
    yaw = -math.pi/4 if mirror else math.pi/4
    center = np.array([WIDTH/2, 0., THICK/2])
    desired = np.array([FACE_X, y, TOP+RISER+THICK/2])
    origin = desired-yaw_matrix(yaw)@center
    return Miter(name=name, pos=tuple(map(float,origin)), size=(LENGTH, WIDTH, THICK), color=WOOD,
                 mass=None if fixed else .12, yaw=yaw, mirror=mirror)


HEAD = casing("head_casing", -.060, True, False)
LEG = casing("loose_leg", .065, False, False)


def fixed_box(name, center, size, color=GREY, yaw=0.):
    return Box(name=name, pos=tuple(map(float,center)), size=tuple(map(float,size)), color=color, yaw=yaw)


# Bottle stands directly on the bench, without a holder or support.
HOLDER = []
BOTTLE_BASE = np.array([*HOLDER_XY, TOP])
BOTTLE = tool("glue_bottle", tuple(map(float,BOTTLE_BASE)),
              mass=BOTTLE_EMPTY_MASS+.020*GLUE_DENSITY_G_ML-.010, yaw=BOTTLE_YAW,
              boxes=[((0.,0.,.040+BOTTLE_BODY_EXTENSION/2),(2*BOTTLE_RADII[0],.032*BOTTLE_RADII[1]/.029,.080+BOTTLE_BODY_EXTENSION),CREAM),
                     ((0.,0.,.089125+BOTTLE_BODY_EXTENSION),(.021,.021,.01375),(.86,.86,.82)),
                     ((0.,0.,.103+BOTTLE_BODY_EXTENSION),(.005,.005,.014),CREAM),
                     ((0.,0.,.112+BOTTLE_BODY_EXTENSION),(.0014,.0014,.004),CREAM)],
              sliders=[Slider("panel_left", (0.,.025*BOTTLE_RADII[1]/.029,.045+BOTTLE_BODY_EXTENSION),
                              (.034*BOTTLE_RADII[0]/.0175,.008*BOTTLE_RADII[1]/.029,.040), axis="Y",
                              lower=-.006, upper=0., rate=BOTTLE_PANEL_RATE, damping=8., color=YELLOW, mass=.005),
                       Slider("panel_right", (0.,-.025*BOTTLE_RADII[1]/.029,.045+BOTTLE_BODY_EXTENSION),
                              (.034*BOTTLE_RADII[0]/.0175,.008*BOTTLE_RADII[1]/.029,.040), axis="Y",
                              lower=0., upper=.006, rate=BOTTLE_PANEL_RATE, damping=8., color=YELLOW, mass=.005)],
              points=[("tip",tuple(TIP_LOCAL)), ("grasp",tuple(GRASP_LOCAL))], friction=.65)
COVER = fixed_box('bench_cover',(.36,0.,BARE_TOP+COVER_THICKNESS/2),
                  (1.10,1.30,COVER_THICKNESS),color=(.35,.57,.73))
OBJECTS = [HEAD, LEG, *HOLDER, BOTTLE, COVER]


def initial_poses():
    return {s.name:Pose(np.asarray(s.pos),yaw_matrix(s.yaw)) for s in OBJECTS}


def surfaces():
    out = []
    for spec in OBJECTS:
        if isinstance(spec,Miter):
            out.extend(spec.mesh()[2])
        elif spec.name != "glue_bottle":
            out.extend(box_surfaces(spec.name,spec.size))
    # Bench includes the floor plane so off-table glue is never silently discarded.
    out.append(Surface("bench/top","bench",np.array([[-2.,-2.,TOP],[2.,-2.,TOP],[2.,2.,TOP],[-2.,2.,TOP]])))
    return out


def all_initial_poses():
    return {**initial_poses(), "bench":Pose(np.zeros(3))}
