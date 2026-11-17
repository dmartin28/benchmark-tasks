"""Commercial cabinet-light preparation with its packaging and a protective mat.

Dimensions are metres. New geometry is procedural; no installed simulator is changed.
"""
from __future__ import annotations

import math
import numpy as np
from yam_arena.task_file import TABLE_TOP
from . import mechkit as m
from .lettering import text_mesh

INSTRUCTION = 'Solder the red power lead to the positive pad of the under-cabinet LED strip.'
TOP = TABLE_TOP['station']
# Full-size inverted bridge-cabinet carcass: exterior bottom faces up for fitting.
CABINET_CENTER = np.array([.525,0.,TOP])
CABINET_SIZE = np.array([.350,.450,.300])
PANEL_SIZE = np.array([.350,.450,.018])
PANEL_CENTER = np.array([.525,0.,TOP+.291])
WORK_TOP = TOP+.300
LIGHT_START = np.array([.490,.010,WORK_TOP])
LIGHT_YAW_DEG = -90.
LIGHT_ROTATION = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
LIGHT_SKU = '142210'
CHANNEL_SKU = '960050N'
CUT_PITCH = .0254*2  # Published nominal 2 in cut spacing, three LEDs per section.
TAPE_LENGTH = 4*CUT_PITCH
TAPE_WIDTH = .008
TAPE_THICKNESS = .001
LED_COUNT = 12
CHANNEL_WIDTH,CHANNEL_HEIGHT = .017526,.007112  # 0.69 x 0.28 in, rounded supplier dimensions.
CHANNEL_OFFSET = .010  # Tape's working end projects 10 mm during connection preparation.
LIGHT_SIZE = np.array([CHANNEL_WIDTH,TAPE_LENGTH+CHANNEL_OFFSET,CHANNEL_HEIGHT])
JOINT_LOCAL = np.array([.002,-TAPE_LENGTH/2+.0018,.00145])
JOINT = LIGHT_START+LIGHT_ROTATION@JOINT_LOCAL
MAT_SIZE = np.array([.3048,.2286,.003])
MAT_START = np.array([.100,-.020,TOP+MAT_SIZE[2]/2])
MAT_TARGET = MAT_START.copy()  # Use the existing bench mat for the soldering work.
LIGHT_TARGET = np.array([*MAT_START[:2],TOP+MAT_SIZE[2]])
# Representative shipping carton for the lighting parts, not factory retail packaging.
CARDBOARD_SIZE = np.array([.220,.200,.065])
CARDBOARD_WALL = .003
# Keep packaging beside the light, with room for the angled channel pickup.
CARDBOARD_START = np.array([.580,-.120,WORK_TOP+CARDBOARD_SIZE[2]/2])
IRON_Y = -.320
IRON_START = np.array([.285,IRON_Y,TOP+.059])
IRON_TIP = np.array([.160,0.,0.])
IRON_AXIS = np.array([1.,0.,0.])


def look(c,roughness=.6,metallic=0.):
    return m.Look(color=m.rgb(*c),roughness=roughness,metallic=metallic)


LOOKS=dict(
    maple=look((.79,.66,.47),.72),grain=look((.64,.48,.30),.85),
    ply_light=look((.77,.64,.45),.86),ply_dark=look((.58,.43,.27),.9),
    cardboard=look((.64,.46,.27),.96),paper_edge=look((.48,.32,.17),.98),
    tape=look((.55,.38,.20),.85),ink=look((.25,.21,.17),.95),
    aluminum=look((.68,.71,.74),.32,.8),steel=look((.57,.60,.63),.28,.8),
    mat=look((.105,.11,.115),.96),weave=look((.16,.165,.17),.98),
    brass=look((.66,.49,.20),.3,.8),blue=look((.10,.30,.57)),
    red=look((.77,.10,.065)),rubber=look((.065,.07,.08),.85),
    white=look((.90,.91,.87)),led=look((.96,.80,.29),.55),
    copper=look((.80,.44,.24),.3,.8),solder=look((.70,.72,.75),.24,.85),
    green=look((.09,.40,.18)),dark=look((.11,.13,.15)),
    package_grey=look((.49,.50,.53),.85),package_pale=look((.83,.84,.86),.9),
    package_yellow=look((.99,.79,.03),.8),cabinet_white=look((.90,.90,.86),.72),
)


def add_box(name,size,center,material,parts,colliders):
    parts.append((name,m.box(size,center),material))
    colliders.append(m.col_box(name+'_c',size,center))


def cabinet():
    p,c=[],[]
    add_box('panel',tuple(PANEL_SIZE),(0.,0.,0.),'maple',p,c)
    # Exterior bottom of a full-size upside-down cabinet. The open front faces the robot.
    for k,y in enumerate((-.216,.216)):
        add_box(f'cabinet_side_{k}',(.350,.018,.264),(0.,y,-.141),'cabinet_white',p,c)
    add_box('cabinet_back',(.018,.414,.264),(.166,0.,-.141),'cabinet_white',p,c)
    add_box('cabinet_inner_top',(.350,.450,.018),(0.,0.,-.282),'cabinet_white',p,c)
    # Hinge mounting plates on the open front identify the carcass as a cabinet.
    for k,z in enumerate((-.075,-.225)):
        p.append((f'hinge_mount_{k}',m.box((.025,.002,.030),(-.135,.205,z)),'steel'))
    # Visible plywood plies on the exposed front edge of the work face.
    for k in range(7):
        z=-.009+(k+.5)*.018/7
        p.append((f'ply_{k}',m.box((.0003,.450,.018/7),(-.1752,0.,z)),
                  'ply_light' if k%2==0 else 'ply_dark'))
    rng=np.random.default_rng(1917)
    for k in range(25):
        x=np.linspace(-.170,.170,55)
        y=rng.uniform(-.220,.220)+.0007*np.sin(x*22+rng.uniform(0,6))
        p.append((f'grain_{k}',m.tube(np.c_[x,y,np.full_like(x,.0091)],.00011,4),'grain'))
    return m.fixed('cabinet',PANEL_CENTER,p,c,LOOKS)


def led_light():
    p,c=[],[]
    # Purchased 142210 tape: 8 mm wide, four nominal two-inch sections, twelve 2835 LEDs.
    # The tape temporarily projects from its matching channel, exposing the actual cut pads.
    add_box('channel_bottom',(CHANNEL_WIDTH,TAPE_LENGTH,.001),(0.,CHANNEL_OFFSET,.0005),'aluminum',p,c)
    for k,x in enumerate((-(CHANNEL_WIDTH-.0012)/2,(CHANNEL_WIDTH-.0012)/2)):
        add_box(f'channel_rail_{k}',(.0012,TAPE_LENGTH,CHANNEL_HEIGHT-.001),
                (x,CHANNEL_OFFSET,(CHANNEL_HEIGHT+.001)/2),'aluminum',p,c)
    p.append(('white_led_strip',m.box((TAPE_WIDTH,TAPE_LENGTH,.00025),(0.,0.,.001125)),'white'))
    c.append(m.col_box('tape_c',(TAPE_WIDTH,TAPE_LENGTH,.00025),(0.,0.,.001125)))
    for k in range(LED_COUNT):
        y=-TAPE_LENGTH/2+(k+.5)*CUT_PITCH/3
        p.append((f'led_package_{k}',m.box((.0028,.0035,.00075),(0.,y,.001625)),'white'))
        p.append((f'led_phosphor_{k}',m.box((.0018,.0025,.00010),(0.,y,.00205)),'led'))
    for k in range(4):
        y=-TAPE_LENGTH/2+(k+.5)*CUT_PITCH
        p.append((f'resistor_{k}',m.box((.00125,.0020,.0005),(0.,y+.006,.0015)),'rubber'))
    # Internal cut lands are full length; the cut ends retain half a land.
    for land in range(5):
        y=-TAPE_LENGTH/2+land*CUT_PITCH
        length=.0072
        if land==0:y+=.0018;length=.0036
        if land==4:y-=.0018;length=.0036
        for sign in (-1,1):
            p.append((f'cut_pad_{land}_{sign+1}',m.box((.0017,length,.00006),(sign*.002,y,.00128)),'copper'))
        if 0<land<4:
            p.append((f'polarity_plus_{land}',m.join(m.box((.0008,.00016,.00004)),m.box((.00016,.0008,.00004))).moved((.0032,y+.0046,.00131)),'ink'))
    # The black return is attached. The red lead is pre-tinned and aligned for soldering.
    for k,(x,material) in enumerate(((-.002,'rubber'),(.002,'red'))):
        wire=np.array([[x,-TAPE_LENGTH/2-.001,.0015],[x,-.117,.002],
                       [-.010+x,-.144,.002],[-.020+x,-.170,.0018]])
        p.append((f'insulated_lead_{k}',m.tube(wire,.0011,10),material))
        p.append((f'tinned_wire_end_{k}',m.cylinder(.0004,.003,(x,JOINT_LOCAL[1]-.0007,.00155),axis='y',segments=12),'solder'))
        c.append(m.col_hull(f'tinned_end_c_{k}',m.cylinder(.0004,.003,(x,JOINT_LOCAL[1]-.0007,.00155),axis='y',segments=8)))
        c.append(m.col_hull(f'wire_c_{k}',m.tube(wire,.0011,6)))
    p.append(('black_wire_joint',m.sphere(.00075,(-.002,JOINT_LOCAL[1],.00155),rings=5,segments=12),'solder'))
    p.append(('red_wire_joint',m.sphere(.00085,(.002,JOINT_LOCAL[1],.00165),rings=5,segments=12),'solder',True))
    mark_y=-TAPE_LENGTH/2+.0052
    p.append(('front_positive_mark',m.join(m.box((.0008,.00016,.00004)),m.box((.00016,.0008,.00004))).moved((.002,mark_y,.00131)),'ink'))
    p.append(('front_negative_mark',m.box((.0008,.00016,.00004),(-.002,mark_y,.00131)),'ink'))
    # The front endcap and diffuser are off for soldering. The rear endcap stays fitted.
    add_box('rear_end_cap',(CHANNEL_WIDTH,.0015,CHANNEL_HEIGHT),
            (0.,CHANNEL_OFFSET+TAPE_LENGTH/2,CHANNEL_HEIGHT/2),'white',p,c)
    return m.free('led_light',LIGHT_START,p,c,LOOKS,mass=.024,friction=(.65,.5),
                  torsional_patch=.005,yaw_deg=LIGHT_YAW_DEG)


def cardboard():
    p,c=[],[]
    length,width,height=map(float,CARDBOARD_SIZE)
    wall=CARDBOARD_WALL
    # Five separate physical panels preserve an empty, open interior.
    add_box('carton_bottom',(length,width,wall),(0.,0.,-height/2+wall/2),'cardboard',p,c)
    for k,x in enumerate((-(length-wall)/2,(length-wall)/2)):
        add_box(f'carton_end_{k}',(wall,width,height-wall),(x,0.,wall/2),'cardboard',p,c)
    for k,y in enumerate((-(width-wall)/2,(width-wall)/2)):
        add_box(f'carton_side_{k}',(length-2*wall,wall,height-wall),(0.,y,wall/2),'cardboard',p,c)
    # The removed lid is offstage. Brown cut rims and seams show the carton depth.
    for k,y in enumerate((-(width-wall)/2,(width-wall)/2)):
        p.append((f'corrugated_rim_{k}',m.box((length-2*wall,wall,.00035),(0.,y,height/2+.0001)),'paper_edge'))
    for k,x in enumerate((-(length-wall)/2,(length-wall)/2)):
        p.append((f'end_rim_{k}',m.box((wall,width,.00035),(x,0.,height/2+.0001)),'paper_edge'))
    # Shipping/product label on the vertical face toward the robot.
    label_x=-length/2-.00012
    p.append(('shipping_label_backing',m.box((.00012,.165,.046),(label_x,0.,0.)),'white'))
    for name,words,z,w,h in [
        ('brand','Armacost',.014,.145,.008),
        ('product','RibbonFlex PRO',.001,.150,.007),
        ('sku','142210 | 12V | 3000K',-.013,.152,.0055),
    ]:
        # Separate ink from the label plane enough to avoid RTX depth fighting.
        glyphs=text_mesh(words,w,h,(0.,0.,0.)).turned(-90.,'y').moved((label_x-.0007,0.,z))
        p.append((name+'_label',glyphs,'ink'))
    return m.free('cardboard',CARDBOARD_START,p,c,LOOKS,mass=.045,friction=(.65,.5),torsional_patch=.012)


def soldering_mat():
    p,c=[],[]
    add_box('felt',tuple(MAT_SIZE),(0.,0.,0.),'mat',p,c)
    for axis in (0,1):
        for sign in (-1,1):
            center=np.zeros(3);center[axis]=sign*(MAT_SIZE[axis]/2-.005);center[2]=.00165
            size=[MAT_SIZE[0]-.008,MAT_SIZE[1]-.008,.0004];size[axis]=.001
            p.append((f'hem_{axis}_{sign+1}',m.box(size,center),'weave'))
    for k,y in enumerate((-.090,.090)):
        eyelet=m.lathe([(.0025,-.0018),(.005,-.0018),(.005,.0018),(.0025,.0018),(.0025,-.0018)])
        p.append((f'eyelet_{k}',eyelet.moved((-.130,y,0.)),'brass'))
    for k,x in enumerate(np.arange(-.147,.148,.006)):
        p.append((f'weave_{k}',m.box((.00035,.211,.0001),(x,0.,.00156)),'weave'))
    return m.free('soldering_mat',MAT_START,p,c,LOOKS,mass=.090,friction=(.75,.65),torsional_patch=.006,yaw_deg=90.)


def soldering_iron():
    p,c=[],[]
    # FX-600 reference: 233 mm overall, 61 g without cord. Shape subdivisions estimated.
    handle=m.lathe([(0.,-.073),(.009,-.073),(.012,-.060),(.011,.025),(.0145,.040),
                    (.0145,.048),(.007,.055),(0.,.055)],32).turned(90,'y')
    p.append(('blue_handle',handle,'blue'))
    c.append(m.col_hull('handle_c',handle))
    for k,x in enumerate(np.linspace(-.043,.025,8)):
        p.append((f'grip_ring_{k}',m.cylinder(.01125,.0014,(x,0.,0.),axis='x'),'rubber'))
    p.append(('heater_shroud',m.cylinder(.0055,.067,(.0885,0.,0.),axis='x'),'steel'))
    c.append(m.col_hull('shroud_c',m.cylinder(.0055,.067,(.0885,0.,0.),axis='x',segments=12)))
    tip=m.lathe([(.0030,0.),(.0030,.018),(.0012,.035),(0.,.038)],24).turned(90,'y').moved((.122,0.,0.))
    p.append(('soldering_tip',tip,'steel'))
    c.append(m.col_hull('tip_c',tip))
    p.append(('temperature_dial',m.cylinder(.007,.003,(-.038,0.,.0125)),'white'))
    p.append(('dial_mark',m.box((.0012,.005,.0003),(-.038,.002,.01415)),'rubber'))
    p.append(('cord_relief',m.cylinder(.005,.022,(-.080,0.,0.),axis='x'),'rubber'))
    return m.free('soldering_iron',IRON_START,p,c,LOOKS,mass=.061,friction=(.85,.7),torsional_patch=.006)


def iron_rest():
    p,c=[],[]
    add_box('steel_base',(.285,.085,.004),(.320,IRON_Y,TOP+.002),'steel',p,c)
    for k,x in enumerate((.245,.310)):
        add_box(f'cradle_{k}',(.012,.028,.044),(x,IRON_Y,TOP+.026),'rubber',p,c)
    # Conventional coil guard surrounds the metal shaft and tip while parked.
    a=np.linspace(0,14*math.pi,240)
    x=np.linspace(.352,.435,len(a));radius=np.linspace(.019,.008,len(a))
    coil=np.c_[x,IRON_Y+radius*np.cos(a),TOP+.059+radius*np.sin(a)]
    p.append(('coil_guard',m.tube(coil,.0017,7),'steel'))
    for k in range(0,len(coil)-1,4):
        c.append(m.col_hull(f'coil_c_{k}',m.tube(coil[k:min(k+5,len(coil))],.0017,5)))
    add_box('guard_bracket',(.008,.030,.029),(.393,IRON_Y,TOP+.019),'steel',p,c)
    return m.fixed('iron_rest',(0.,0.,0.),p,c,LOOKS)


def power_switch():
    p,c=[],[]
    add_box('inline_case',(.065,.043,.023),(0.,0.,.0115),'dark',p,c)
    p.append(('red_off_mark',m.box((.007,.012,.0004),(-.021,0.,.0232)),'red'))
    p.append(('green_on_mark',m.box((.007,.012,.0004),(.021,0.,.0232)),'green'))
    # Joint zero is the initial ON pose, so Arena's zero-joint reset preserves it.
    # Logical angle = native angle + 24 degrees: OFF=0, ON=24.
    shape=m.box((.026,.020,.006)).turned(12.,'y').moved((0.,0.,.025))
    rocker=m.hinge('power','Y',(0.,0.,.024),-24.,0.,
                   [('rocker',shape,'rubber')],
                   [m.col_hull('rocker_c',shape)],spring=.015,friction=.0001,damping=.006,mass=.005)
    return m.mechanism('iron_power_switch',(.600,-.460,TOP),[rocker],LOOKS,
                       frame_parts=p,frame_colliders=c)


def cord_points(iron_pos,iron_R):
    """Visual slack path; endpoint follows the iron, with no cable-force model."""
    start=np.asarray(iron_pos)+iron_R@[-.091,0.,0.]
    anchors=np.array([[.170,-.485,TOP+.008],[.270,-.550,TOP+.005],
                      [.430,-.550,TOP+.005],[.560,-.525,TOP+.005],[.570,-.460,TOP+.012]])
    u=np.linspace(0.,1.,18)[:,None]
    a=start-iron_R@[.040,0.,0.];b=anchors[0]+[0.,.025,.015]
    first=(1-u)**3*start+3*(1-u)**2*u*a+3*(1-u)*u*u*b+u**3*anchors[0]
    distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(anchors,axis=0),axis=1))]
    along=np.linspace(0.,distances[-1],23)[1:]
    last=np.column_stack([np.interp(along,distances,anchors[:,k]) for k in range(3)])
    return np.vstack((first,last))


def power_cord():
    points=cord_points(IRON_START,np.eye(3))
    return m.fixed('iron_cord',(0.,0.,0.),[('slack_cord',m.tube(points,.0025,10),'rubber')],[],LOOKS)


OBJECTS=[cabinet(),led_light(),cardboard(),soldering_mat(),soldering_iron(),
         iron_rest(),power_switch(),power_cord()]
