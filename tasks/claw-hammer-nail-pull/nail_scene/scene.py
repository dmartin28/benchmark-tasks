"""Procedural Stanley 51-613 reference hammer, cedar offcut and common nail.

All dimensions are metres. Product photographs are reference only; every mesh
in this file is original geometry, with no downloaded mesh or texture assets.
"""
from dataclasses import replace
import math
import numpy as np
from yam_arena.task_file import TABLE_TOP
from . import mechkit as m
from .lettering import text_mesh
from .material import (NAIL_LENGTH, NAIL_DIAMETER, NAIL_HEAD_DIAMETER,
                       NAIL_HEAD_THICKNESS, INITIAL_EMBEDMENT,
                       ELASTIC_STIFFNESS, withdrawal_force)

INSTRUCTION = 'Remove the protruding nail from the board with the claw hammer and put the nail in the tray.'
TOP = TABLE_TOP['station']
BOARD_SIZE = np.array([.600,.1397,.0381])
BOARD_START = np.array([.240,.100,TOP+BOARD_SIZE[2]/2+.0005])
BOARD_MASS = 1.02
NAIL_IN_BOARD = np.array([.085,-.020,BOARD_SIZE[2]/2-INITIAL_EMBEDMENT])
NAIL_START = BOARD_START+NAIL_IN_BOARD
HAMMER_START = np.array([.315,-.190,TOP+.014])
HAMMER_QUAT = (0.,math.sqrt(.5),0.,math.sqrt(.5))  # On its side, handle along -y.
TRAY_START = np.array([.140,.320,TOP+.0015])
TRAY_SIZE = np.array([.140,.110,.020])
HAMMER_HEAD_MASS = .198
HAMMER_HANDLE_MASS = .055  # Estimated black lacquered wood, not shipping weight.
HAMMER_MASS = HAMMER_HEAD_MASS+HAMMER_HANDLE_MASS
HANDLE_END = .273
FACE_Z = -.050
CLAW_TIP_Z = .061


def look(color, roughness=.6, metallic=0.):
    return m.Look(color=m.rgb(*color),roughness=roughness,metallic=metallic)


LOOKS = {
    'cedar':look((.72,.51,.34),.86), 'grain':look((.47,.30,.18),.94),
    'grain_light':look((.82,.63,.42),.91), 'endgrain':look((.65,.44,.28),.9),
    'steel':look((.40,.42,.44),.33,.83),
    'polished':look((.70,.72,.74),.22,.90),
    'black_wood':look((.075,.065,.055),.40),
    'handle_grain':look((.14,.115,.085),.52),
    'yellow':look((.98,.76,.015),.63), 'ink':look((.07,.065,.045),.7),
    'nail':look((.72,.74,.77),.28,.85),
    'tray':look((.14,.32,.50),.75),
}


def precision(colliders, offset=.00015):
    return [s.replace('contactOffset = 0.004',f'contactOffset = {offset}') for s in colliders]


def board():
    p=[('cedar_offcut',m.box(BOARD_SIZE),'cedar')]
    # Real wood grain, with no printed grasp target or artificial tool guide.
    rng=np.random.default_rng(708)
    for i in range(38):
        x=np.linspace(-BOARD_SIZE[0]/2+.001,BOARD_SIZE[0]/2-.001,95)
        y=rng.uniform(-.068,.068)+.0009*np.sin(x*18+rng.uniform(0,6))
        p.append((f'grain_{i}',m.tube(np.c_[x,y,np.full_like(y,.01916)],.00010,4),
                  'grain' if i%3 else 'grain_light'))
    for sign in (-1,1):
        p.append((f'end_grain_{sign+1}',m.box((.00025,.1395,.0378),
                    (sign*(BOARD_SIZE[0]/2+.0001),0.,0.)),'endgrain'))
    return m.free('board',BOARD_START,p,
                  precision([m.col_box('wood_collision',BOARD_SIZE)]),LOOKS,
                  mass=BOARD_MASS,friction=(.65,.50),torsional_patch=.012,contact_offset=.0002)


def nail_joint_text():
    """D6 material constraint authored before the first physics step.

    The work cell has one environment. The board is a separate free rigid body;
    the USD material joint transmits equal and opposite forces to it.
    """
    axes=[]
    for name,low,high in [('transX',-.0001,.0001),('transY',-.0001,.0001),
                          ('transZ',0.,1.),('rotX',-3.,3.),('rotY',-3.,3.),
                          ('rotZ',-170.,170.)]:
        axes.append(f'''        float limit:{name}:physics:low = {low}
        float limit:{name}:physics:high = {high}''')
    apis=', '.join('"PhysicsLimitAPI:'+s+'"' for s in
                   ('transX','transY','transZ','rotX','rotY','rotZ'))
    x,y,z=NAIL_IN_BOARD
    return f'''
    def PhysicsJoint "wood_joint" (
        prepend apiSchemas = [{apis}, "PhysicsDriveAPI:transZ", "PhysicsDriveAPI:rotX", "PhysicsDriveAPI:rotY"]
    ) {{
        rel physics:body1 = </P_nail>
        bool physics:collisionEnabled = false
        bool physics:jointEnabled = true
        point3f physics:localPos0 = ({x}, {y}, {z})
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
{chr(10).join(axes)}
        token drive:transZ:physics:type = "force"
        float drive:transZ:physics:stiffness = {ELASTIC_STIFFNESS}
        float drive:transZ:physics:damping = 25
        float drive:transZ:physics:targetPosition = 0
        float drive:transZ:physics:maxForce = {withdrawal_force(INITIAL_EMBEDMENT)}
        token drive:rotX:physics:type = "force"
        float drive:rotX:physics:stiffness = {15*np.pi/180}
        float drive:rotX:physics:damping = {.02*np.pi/180}
        float drive:rotX:physics:targetPosition = 0
        float drive:rotX:physics:maxForce = .5
        token drive:rotY:physics:type = "force"
        float drive:rotY:physics:stiffness = {15*np.pi/180}
        float drive:rotY:physics:damping = {.02*np.pi/180}
        float drive:rotY:physics:targetPosition = 0
        float drive:rotY:physics:maxForce = .5
    }}
'''


def nail_mesh():
    r=NAIL_DIAMETER/2
    return m.lathe([(0.,0.),(r,.003),(r,NAIL_LENGTH),
                  (NAIL_HEAD_DIAMETER/2,NAIL_LENGTH),
                  (NAIL_HEAD_DIAMETER/2,NAIL_LENGTH+NAIL_HEAD_THICKNESS),
                  (0.,NAIL_LENGTH+NAIL_HEAD_THICKNESS)],32)


NAIL_VERTICES=nail_mesh().v


def nail():
    r=NAIL_DIAMETER/2
    p=[('diamond_point',nail_mesh(),'nail')]
    c=[m.col_cylinder('shank_c',r,NAIL_LENGTH-.002,(0,0,(NAIL_LENGTH+.002)/2)),
       m.col_cylinder('head_c',NAIL_HEAD_DIAMETER/2,NAIL_HEAD_THICKNESS,
                      (0.,0.,NAIL_LENGTH+NAIL_HEAD_THICKNESS/2)),
       m.col_hull('point_c',m.lathe([(0.,0.),(r,.003),(0.,.003)],8))]
    spec=m.free('nail',NAIL_START,p,precision(c,.00005),LOOKS,mass=.00155,
                friction=(.45,.35),torsional_patch=None,solver_iters=128,contact_offset=.00005)
    build=spec.build
    def with_joint():
        text=build()
        return text[:text.rfind('}')]+nail_joint_text()+text[text.rfind('}'):]
    return replace(spec,build=with_joint,post_spawn=bind_nail_to_board)


def bind_nail_to_board(prim_path):
    """Bind cross-asset joint targets in the scene layer, outside USD references.

    Targets outside a referenced USD asset are discarded by USD composition;
    authoring here makes the board an actual body rather than the world anchor.
    """
    import isaaclab.sim as sim_utils
    from pxr import Sdf,UsdPhysics
    stage=sim_utils.get_current_stage()
    joint=UsdPhysics.Joint.Get(stage,prim_path+'/wood_joint')
    joint.GetBody0Rel().SetTargets([Sdf.Path(prim_path.rsplit('/',1)[0]+'/board')])
    joint.GetBody1Rel().SetTargets([Sdf.Path(prim_path)])


def handle_mesh():
    # Elliptical cross sections, gentle waist and flared butt seen in the photo.
    rings=[(-.012,.0105,.012),(-.028,.0098,.012),(-.050,.0090,.0110),
           (-.085,.0082,.0100),(-.140,.0084,.0108),(-.205,.0090,.0117),
           (-.250,.0100,.0128),(-.269,.0115,.0140),(-.273,.0110,.0135)]
    vertices=[];faces=[];n=32
    for y,a,b in rings:
        for t in np.linspace(0,2*np.pi,n,endpoint=False):
            vertices.append((a*np.cos(t),y,b*np.sin(t)))
    for k in range(len(rings)-1):
        for i in range(n):
            j=(i+1)%n;a=k*n+i;b=k*n+j;c=(k+1)*n+j;d=(k+1)*n+i
            faces.extend([(a,b,c),(a,c,d)])
    # Close end faces; the head covers the handle's top cap.
    for ring in (0,len(rings)-1):
        index=len(vertices);vertices.append((0.,rings[ring][0],0.))
        for i in range(n):
            j=(i+1)%n
            faces.append((index,ring*n+j,ring*n+i) if ring==0 else
                         (index,ring*n+i,ring*n+j))
    return m.Mesh(np.array(vertices),np.array(faces)),rings


def claw_prong(sign):
    """Separate tapered prongs leave a real open V-slot, including colliders."""
    verts=[];n=25
    for t in np.linspace(0,1,n):
        y=-.003-.037*t*t
        z=.014+.047*t
        normal=np.array([.047,.074*t]);normal/=np.linalg.norm(normal)
        half_thickness=.004*(1-t)+.00055*t
        inner=.00055+.00365*t**1.35
        outer=.0105-.0010*t
        lo,hi=sorted((sign*inner,sign*outer))
        for x,s in ((lo,-1),(hi,-1),(hi,1),(lo,1)):
            verts.append((x,y+s*half_thickness*normal[0],z+s*half_thickness*normal[1]))
    faces=[]
    for k in range(n-1):
        for j in range(4):
            a=k*4+j;b=k*4+(j+1)%4;c=(k+1)*4+(j+1)%4;d=(k+1)*4+j
            faces.extend([(a,b,c),(a,c,d)])
    faces.extend([(0,2,1),(0,3,2),(4*n-4,4*n-3,4*n-2),(4*n-4,4*n-2,4*n-1)])
    mesh=m.Mesh(np.array(verts),np.array(faces))
    # Per-section hulls retain the curved underside and do not bridge the slot.
    colliders=[]
    for k in range(0,n-1,3):
        section=np.array(verts[k*4:(min(k+3,n-1)+1)*4])
        colliders.append(m.col_hull(f'claw_{sign+1}_{k}_c',m.Mesh(section,np.array([[0,1,2]]))))
    return mesh,colliders


def forged_head_mesh():
    # Side silhouette of the forged head, from face neck to the curved crown.
    yz=[(-.013,-.035),(-.010,-.049),(.009,-.049),(.012,-.034),
        (.010,-.023),(.015,-.010),(.015,.012),(.009,.024),
        (-.003,.028),(-.014,.018),(-.017,-.015)]
    head=m.prism(yz,-.0105,.0105)
    return m.Mesh(head.v[:,[2,0,1]],head.f)  # Cyclic permutation preserves winding.


def striking_face_mesh():
    return m.lathe([(0.,-.050),(.0106,-.050),(.0115,-.0488),
                    (.0115,-.041),(.009,-.037),(0.,-.037)],40)


def hammer():
    p=[];c=[]
    head=forged_head_mesh()
    p.append(('forged_head',head,'steel'));c.append(m.col_hull('eye_c',head))
    face=striking_face_mesh()
    p.append(('polished_striking_face',face,'polished'))
    c.append(m.col_hull('face_c',face))
    for sign in (-1,1):
        mesh,cols=claw_prong(sign)
        p.append((f'curved_claw_{sign+1}',mesh,'steel'));c.extend(cols)
        # Brighter edge follows the ground bevel near each tine's tip.
        tip=mesh.v[-16:]
        p.append((f'claw_ground_bevel_{sign+1}',m.Mesh(tip,
                    np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],
                              [8,9,10],[8,10,11],[12,13,14],[12,14,15]])),'polished'))
    handle,rings=handle_mesh();p.append(('black_lacquered_wood_handle',handle,'black_wood'))
    for k in range(len(rings)-1):
        y0,a0,b0=rings[k];y1,a1,b1=rings[k+1]
        segment=m.cylinder(1.,1.,segments=16)
        segment=m.Mesh(segment.v*np.array([max(a0,a1),max(b0,b1),abs(y1-y0)]),segment.f)
        segment=segment.turned(90.,'x').moved((0.,(y0+y1)/2,0.))
        c.append(m.col_hull(f'handle_{k}_c',segment))
    # Label is ordinary product identification, not a team identifier or grip aid.
    for sign in (-1,1):
        p.append((f'yellow_label_{sign+1}',m.box((.00018,.086,.010),
                    (sign*.0089,-.095,.0005)),'yellow'))
        glyph=text_mesh('STANLEY  7oz',.077,.006,(0.,0.,0.),angle_deg=0.)
        # Label text lies on the x-facing sides, visible when the hammer is flat.
        glyph=glyph.turned(90.,'y').turned(90.,'x').moved((sign*.00905,-.095,.0005))
        if sign<0:
            vertices=glyph.v.copy();vertices[:,1]=-.190-vertices[:,1]
            glyph=m.Mesh(vertices,glyph.f)
        p.append((f'product_print_{sign+1}',glyph,'ink'))
    return m.free('hammer',HAMMER_START,p,precision(c),LOOKS,mass=HAMMER_MASS,
                  friction=(.65,.50),torsional_patch=.005,solver_iters=128,
                  start_quat=HAMMER_QUAT,com=(0.,-.035,-.004),contact_offset=.0002)


def tray():
    p=[];c=[];w,l,h=TRAY_SIZE
    for name,size,center in [('floor',(w,l,.003),(0,0,0)),
            ('left',(.003,l,h),(-(w-.003)/2,0,(h-.003)/2)),
            ('right',(.003,l,h),((w-.003)/2,0,(h-.003)/2)),
            ('front',(w-.006,.003,h),(0,-(l-.003)/2,(h-.003)/2)),
            ('back',(w-.006,.003,h),(0,(l-.003)/2,(h-.003)/2))]:
        p.append((name,m.box(size,center),'tray'));c.append(m.col_box(name+'_c',size,center))
    return m.free('tray',TRAY_START,p,precision(c),LOOKS,mass=.070,
                  friction=(.6,.45),torsional_patch=.012)


OBJECTS=[board(),nail(),hammer(),tray()]
# Collision-independent geometry for support and placement checks.
HAMMER_VERTICES=np.vstack([handle_mesh()[0].v,claw_prong(-1)[0].v,
    claw_prong(1)[0].v,striking_face_mesh().v,forged_head_mesh().v])
