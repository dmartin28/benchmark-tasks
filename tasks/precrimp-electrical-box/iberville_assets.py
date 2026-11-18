"""Procedural edge-flap box prototype using CI3104-KSSX nominal size. Dimensions are SI.

Published box size is distinguished from estimated sheet and strap geometry.
The passive joints approximate bending; they are not a calibrated commercial crimp.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import shapely
from shapely.geometry import Polygon, LineString, Point, box as rectangle
from shapely.geometry.polygon import orient
import trimesh
from yam_arena.ik import matrix_to_quat_xyzw

TOP = .7475
HEIGHT, WIDTH, DEPTH = .0762, .0508, .0635
GEOMETRY_REVISION = "edge-flaps-return-wrap-v2"
PRODUCT_REFERENCE = "ABB Iberville CI3104-KSSX nominal envelope; revised edge-flap approximation"
BODY_SHEET = .0012                 # assumed, not published for this item
BODY_SUPPORT_SHEET = .0008
STRAP_SHEET = .0008                # legacy default; task and asset CLI select 0.3 mm
STRAP_LENGTH, STRAP_HEIGHT = .054, .044
EDGE_CLEARANCE = .0005             # per side between stud and inner tab face
ROOT_OFFSET = EDGE_CLEARANCE + STRAP_SHEET/2
# Nominally at the box edges; the 0.9 mm allowance includes metal thickness.
FRONT_SETBACK = -ROOT_OFFSET
STRAP_ROOTS = {'front':-ROOT_OFFSET, 'rear':DEPTH+ROOT_OFFSET}
STUD_FLANGE, STUD_LIP, STUD_SHEET = .03175, .0047625, .00047752
WEB_FACE_CLEARANCE = .0005
CORNER_STATION = WEB_FACE_CLEARANCE+STUD_FLANGE+EDGE_CLEARANCE+STRAP_SHEET/2
WRAP_ANGLE_DEG = 175.
WRAP_ARC_SEGMENTS = 12
# Neutral-axis radius: half channel sheet + half flap sheet + surface clearance.
WRAP_RADIUS = (STUD_SHEET+STRAP_SHEET)/2+EDGE_CLEARANCE
WRAP_CHORD = 2*WRAP_RADIUS*np.sin(np.deg2rad(WRAP_ANGLE_DEG)/(2*WRAP_ARC_SEGMENTS))
FOLD_STATIONS = tuple([0.,.018]+[float(CORNER_STATION+k*WRAP_CHORD)
    for k in range(WRAP_ARC_SEGMENTS+1)]+[STRAP_LENGTH])
YOUNGS_MODULUS_ASSUMED = 200e9
DENSITY = 7850.
STUD_DEPTH, STUD_HEIGHT = DEPTH, .500
STUD_FRONT, STUD_Y = .360, .100
# Tiny tilt lets the rear flap and opposite rear body edge share the table.
INITIAL_TILT = float(np.arctan((ROOT_OFFSET+STRAP_SHEET/2)/WIDTH))
BOX_POSITION = (.220,.040,float(TOP+DEPTH*np.cos(INITIAL_TILT)+WIDTH/2*np.sin(INITIAL_TILT)+.0002))
BOX_ROT_XYZW = tuple(float(v) for v in np.array([np.sin(INITIAL_TILT/2),np.cos(INITIAL_TILT/2),
                               np.sin(INITIAL_TILT/2),np.cos(INITIAL_TILT/2)])*2**-.5)



def configure_flap_sheet(thickness):
    """Select a physical flap thickness before constructing a trial or asset.

    Box walls, device ears and the opposite support arm are unchanged. Never
    call this while a trial is running. The submitted task selects 0.3 mm.
    """
    global STRAP_SHEET,ROOT_OFFSET,FRONT_SETBACK,STRAP_ROOTS,CORNER_STATION,WRAP_RADIUS,WRAP_CHORD,FOLD_STATIONS
    if not .0002<=thickness<=.0012:raise ValueError('Flap thickness outside prototype range')
    STRAP_SHEET=float(thickness);ROOT_OFFSET=EDGE_CLEARANCE+STRAP_SHEET/2
    FRONT_SETBACK=-ROOT_OFFSET;STRAP_ROOTS={'front':-ROOT_OFFSET,'rear':DEPTH+ROOT_OFFSET}
    CORNER_STATION=WEB_FACE_CLEARANCE+STUD_FLANGE+EDGE_CLEARANCE+STRAP_SHEET/2
    WRAP_RADIUS=(STUD_SHEET+STRAP_SHEET)/2+EDGE_CLEARANCE
    WRAP_CHORD=2*WRAP_RADIUS*np.sin(np.deg2rad(WRAP_ANGLE_DEG)/(2*WRAP_ARC_SEGMENTS))
    FOLD_STATIONS=tuple([0.,.018]+[float(CORNER_STATION+k*WRAP_CHORD) for k in range(WRAP_ARC_SEGMENTS+1)]+[STRAP_LENGTH])


def vec(values):
    return '('+', '.join(f'{float(x):.10g}' for x in values)+')'


@dataclass
class Solid:
    mesh: trimesh.Trimesh
    cells: list
    material: str = 'Steel'
    physical: bool = True


def box_solid(center, size, material='Steel', physical=True):
    mesh=trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    return Solid(mesh,[mesh] if physical else [],material,physical)


def extrude_yz(poly, thickness):
    """Closed extrusion with holes, using constrained 2-D triangulation."""
    poly=orient(poly,sign=1.)
    triangles=shapely.constrained_delaunay_triangles(poly)
    vertices=[];indices={};tri=[]
    def vertex(p):
        key=tuple(np.round(p,12))
        if key not in indices:indices[key]=len(vertices);vertices.append(key)
        return indices[key]
    for item in triangles.geoms:
        xy=list(item.exterior.coords)[:3]
        a,b=np.array(xy[1])-xy[0],np.array(xy[2])-xy[0]
        if a[0]*b[1]-a[1]*b[0]<0:xy[1],xy[2]=xy[2],xy[1]
        tri.append([vertex(p) for p in xy])
    n=len(vertices);points=[[-thickness/2,*p] for p in vertices]+[[thickness/2,*p] for p in vertices]
    faces=[]
    for a,b,c in tri:faces.extend([[c,b,a],[a+n,b+n,c+n]])
    for ring in [poly.exterior,*poly.interiors]:
        xy=list(ring.coords)
        for a,b in zip(xy,xy[1:]):
            i,j=vertex(a),vertex(b)
            if i>=n or j>=n:raise ValueError('Missing constrained boundary vertex')
            faces.extend([[i,j,j+n],[i,j+n,i+n]])
    mesh=trimesh.Trimesh(vertices=points,faces=faces,process=True)
    if mesh.volume<0:mesh.invert()
    if not mesh.is_watertight:raise ValueError('Sheet extrusion is not watertight')
    if not np.isclose(mesh.volume,poly.area*thickness,rtol=1e-5):raise ValueError('Sheet volume mismatch')
    return mesh



@lru_cache(maxsize=256)
def convex_regions(wkb):
    """Merge coplanar triangles only when their union remains exactly convex.

    This changes collision partitioning, preserving holes and the outer contour.
    It is used only for planar extrusions, never the jogged opposite support.
    """
    source=shapely.from_wkb(wkb);regions=[]
    for triangle in shapely.constrained_delaunay_triangles(source).geoms:
        current=triangle;again=True
        while again:
            again=False
            for j,other in enumerate(regions):
                if current.boundary.intersection(other.boundary).length<1e-10:continue
                merged=current.union(other)
                if merged.geom_type!='Polygon' or len(merged.exterior.coords)>17:continue
                if abs(merged.convex_hull.area-merged.area)>1e-14:continue
                current=merged;regions.pop(j);again=True;break
        regions.append(current)
    union=shapely.union_all(regions)
    if source.symmetric_difference(union).area>1e-13:raise ValueError('Convex partition changed the sheet')
    if abs(sum(r.area for r in regions)-source.area)>1e-13:raise ValueError('Overlapping convex partition')
    return tuple(regions)


def plate_solid(poly, x, direction=1, s_origin=0., material='Strap', jog=False, thickness=None, merge_cells=False):
    """Visual sheet and convex triangular-prism colliders retain every opening."""
    thickness=STRAP_SHEET if thickness is None else thickness
    def transform(mesh):
        mesh.vertices[:,1]=direction*(mesh.vertices[:,1]-s_origin)
        mesh.vertices[:,0]+=x
        if jog:
            s=-mesh.vertices[:,1]
            mesh.vertices[:,0]+=.008*np.clip((s-.040)/.008,0,1)
        if direction<0:mesh.invert()
        return mesh
    polygons=list(poly.geoms) if poly.geom_type=='MultiPolygon' else [poly]
    mesh=trimesh.util.concatenate([transform(extrude_yz(p,thickness)) for p in polygons])
    if merge_cells and jog:raise ValueError('Convex merging requires a planar sheet')
    cells=[transform(extrude_yz(t,thickness)) for p in polygons
           for t in (convex_regions(p.wkb) if merge_cells else shapely.constrained_delaunay_triangles(p).geoms)]
    return Solid(mesh,cells,material)


def strap_polygon():
    p=rectangle(0,-STRAP_HEIGHT/2,STRAP_LENGTH,STRAP_HEIGHT/2)
    for z in [-.010,.010]:
        p=p.difference(LineString([(.020,z),(.043,z)]).buffer(.006,quad_segs=8))
    for s in [.006,.050]:
        for z in [-.019,0.,.019]:p=p.difference(Point(s,z).buffer(.002,quad_segs=6))
    return p


def wrap_angles(side,total_degrees=WRAP_ANGLE_DEG):
    """Finite-radius polygon: half-angle end joints and full-angle inner joints."""
    sign=-1 if side=='front' else 1
    return {f'{side}_fold_{2+k}':sign*total_degrees/WRAP_ARC_SEGMENTS*
            (.5 if k in [0,WRAP_ARC_SEGMENTS] else 1.)
            for k in range(WRAP_ARC_SEGMENTS+1)}


def fold_stiffness(index,stations=None):
    """EI / dual length keeps subdivisions from silently softening the sheet.

    Young's modulus is an assumption. This elastic discretization is not a
    calibration of plastic forming or the actual stamped bracket.
    """
    stations=FOLD_STATIONS if stations is None else stations
    s=stations[index]
    net=strap_polygon().intersection(LineString([(max(1e-6,s),-STRAP_HEIGHT),
                                                (max(1e-6,s),STRAP_HEIGHT)])).length
    length=(stations[1]-stations[0])/2 if index==0 else (stations[index+1]-stations[index-1])/2
    return float(YOUNGS_MODULUS_ASSUMED*net*STRAP_SHEET**3/(12*length))


def strap_centerline(side,total_degrees=WRAP_ANGLE_DEG):
    angles=wrap_angles(side,total_degrees);point=np.array([STRAP_ROOTS[side],WIDTH/2])
    points=[point.copy()];theta=0.
    for k,(start,end) in enumerate(zip(FOLD_STATIONS,FOLD_STATIONS[1:])):
        theta+=np.deg2rad(angles.get(f'{side}_fold_{k}',0.))
        point=point+(end-start)*np.array([-np.sin(theta),np.cos(theta)])
        points.append(point.copy())
    return np.array(points)


def stamp_ring(center, radius, axis='X'):
    mesh=trimesh.creation.annulus(r_min=radius-.00018,r_max=radius,height=.00008,sections=48)
    if axis=='X':mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
    mesh.apply_translation(center)
    return Solid(mesh,[],'Seam',False)


def body_solids():
    t=BODY_SHEET;w=WIDTH;h=HEIGHT;d=DEPTH;depth_scale=d/.0762
    solids=[box_solid((d-t/2,0,0),(t,w,h))]
    for side in [-1,1]:
        solids.extend([box_solid(((d-t)/2,side*(w-t)/2,0),(d-t,t,h)),
                       box_solid(((d-t)/2,0,side*(h-t)/2),(d-t,w-2*t,t))])
    # Closed knockout outlines: decoration only; the wall is not punched out.
    for z in [-.020,.020]:
        for r in [.0111,.0141]:solids.append(stamp_ring((d-t-.00006,0,z),r))
    for side in [-1,1]:
        for x in [depth_scale*.022,depth_scale*.055]:
            solids.append(stamp_ring((x,0,side*(h/2+.00004)),.0111,axis='Z'))
    # Front device-mounting ears with real holes, rather than filled blocks.
    for side in [-1,1]:
        ear=rectangle(-.006,side*h/2-.006,-.006+.012,side*h/2+.009) if side>0 else rectangle(-.006,-h/2-.009,.006,-h/2+.006)
        ear=ear.difference(Point(0,side*(h/2+.004)).buffer(.0021,quad_segs=8))
        solids.append(plate_solid(ear,.001,material='Steel',thickness=BODY_SUPPORT_SHEET))
    # Opposite-side factory support arm (the separate X support).
    support=Polygon([(0,-.022),(.027,-.022),(.027,-.005),(.015,-.005),
                     (.015,.003),(.067,.003),(.067,.020),(0,.020)])
    for s in [.025,.035,.045]:support=support.difference(rectangle(s-.0008,.006,s+.0008,.017))
    for s,z in [(.006,.014),(.061,.012),(.022,-.013)]:
        support=support.difference(Point(s,z).buffer(.002,quad_segs=6))
    arm=plate_solid(support,.060*depth_scale,direction=-1,jog=True,thickness=BODY_SUPPORT_SHEET)
    arm.mesh.apply_translation((0,-w/2,0))
    for c in arm.cells:c.apply_translation((0,-w/2,0))
    solids.append(arm)
    # Two visible factory joining-screw heads; these are not installation tools.
    for z in [-.029,.029]:
        head=trimesh.creation.cylinder(radius=.0032,height=.0015,sections=20)
        head.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
        head.apply_translation((.066*depth_scale,-w/2-.0007,z))
        solids.append(Solid(head,[],'Steel',False))
    # Small continuous attachment lands bridge the thickness/clearance allowance.
    # They belong to the box-side bracket, not to a stud fixture or robot aid.
    for x0,x1 in [(-ROOT_OFFSET-STRAP_SHEET/2,.003),
                  (d-.003,d+ROOT_OFFSET+STRAP_SHEET/2)]:
        solids.append(box_solid(((x0+x1)/2,w/2-STRAP_SHEET/2,0),
                                (x1-x0,STRAP_SHEET,STRAP_HEIGHT),'Strap'))
    return solids


def material(name,color,metallic=.65,roughness=.40,root='Box'):
    return f'''def Material "{name}" {{
      token outputs:surface.connect = </{root}/{name}/shader.outputs:surface>
      def Shader "shader" {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = {vec(color)}
        float inputs:metallic = {metallic}
        float inputs:roughness = {roughness}
        token outputs:surface
      }}
    }}'''


# Cook thin convex prisms in millimetre-sized coordinates. The reciprocal
# mesh transform preserves every physical dimension and the authored mass.
# PhysX otherwise expands some submillimetre hulls by about 0.7 mm.
COLLISION_COORDINATE_SCALE=1000.


def mesh_prim(name,mesh,material_path=None,collider=False):
    points=mesh.vertices*(COLLISION_COORDINATE_SCALE if collider else 1.)
    faces=np.asarray(mesh.faces);schema=' (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "PhysxConvexHullCollisionAPI"])' if collider else ' (prepend apiSchemas = ["MaterialBindingAPI"])'
    attrs='''double3 xformOp:scale = (0.001, 0.001, 0.001)
      uniform token[] xformOpOrder = ["xformOp:scale"]
      uniform token purpose = "guide"
      uniform token physics:approximation = "convexHull"
      float physxCollision:contactOffset = 0.0002
      float physxCollision:restOffset = 0
      float physxConvexHullCollision:minThickness = 0.000001''' if collider else f'rel material:binding = <{material_path}>'
    return f'''def Mesh "{name}"{schema} {{
      point3f[] points = [{', '.join(vec(p) for p in points)}]
      int[] faceVertexCounts = [{', '.join(['3']*len(faces))}]
      int[] faceVertexIndices = [{', '.join(map(str,faces.ravel()))}]
      uniform token subdivisionScheme = "none"
      bool doubleSided = true
      {attrs}
    }}'''


def aggregate_mass(solids):
    props=[s.mesh.mass_properties for s in solids if s.physical]
    masses=np.array([p['volume']*DENSITY for p in props])
    center=sum(m*p['center_mass'] for m,p in zip(masses,props))/masses.sum()
    inertia=np.zeros((3,3))
    for m,p in zip(masses,props):
        r=p['center_mass']-center
        inertia+=DENSITY*p['inertia']+m*(r@r*np.eye(3)-np.outer(r,r))
    eig,rot=np.linalg.eigh(inertia)
    if np.linalg.det(rot)<0:rot[:,0]*=-1
    q=matrix_to_quat_xyzw(rot)[[3,0,1,2]]
    return float(masses.sum()),center,eig,q


def body(name,solids,position=(0,0,0),root=False):
    mass,center,inertia,q=aggregate_mass(solids)
    apis=['PhysicsRigidBodyAPI','PhysicsMassAPI','PhysxRigidBodyAPI']
    if root:apis+=['PhysicsArticulationRootAPI','PhysxArticulationAPI']
    geometry=[]
    for k,s in enumerate(solids):
        geometry.append(mesh_prim(f'visual_{k}',s.mesh,f'/Box/{s.material}'))
        geometry.extend(mesh_prim(f'contact_{k}_{j}',c,collider=True) for j,c in enumerate(s.cells))
    return f'''def Xform "{name}" (prepend apiSchemas = {str(apis).replace(chr(39),chr(34))}) {{
      double3 xformOp:translate = {vec(position)}
      uniform token[] xformOpOrder = ["xformOp:translate"]
      float physics:mass = {mass}
      point3f physics:centerOfMass = {vec(center)}
      float3 physics:diagonalInertia = {vec(inertia)}
      quatf physics:principalAxes = {vec(q)}
      bool physxArticulation:enabledSelfCollisions = false
      int physxArticulation:solverPositionIterationCount = 32
      float physxRigidBody:maxDepenetrationVelocity = 0.5
      {chr(10).join(geometry)}
    }}'''


def fold_joint(name,parent,child,pivot,stiffness):
    return f'''def PhysicsRevoluteJoint "{name}" (prepend apiSchemas = ["PhysicsDriveAPI:angular"]) {{
      rel physics:body0 = </Box/{parent}>
      rel physics:body1 = </Box/{child}>
      uniform token physics:axis = "Z"
      point3f physics:localPos0 = {vec(pivot)}
      point3f physics:localPos1 = (0,0,0)
      float physics:lowerLimit = -175
      float physics:upperLimit = 175
      uniform token drive:angular:physics:type = "force"
      float drive:angular:physics:stiffness = {stiffness*np.pi/180}
      float drive:angular:physics:damping = 0.00006981317
      float drive:angular:physics:maxForce = 100
      float drive:angular:physics:targetPosition = 0
    }}'''


def box_usd(merge_flap_colliders=False):
    parts=[material('Steel',(.25,.28,.31)),material('Strap',(.35,.38,.41)),
           material('Seam',(.065,.075,.085),metallic=.35),body('housing',body_solids(),root=True)]
    polygon=strap_polygon();masses={'housing':aggregate_mass(body_solids())[0]}
    for side,x in STRAP_ROOTS.items():
        parent='housing'
        for k,(start,end) in enumerate(zip(FOLD_STATIONS,FOLD_STATIONS[1:])):
            poly=polygon.intersection(rectangle(start,-STRAP_HEIGHT,end,STRAP_HEIGHT))
            name=f'{side}_strap_{k}'
            solid=plate_solid(poly,0,s_origin=start,merge_cells=merge_flap_colliders)
            position=(x,WIDTH/2+start,0)
            parts.append(body(name,[solid],position))
            pivot=(x,WIDTH/2,0) if k==0 else (0,start-FOLD_STATIONS[k-1],0)
            parts.append(fold_joint(f'{side}_fold_{k}',parent,name,pivot,fold_stiffness(k)))
            masses[name]=aggregate_mass([solid])[0];parent=name
    text='#usda 1.0\n(defaultPrim="Box"\n metersPerUnit=1\n upAxis="Z")\ndef Xform "Box" {\n'+'\n'.join(parts)+'\n}\n'
    return text,masses


def frame_usd(profile='c'):
    if profile not in ['c','u']:raise ValueError('Channel profile must be c or u')
    lip=STUD_LIP if profile=='c' else 0.
    t=STUD_SHEET;h=STUD_HEIGHT;d=STUD_DEPTH;bottom=.0008
    solids=[box_solid((d/2,t/2,bottom+h/2),(d,t,h))]
    for x in [t/2,d-t/2]:
        solids.append(box_solid((x,STUD_FLANGE/2,bottom+h/2),(t,STUD_FLANGE,h)))
        if lip>0:
            lip_x=lip/2 if x<d/2 else d-lip/2
            solids.append(box_solid((lip_x,STUD_FLANGE-t/2,bottom+h/2),(lip,t,h)))
    solids.append(box_solid((d/2,STUD_FLANGE/2,.00035),(d+.004,.180,.0007)))
    for x in [-.00165,d+.00165]:solids.append(box_solid((x,STUD_FLANGE/2,.0157),(.0007,.180,.030)))
    # Visual height reference, supplying no support.
    solids.append(box_solid((-.00035,STUD_FLANGE/2,.200),(.0001,.024,.003),'Mark',False))
    geometry=[]
    for k,s in enumerate(solids):
        geometry.append(mesh_prim(f'visual_{k}',s.mesh,f'/Frame/{s.material}'))
        geometry.extend(mesh_prim(f'contact_{k}_{j}',c,collider=True) for j,c in enumerate(s.cells))
    parts=[material('Steel',(.30,.33,.36),root='Frame'),
           material('Mark',(.02,.15,.48),metallic=0,root='Frame'),*geometry]
    return '#usda 1.0\n(defaultPrim="Frame"\n metersPerUnit=1\n upAxis="Z")\ndef Xform "Frame" {\n'+'\n'.join(parts)+'\n}\n'


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Rebuild the submitted 0.3 mm prototype assets.')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent)
    folder=parser.parse_args().output
    configure_flap_sheet(.0003)
    out=folder/'assets';out.mkdir(parents=True,exist_ok=True)
    usd,masses=box_usd(merge_flap_colliders=True);(out/'device_box.usda').write_text(usd)
    (out/'stud_frame_u.usda').write_text(frame_usd('u'))
    manifest={'product':PRODUCT_REFERENCE,'geometry_revision':GEOMETRY_REVISION,'published_box_size_mm':{'height':HEIGHT*1000,'width':WIDTH*1000,'depth':DEPTH*1000},
        'estimated_body_sheet_mm':BODY_SHEET*1000,'estimated_strap_sheet_mm':STRAP_SHEET*1000,
        'estimated_strap_size_mm':[STRAP_LENGTH*1000,STRAP_HEIGHT*1000],
        'front_setback_mm':FRONT_SETBACK*1000,
        'front_setback_basis':'Prototype edge planes with 0.5 mm clearance plus half flap thickness; not an ABB mounting drawing.',
        'strap_root_planes_mm':{side:x*1000 for side,x in STRAP_ROOTS.items()},
        'nominal_edge_planes_mm':[0,DEPTH*1000],
        'clear_opening_between_flaps_mm':(STRAP_ROOTS['rear']-STRAP_ROOTS['front']-STRAP_SHEET)*1000,
        'clearance_each_side_mm':EDGE_CLEARANCE*1000,
        'stud_web_mm':STUD_DEPTH*1000,'stud_flange_mm':STUD_FLANGE*1000,
        'stud_lip_mm':0.,'stud_sheet_mm':STUD_SHEET*1000,
        'channel_profile':'Three-sided U prototype; return lips omitted.',
        'initial_tilt_degrees':float(np.degrees(INITIAL_TILT)),
        'manufacturer_mounting_geometry_verified':False,
        'estimated_fold_stations_mm':[x*1000 for x in FOLD_STATIONS],
        'folds_per_flap':len(FOLD_STATIONS)-1,'wrap_target_degrees_each_flap':WRAP_ANGLE_DEG,
        'wrap_neutral_radius_mm':WRAP_RADIUS*1000,'wrap_arc_segments':WRAP_ARC_SEGMENTS,
        'return_tail_length_mm':(STRAP_LENGTH-CORNER_STATION-WRAP_ARC_SEGMENTS*WRAP_CHORD)*1000,
        'elastic_model':'EI/dual_length; E=200 GPa assumed; plastic forming uncalibrated',
        'body_masses_kg':masses,'total_mass_kg':sum(masses.values()),
        'product_cad':False,'measured_product_mass_kg':None,'material_response_calibrated':False,
        'scope':'Supported prebending prototype; final crimp and unsupported retention are outside the task.'}
    (folder/'geometry.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
