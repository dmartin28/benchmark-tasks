"""Rounded, unbranded bottle skin; the moving side walls follow the real squeeze links."""
from pathlib import Path

import numpy as np

YELLOW = (.91,.89,.76)  # Natural LDPE, warm tint from yellow glue; opacity simplified.
ORANGE = (.96,.29,.035)
IVORY = (.94,.94,.89)
SEGMENTS = 64


def rings_mesh(rings,skip_wall_window=False,top_hole=0.):
    """Elliptic rings, with a mid-body opening for each compliant wall."""
    angles = np.arange(SEGMENTS)*2*np.pi/SEGMENTS
    points = np.array([[rx*np.cos(a),ry*np.sin(a),z] for z,rx,ry in rings for a in angles])
    faces = []
    for j in range(len(rings)-1):
        for k in range(SEGMENTS):
            mid_angle = (k+.5)*2*np.pi/SEGMENTS
            if skip_wall_window and rings[j][0] >= .024 and rings[j+1][0] <= .066 and abs(np.sin(mid_angle)) > 21/29:
                continue
            a,b = j*SEGMENTS+k,j*SEGMENTS+(k+1)%SEGMENTS
            c,d = a+SEGMENTS,b+SEGMENTS
            faces.extend([(a,b,d),(a,d,c)])
    for j,reverse in ((0,True),(len(rings)-1,False)):
        if not reverse and top_hole:
            inner=len(points)
            points=np.vstack([points,[[top_hole*np.cos(a),top_hole*np.sin(a),rings[j][0]] for a in angles]])
            for k in range(SEGMENTS):
                a,b=j*SEGMENTS+k,j*SEGMENTS+(k+1)%SEGMENTS
                c,d=inner+k,inner+(k+1)%SEGMENTS
                faces.extend([(a,b,d),(a,d,c)])
            continue
        centre = len(points)
        points = np.vstack([points,[0.,0.,rings[j][0]]])
        for k in range(SEGMENTS):
            a,b = j*SEGMENTS+k,j*SEGMENTS+(k+1)%SEGMENTS
            faces.append((centre,b,a) if reverse else (centre,a,b))
    return points,np.array(faces,int)


def bottle_meshes():
    from .scene import BOTTLE_RADII, BOTTLE_UNCAPPED_HEIGHT, NOZZLE_DIAMETER
    scale_xy = BOTTLE_RADII/np.array([.0175,.029])
    shell = [(0.,.015,.025),(.002,.0175,.029),(.024,.0175,.029),
             (.066,.0175,.029),(.071,.0175,.028),(.078,.015,.023),(.084,.0105,.0105)]
    out = [('body','shell',*rings_mesh(shell,True),YELLOW)]
    # Side tips of the oval, integrated into the wall rather than rectangular external pads.
    for sign,name in ((1,'panel_left'),(-1,'panel_right')):
        theta = np.linspace(np.arcsin(20/29),np.pi-np.arcsin(20/29),19)
        verts = np.array([[.0175*np.cos(t),sign*(.029*np.sin(t)-.025),z-.045]
                          for z in (.024,.066) for t in theta])
        n = len(theta)
        faces = [(k,k+1,n+k+1) for k in range(n-1)]+[(k,n+k+1,n+k) for k in range(n-1)]
        if sign < 0: faces = [f[::-1] for f in faces]
        out.append((name,'flexible_wall',verts,np.array(faces,int),YELLOW))
    # Scale the shell and moving walls together, retaining their local link origins.
    for _,_,points,_,_ in out:
        points[:,:2] *= scale_xy
    # The uncapped 18 mm neck is hidden beneath the estimated mounted cap.
    out.append(('body','neck',*rings_mesh([(.083,.009,.009),(BOTTLE_UNCAPPED_HEIGHT,.009,.009)]),IVORY))
    out.append(('body','collar',*rings_mesh([(.08225,.0105,.0105),(.083,.011,.011),
                                           (.095,.011,.011),(.096,.009,.009)]),IVORY))
    for i,z in enumerate(np.linspace(.0835,.094,7)):
        out.append(('body',f'cap_rib_{i}',*rings_mesh([(z,.0112,.0112),(z+.0006,.0112,.0112)]),IVORY))
    out.append(('body','tapered_nozzle',*rings_mesh([(.096,.0057,.0057),(.100,.0045,.0045),
                                                   (.110,.0012,.0012),(.114,.0007,.0007)],top_hole=NOZZLE_DIAMETER/2),IVORY))
    # The removable red protector is off the nozzle before execution.
    radius=NOZZLE_DIAMETER/2
    out.append(('body','open_round_outlet',*rings_mesh([(.11397,radius,radius),(.11398,radius,radius)]),(.12,.11,.09)))
    from .scene import BOTTLE_BODY_EXTENSION
    # Extend the lower body; retain the neck, nozzle and compliant wall geometry.
    for parent,name,points,faces,color in out:
        if parent == 'body':
            if name == 'shell': points[:,2] += np.where(points[:,2] > .002,BOTTLE_BODY_EXTENSION,0.)
            else: points[:,2] += BOTTLE_BODY_EXTENSION
    return out


def install(stage,root,spec):
    from pxr import Gf,Sdf,UsdGeom,UsdShade
    from yam_arena.station import linear_rgb

    # Hide only collision geometry: new meshes remain visible and all original colliders stay active.
    for parent,count in [('body',len(spec.boxes)),*[(s.name,1) for s in spec.sliders]]:
        for i in range(count):
            UsdGeom.Imageable(stage.GetPrimAtPath(f'{root}/{parent}/box_{i}')).MakeInvisible()

    def mesh(parent,name,points,faces,color):
        m = UsdGeom.Mesh.Define(stage,f'{root}/{parent}/{name}')
        m.CreatePointsAttr(points.tolist())
        m.CreateFaceVertexCountsAttr([3]*len(faces))
        m.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        m.CreateSubdivisionSchemeAttr('none')
        m.CreateDoubleSidedAttr(True)
        m.CreateDisplayColorAttr([Gf.Vec3f(*linear_rgb(color))])
        # Smooth normals on the oval sections and shoulders.
        normals = np.zeros_like(points)
        face_n = np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
        for corner in range(3): np.add.at(normals,faces[:,corner],face_n)
        normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        m.CreateNormalsAttr(normals.tolist())
        m.SetNormalsInterpolation('vertex')
        return m

    for parent,name,points,faces,color in bottle_meshes():
        mesh(parent,name,points,faces,color)

    material = UsdShade.Material.Define(stage,root+'/Looks/label')
    shader = UsdShade.Shader.Define(stage,str(material.GetPath())+'/surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.7)
    texture = UsdShade.Shader.Define(stage,str(material.GetPath())+'/texture')
    texture.CreateIdAttr('UsdUVTexture')
    label = Path(__file__).resolve().parents[1]/'assets'/'bottle-label.png'
    texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(label)))
    texture.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    reader = UsdShade.Shader.Define(stage,str(material.GetPath())+'/uv')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    for sign in (-1,1):
        angles = np.linspace(-.66,.66,25)
        pts = np.array([[sign*.01762*np.cos(t),.02912*np.sin(t),z] for z in (.023,.066) for t in angles])
        from .scene import BOTTLE_RADII
        pts[:,:2] *= BOTTLE_RADII/np.array([.0175,.029])
        from .scene import BOTTLE_BODY_EXTENSION
        pts[:,2] += BOTTLE_BODY_EXTENSION
        n = len(angles)
        faces = np.array([(k,k+1,n+k+1) for k in range(n-1)]+[(k,n+k+1,n+k) for k in range(n-1)])
        m = mesh('body','printed_label_front' if sign > 0 else 'printed_label_back',pts,faces,(1.,1.,1.))
        uv = [(float(i/(n-1)),float(j)) for j in (0,1) for i in range(n)]
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex)
        st.Set(uv)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
