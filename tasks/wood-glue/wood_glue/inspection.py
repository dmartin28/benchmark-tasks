"""An auxiliary native-rendered view of the held cut face, separate from robot observations."""
import numpy as np

EYE = (.51,.13,1.34)
TARGET = (.277,-.075,1.020)
ASSEMBLY_EYE = (.62,.35,1.22)
ASSEMBLY_TARGET = (.36,0.,.76)


class InspectionCamera:
    def __init__(self,emb,width,height):
        import omni.replicator.core as rep
        from pxr import Gf,Sdf,UsdGeom

        self.emb = emb
        self.shape = (height,width,3)
        camera = UsdGeom.Camera.Define(emb._base.sim.stage,"/World/WoodGlueInspectionCamera")
        self.camera = camera
        self.assembly_view = False
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*EYE),Gf.Vec3d(*TARGET),Gf.Vec3d(0.,0.,1.)).GetInverse())
        camera.CreateFocalLengthAttr(40.)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955*height/width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.005,30.))
        camera.GetPrim().CreateAttribute("omni:scenePartition",Sdf.ValueTypeNames.Token).Set("env_0")
        self.product = rep.create.render_product(str(camera.GetPath()),(width,height))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb",device="cpu")
        self.rgb.attach([self.product])
        for _ in range(8):
            emb._base.sim.render()
            self._sync_render()

    def show_assembly(self):
        if self.assembly_view:
            return
        from pxr import Gf
        self.camera.GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*ASSEMBLY_EYE),Gf.Vec3d(*ASSEMBLY_TARGET),Gf.Vec3d(0.,0.,1.)).GetInverse())
        self.camera.GetFocalLengthAttr().Set(24.)
        self.assembly_view = True
        self.emb._base.sim.render()

    def show_bottle(self):
        from pxr import Gf
        from .scene import BOTTLE_BASE,TIP_LOCAL
        target = BOTTLE_BASE+np.array([0.,0.,TIP_LOCAL[2]/2])
        eye = BOTTLE_BASE+np.array([.38,-.30,.29])
        self.camera.GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*map(float,eye)),Gf.Vec3d(*map(float,target)),Gf.Vec3d(0.,0.,1.)).GetInverse())
        self.camera.GetFocalLengthAttr().Set(24.)
        for _ in range(4):
            self.emb._base.sim.render()
            self._sync_render()

    @staticmethod
    def _sync_render():
        # The Arena renderer updates its registered sensors; this separate render product also
        # needs the native RTX update after being created and after each physics interval.
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        ensure_isaac_rtx_render_update(force=True)

    def frame(self):
        self._sync_render()
        frame = np.ascontiguousarray(self.rgb.get_data()[...,:3])
        if frame.shape != self.shape:
            raise RuntimeError(f"inspection camera returned {frame.shape}, expected {self.shape}")
        return frame

    def close(self):
        self.rgb.detach()
        self.product.destroy()
