"""Auxiliary still camera; never substitutes for the native policy top camera."""
import numpy as np


class PreviewCamera:
    def __init__(self,emb,width,height):
        import omni.replicator.core as rep
        from pxr import Gf,Sdf,UsdGeom
        self.emb=emb
        self.shape=(height,width,3)
        self.camera=UsdGeom.Camera.Define(emb._base.sim.stage,'/World/NailScenePreview')
        self.transform=self.camera.AddTransformOp()
        self.camera.CreateFocalLengthAttr(24.)
        self.camera.CreateHorizontalApertureAttr(20.955)
        self.camera.CreateVerticalApertureAttr(20.955*height/width)
        self.camera.CreateClippingRangeAttr(Gf.Vec2f(.005,30.))
        self.camera.GetPrim().CreateAttribute('omni:scenePartition',Sdf.ValueTypeNames.Token).Set('env_0')
        self.product=rep.create.render_product(str(self.camera.GetPath()),(width,height))
        self.rgb=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu')
        self.rgb.attach([self.product])

    def aim(self,eye,target,focal):
        from pxr import Gf
        self.transform.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0.,0.,1.)).GetInverse())
        self.camera.GetFocalLengthAttr().Set(focal)
        for _ in range(8):
            self.emb._base.sim.render()
            self._sync()

    @staticmethod
    def _sync():
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        ensure_isaac_rtx_render_update(force=True)

    def frame(self):
        self._sync()
        frame=np.ascontiguousarray(self.rgb.get_data()[...,:3])
        if frame.shape!=self.shape:raise RuntimeError(f'Bad preview image shape: {frame.shape}')
        return frame

    def close(self):
        self.rgb.detach()
        self.product.destroy()
