"""Explicit accelerated-curing approximation; a native, breakable wood-to-wood joint.

Only a measured long bead and two seconds of stationary aligned contact enable adhesion.
The frames use the current poses, so enabling adhesion never moves either piece into place.
This is not a model of real-time PVA curing or calibrated wood-joint strength.
"""
from .assembly import ContactDwell,FACE_CENTER,wet_contact_length
from .scene import HEAD,LEG


class AdhesiveBond:
    def __init__(self,emb):
        import omni.physx
        from omni.physx.bindings._physx import SimulationEvent
        from pxr import PhysicsSchemaTools

        self.emb = emb
        self.path = '/World/WoodGlueAdhesiveBond'
        self.created = self.broken = False
        self.contact_s = 0.
        self.dwell = ContactDwell()
        self.created_s = None
        self.wet_contact_mm = 0.

        def event(evt):
            if evt.type == int(SimulationEvent.JOINT_BREAK):
                pair = evt.payload['jointPath']
                if str(PhysicsSchemaTools.decodeSdfPath(pair[0],pair[1])) == self.path:
                    self.broken = True
        self.subscription = omni.physx.get_physx_interface().get_simulation_event_stream_v2().create_subscription_to_pop(event)

    @property
    def alive(self):
        return self.created and not self.broken

    def update(self,model,poses,dt):
        if self.created:
            return
        self.wet_contact_mm = wet_contact_length(model,poses)*1000 if model.joint['seated'] else 0.
        touching = model.coating_score() == .5 and model.joint['seated'] and self.wet_contact_mm >= 40.
        self.contact_s = self.dwell.advance(dt,touching,poses)
        if self.contact_s < 2.-1e-9:
            return
        from pxr import Gf,Sdf,UsdPhysics
        from yam_arena.ik import matrix_to_quat_xyzw
        stage = self.emb._base.sim.stage
        paths = self.emb._wood_glue_renderer.roots
        joint = UsdPhysics.FixedJoint.Define(stage,self.path)
        head,leg = poses[HEAD.name],poses[LEG.name]
        anchor = .5*(head.world(FACE_CENTER)+leg.world(FACE_CENTER))
        frame = head.rot
        for i,(name,pose) in enumerate(((HEAD.name,head),(LEG.name,leg))):
            getattr(joint,f'CreateBody{i}Rel')().SetTargets([Sdf.Path(paths[name])])
            getattr(joint,f'CreateLocalPos{i}Attr')().Set(Gf.Vec3f(*map(float,pose.local(anchor))))
            x,y,z,w = map(float,matrix_to_quat_xyzw(pose.rot.T@frame))
            getattr(joint,f'CreateLocalRot{i}Attr')().Set(Gf.Quatf(w,Gf.Vec3f(x,y,z)))
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateBreakForceAttr().Set(50.)
        joint.CreateBreakTorqueAttr().Set(1.)
        self.created,self.created_s = True,model.time

    def metrics(self):
        return dict(accelerated_cure=True,contact_s=self.contact_s,cure_dwell_s=2.,
                    wet_contact_length_mm=self.wet_contact_mm,required_wet_contact_mm=40.,
                    stationary_translation_mm=.5,stationary_rotation_deg=.5,
                    created=self.created,broken=self.broken,alive=self.alive,created_s=self.created_s,
                    break_force_n=50.,break_torque_nm=1.,calibrated=False)

    def clear(self):
        self.emb._base.sim.stage.RemovePrim(self.path)
        self.subscription = None
