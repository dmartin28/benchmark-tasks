"""Contact-gated dry pinch for native cloth, including a fingertip pinch-gather of a flat rag.

PhysX surface cloth has dynamic but no static friction, so neither a pinch of
cloth already between the pads nor a fingertip pinch-gather of a flat rag can
hold by friction alone. This rule stands in for the static friction the
simulator lacks. It applies identically to every policy (scripted demo or Dream).

Two ways to engage, both only from measured contact, never from task landmarks:

1. Between the pads (as in the accepted task `peg-out-a-tshirt`): when the fingers
   close below 4 mm with cloth nodes already between the pad faces, the pad-sized
   patch there is held.
2. Pinch-gather: the hand arms when the fingers are at least 25 mm apart and
   BOTH fingertips press on the cloth (a cloth node within 2 mm short of / 3 mm
   beyond each tip, under that pad's footprint at mesh resolution). It stays armed only while both tips keep contact
   (a lapse over 50 ms disarms). If the fingers then close below 4 mm, the
   12 mm patch of cloth under the closing point is held.

Opening past 8 mm releases. A held node more than 20 mm from its target (failed
binding) releases and blocks re-engaging until the hand opens. No node is moved
when the grip engages: held nodes keep their measured offsets from the hand, and
PhysX's native vertex attachment then moves cloth and hand together.

Adapted from `peg-out-a-tshirt` (pinch_constraint.py, MIT licence, see LICENSE).
"""
import numpy as np

ARM_GAP = 0.025       # fingers at least this far apart to arm a gather
CLOSED_GAP = 0.004    # engage at or below this gap
OPEN_GAP = 0.008      # release above this gap
PAD_HALF_W = 0.009    # pad half-width across the closing direction
PAD_DEPTH = 0.006     # pad thickness band along the closing direction
TIP = 0.013           # pad tips beyond the TCP along the approach axis
LAPSE_S = 0.05
PATCH_R = 0.012
MESH_HALF = 0.005     # half the cloth mesh spacing: a node this close to a pad's footprint means cloth under it


class GatherPinch:
    def __init__(self):
        self.indices = np.empty(0, dtype=int)
        self.local = None
        self.events = []
        self.time = 0.0
        self.armed = False
        self.arm_gap = 0.0
        self.lapse = 0.0
        self.blocked_until_open = False

    def _tip_contact(self, local, gap):
        """Both pads touching cloth at their tips: nodes under each pad's tip face."""
        near_tip = ((local[:, 2] >= TIP - 0.002) & (local[:, 2] <= TIP + 0.003) &
                    (abs(local[:, 1]) <= PAD_HALF_W + MESH_HALF))
        left = near_tip & (abs(local[:, 0] - gap / 2) <= PAD_DEPTH / 2 + MESH_HALF)
        right = near_tip & (abs(local[:, 0] + gap / 2) <= PAD_DEPTH / 2 + MESH_HALF)
        return left.any() and right.any()

    def update(self, points, tcp, rotation, gap, dt):
        self.time += dt
        p = np.asarray(points)
        local = (p - tcp) @ rotation
        if gap > OPEN_GAP:
            if len(self.indices):
                self.events.append(dict(time_s=self.time, event="opened", count=len(self.indices)))
            self.indices = np.empty(0, dtype=int)
            self.blocked_until_open = False
            # Arm or keep armed only while both fingertips press on the cloth.
            if self._tip_contact(local, gap):
                if not self.armed and gap >= ARM_GAP:
                    self.armed, self.arm_gap = True, gap
                    self.events.append(dict(time_s=self.time, event="gather_armed", gap_m=float(gap)))
                self.lapse = 0.0
            elif self.armed:
                self.lapse += dt
                if self.lapse > LAPSE_S:
                    self.armed = False
                    self.events.append(dict(time_s=self.time, event="gather_disarmed"))
            return
        if len(self.indices) or self.blocked_until_open:
            return self._check(p, tcp, rotation)
        if gap > CLOSED_GAP:
            # Closing through 4-8 mm: the gather must keep tip contact on the way.
            if self.armed and not self._tip_contact(local, max(gap, 0.002)):
                self.lapse += dt
                if self.lapse > LAPSE_S:
                    self.armed = False
                    self.events.append(dict(time_s=self.time, event="gather_disarmed"))
            return
        # 1. Cloth already between the pad faces (the T-shirt task's rule).
        between = ((abs(local[:, 0]) <= gap / 2 + 0.0012) & (abs(local[:, 1]) <= PAD_HALF_W) &
                   (local[:, 2] >= -0.029) & (local[:, 2] <= 0.011))
        idx = np.flatnonzero(between)
        how = "between_pads"
        if len(idx) < 3 and self.armed:
            # 2. Pinch-gather: the patch under the closing point, at the tips.
            under = (np.hypot(local[:, 0], local[:, 1]) <= PATCH_R) & \
                    (local[:, 2] >= TIP - 0.002) & (local[:, 2] <= TIP + 0.003)
            idx = np.flatnonzero(under)
            how = "gather"
        self.armed = False
        if len(idx) < 3:
            return
        centre = idx[np.argmin(np.linalg.norm(local[idx, :2], axis=1))]
        idx = idx[np.linalg.norm(p[idx] - p[centre], axis=1) <= PATCH_R]
        if len(idx) < 3:
            return
        self.indices, self.local = idx, local[idx].copy()
        self.events.append(dict(time_s=self.time, event="pinched", how=how, count=len(idx),
                                indices=idx.tolist(), gap_m=float(gap), arm_gap_m=float(self.arm_gap)))

    def _check(self, p, tcp, rotation):
        if not len(self.indices):
            return
        targets = self.local @ rotation.T + tcp
        err = float(np.max(np.linalg.norm(targets - p[self.indices], axis=1)))
        if err > 0.020:
            self.events.append(dict(time_s=self.time, event="slipped", error_m=err))
            self.indices = np.empty(0, dtype=int)
            self.blocked_until_open = True


def install(base, cloth_name):
    """Native vertex attachment per hand, enabled only while a pinch is held. Two-way coupled."""
    import omni.usd
    from pxr import Sdf, UsdPhysics, Vt

    from cloth_bridge import rotation_xyzw
    from yam_arena.constants import FINGER_JOINTS, TCP_OFFSET
    from yam_arena.embodiment import SIDES
    stage = omni.usd.get_context().get_stage()
    cloth = base.scene[cloth_name]
    found = [str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).endswith("/" + cloth_name)
             and "OmniPhysicsDeformableBodyAPI" in p.GetAppliedSchemas()]
    if len(found) != 1:
        raise RuntimeError("Expected one deformable body for " + cloth_name)
    cloth_path, hands = found[0], []
    for side in ("left", "right"):
        robot = base.scene[side + "_robot"]
        arm_prim = SIDES[side]["prim"]
        paths = [str(p.GetPath()) for p in stage.Traverse() if "/" + arm_prim + "/" in str(p.GetPath())
                 and str(p.GetPath()).endswith("/gripper") and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(paths) != 1:
            raise RuntimeError("Expected one gripper link for " + side)
        prim = stage.DefinePrim(cloth_path + "_" + side + "_grip", "OmniPhysicsVtxXformAttachment")
        prim.CreateRelationship("omniphysics:src0").SetTargets([Sdf.Path(cloth_path)])
        prim.CreateRelationship("omniphysics:src1").SetTargets([Sdf.Path(paths[0])])
        enabled = prim.CreateAttribute("omniphysics:attachmentEnabled", Sdf.ValueTypeNames.Bool)
        enabled.Set(False)
        indices = prim.CreateAttribute("omniphysics:vtxIndicesSrc0", Sdf.ValueTypeNames.IntArray)
        indices.Set([])
        positions = prim.CreateAttribute("omniphysics:localPositionsSrc1", Sdf.ValueTypeNames.Point3fArray)
        positions.Set([])
        hands.append((side, robot, list(robot.body_names).index("gripper"),
                      robot.find_joints(list(FINGER_JOINTS), preserve_order=True)[0], GatherPinch(),
                      enabled, indices, positions))
    original = base.scene.write_data_to_sim

    def array(value):
        return getattr(value, "torch", value).detach().cpu().numpy()

    def apply(*args, **kwargs):
        result = original(*args, **kwargs)
        points = array(cloth.data.nodal_pos_w)[0]
        for side, robot, body, fingers, pinch, enabled, indices, positions in hands:
            pose = array(robot.data.body_link_pose_w)[0, body]
            R = rotation_xyzw(pose[3:])
            tcp = pose[:3] + R @ np.asarray(TCP_OFFSET)
            gap = float(abs(array(robot.data.joint_pos)[0, fingers]).sum())
            was = bool(len(pinch.indices))
            pinch.update(points.copy(), tcp, R, gap, float(base.physics_dt))
            now = bool(len(pinch.indices))
            if now and not was:
                indices.Set(pinch.indices.tolist())
                positions.Set(Vt.Vec3fArray((pinch.local + np.asarray(TCP_OFFSET)).tolist()))
                enabled.Set(True)
            elif was and not now:
                enabled.Set(False)
        return result

    base.scene.write_data_to_sim = apply
    return {side: p for side, _, _, _, p, _, _, _ in hands}
