"""Feedback-assisted scripted controller; sends only ordinary 14-D robot actions."""
from __future__ import annotations

import numpy as np

from yam_arena import ik, station
from yam_arena.constants import READY, GRIPPER
from yam_arena.demo import grip_rotation

from .geometry import Pose,yaw_matrix
from .scene import (BOTTLE_BASE, BOTTLE_YAW, BOTTLE_PANEL_RATE, BOTTLE_RADII, FACE_LENGTH, GRASP_LOCAL, HEAD, LEG,
                    THICK, TIP_LOCAL, TOP, all_initial_poses, surfaces)

HZ = 20
PICK_ROT = np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
PICKUP_ROT = yaw_matrix(BOTTLE_YAW) @ PICK_ROT
INVERTED_ROT = PICKUP_ROT @ np.diag([-1.,-1.,1.])
BOTTLE_COAT_ROT = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
COAT_ROT = BOTTLE_COAT_ROT @ PICK_ROT
REST_GAP = 2*BOTTLE_RADII[1]
CARRY_COMPRESSION = .003
GRIP_FACTOR = 1.+BOTTLE_PANEL_RATE/GRIPPER["stiffness"]
LEG_GRASP_LOCAL = np.array([.18,0.,THICK-.0005])
LEG_GRIP_YAW = LEG.yaw-np.pi
# Tilt the cut toward the standard camera and keep the bottle wrist outboard of the bead.
PRESENT_LEFT = np.array([-1.2681798300222615, 2.7260768112602154, 2.116217103838175, 0.8116950144888878, -1.2684094359944589, -0.693017920086157])
PRESENT_RIGHT = np.array([0.07543394079737151, 1.8087231724501458, 1.8198278665000727, -1.3457117805421714, -1.4645994228593875, 1.7533750862211726])


class Arm:
    def __init__(self,side,q=None):
        self.side = side
        self.base = station.BASES[side]
        self.q = np.array(READY if q is None else q,float)
        self.pos,self.rot = ik.tcp_pose(self.q,self.base)
        self.grip = 1.
        self.worst_mm = self.worst_deg = 0.
        self.rng = np.random.default_rng(43 if side=="right" else 45)

    @property
    def command(self):
        return np.r_[self.q,self.grip]

    def solve(self,pos,rot,restarts=False):
        best = None
        seeds = [self.q]
        if restarts:
            seeds += [np.asarray(READY), *[self.rng.uniform(ik.LOWER,ik.UPPER) for _ in range(18)]]
        for seed in seeds:
            q,pe,re = ik.solve(pos,ik.matrix_to_quat_xyzw(rot),seed,base_pos=self.base,
                               iters=120 if restarts else 60,damping=.015,tol=2e-5)
            if best is None or pe+.1*re < best[0]:
                best = (pe+.1*re,q,pe,re)
            if pe < .0001 and re < .001:
                break
        _,q,pe,re = best
        self.worst_mm = max(self.worst_mm,pe*1000)
        self.worst_deg = max(self.worst_deg,float(np.degrees(re)))
        return q,pe,re

    def move(self,pos,rot,duration,grip=None):
        pos = np.asarray(pos,float)
        p0,r0,g0 = self.pos.copy(),self.rot.copy(),self.grip
        g1 = g0 if grip is None else grip
        for i in range(1,round(duration*HZ)+1):
            s = ik.smoothstep(i/round(duration*HZ))
            p,r = p0+s*(pos-p0),ik.slerp_matrix(r0,rot,s)
            self.q,pe,re = self.solve(p,r)
            if pe > .002 or re > np.deg2rad(2):
                raise RuntimeError(f"{self.side}: unreachable move ({pe*1000:.2f} mm, {np.degrees(re):.2f} deg)")
            self.pos,self.rot,self.grip = p,r,g0+s*(g1-g0)
            yield

    def joints_to_pose(self,pos,rot,duration,grip=None):
        """Joint interpolation to enter a different IK branch; audit native contacts separately."""
        q,pe,re = self.solve(pos,rot,restarts=True)
        if pe > .001 or re > np.deg2rad(1):
            raise RuntimeError(f"{self.side}: no reachable endpoint ({pe*1000:.2f} mm, {np.degrees(re):.2f} deg)")
        q0,g0 = self.q.copy(),self.grip
        for i in range(1,round(duration*HZ)+1):
            s = ik.smoothstep(i/round(duration*HZ))
            self.q = q0+s*(q-q0)
            self.pos,self.rot = ik.tcp_pose(self.q,self.base)
            if grip is not None: self.grip = g0+s*(grip-g0)
            yield

    def move_joints(self,target,duration):
        target = np.asarray(target,float)
        if np.any(target < ik.LOWER) or np.any(target > ik.UPPER):
            raise ValueError("joint target outside URDF limits")
        start = self.q.copy()
        for i in range(1,round(duration*HZ)+1):
            self.q = start+ik.smoothstep(i/round(duration*HZ))*(target-start)
            self.pos,self.rot = ik.tcp_pose(self.q,self.base)
            yield

    def aim_tool(self,tip,axis,tool_rotation,tip_offset):
        """Five-DOF IK: tool tip and direction, leaving rotation about its axis free."""
        from scipy.optimize import least_squares
        tip,axis = np.asarray(tip,float),np.asarray(axis,float)
        def residual(q):
            p,r = ik.tcp_pose(q,self.base)
            return np.r_[p+r@tip_offset-tip,.2*(r@tool_rotation[:,2]-axis)]
        result = least_squares(residual,np.clip(self.q,ik.LOWER+1e-5,ik.UPPER-1e-5),
                               bounds=(ik.LOWER+1e-5,ik.UPPER-1e-5),max_nfev=45,
                               ftol=1e-8,xtol=1e-8,gtol=1e-8)
        self.q = result.x
        self.pos,self.rot = ik.tcp_pose(self.q,self.base)
        error = residual(self.q)
        pe,ae = np.linalg.norm(error[:3]),np.linalg.norm(error[3:])/.2
        self.worst_mm = max(self.worst_mm,pe*1000)
        self.worst_deg = max(self.worst_deg,float(np.rad2deg(ae)))
        # Small grip deflections need not prevent deposition: a 5-degree angular residual spans
        # under 0.7 mm across the 8 mm lip. The actual emitted glue still has to land on the wood.
        if pe > .002 or ae > np.deg2rad(5):
            raise RuntimeError(f"{self.side}: tool target unreachable ({pe*1000:.2f} mm, axis {np.rad2deg(ae):.2f} deg)")


def normalized_grip(compression):
    # Slight overtravel supplies contact force; actual spring displacement determines flow.
    return (REST_GAP-GRIP_FACTOR*compression)/.0939


class Controller:
    def __init__(self,emb=None,mode="pickup-flip",accelerated_bond=False):
        self.emb,self.mode = emb,mode
        self.accelerated_bond = accelerated_bond
        self.left,self.right = Arm("left"),Arm("right")
        self.phase = "start"
        self.done = False
        self.steps = 0
        self.tip_bias = np.zeros(3)
        self._offline_objects = all_initial_poses()
        self._offline_grasps = {}
        self.object_bias = {}
        self.object_rot_bias = {}
        self.proof_lift_mm = None
        self.join_motion = None
        self.wood_contacts = {}
        self.arm_phases = {}
        self.generator = self.script()

    def poses(self):
        if self.emb is None:
            poses = self._offline_objects.copy()
            for side,(name,p,r) in self._offline_grasps.items():
                arm = self.left if side == "left" else self.right
                poses[name] = Pose(arm.pos+arm.rot@p,arm.rot@r)
            return poses
        from .runtime import measured_poses
        return measured_poses(self.emb)[0]

    def wait(self,seconds):
        yield from (None for _ in range(round(seconds*HZ)))

    def note_grasp(self,side,name):
        # Ideal kinematic prediction for CPU trajectory checks only. Native grasps use contacts.
        if self.emb is None:
            arm = self.left if side == "left" else self.right
            pose = self.poses()[name]
            self._offline_grasps[side] = (name,arm.rot.T@(pose.pos-arm.pos),arm.rot.T@pose.rot)

    def note_release(self,side):
        if self.emb is None and side in self._offline_grasps:
            name = self._offline_grasps[side][0]
            self._offline_objects[name] = self.poses()[name]
            del self._offline_grasps[side]

    def compression(self):
        if self.emb is None:
            return max(0.,(REST_GAP-.0939*self.right.grip)/GRIP_FACTOR)
        from .runtime import measured_poses
        bottle = measured_poses(self.emb)[1]["glue_bottle"]
        return -bottle.sliders["panel_left"]+bottle.sliders["panel_right"]

    def script(self):
        if self.mode == 'pickup-flip':
            yield from self.pick_bottle()
        else:
            yield from self.parallel_named('pick up bottle and loose miter concurrently',
                                           right=self.pick_bottle(),left=self.pick_leg())
            yield from self.hold_and_glue()
            if self.mode == 'demo':
                yield from self.assemble()
        self.phase = 'finished'

    def parallel_named(self,label,**motions):
        active = {side:iter(motion) for side,motion in motions.items()}
        phases = {side:'start' for side in active}
        while active:
            for side in list(active):
                self.phase = phases[side]
                try:
                    next(active[side])
                    phases[side] = self.phase
                except StopIteration:
                    del active[side]
                    phases[side] = 'holding staged object'
            self.arm_phases = phases.copy()
            self.phase = label
            if active: yield
        self.arm_phases = {}

    def pick_bottle(self):
        R = self.right
        self.phase = 'approach bottle from side'
        grasp = np.asarray(BOTTLE_BASE)+GRASP_LOCAL
        approach = -PICKUP_ROT[:,2]*.060
        yield from R.joints_to_pose(grasp+approach+[0.,0.,.06],PICKUP_ROT,3.,grip=1.)
        yield from R.move(grasp+approach,PICKUP_ROT,1.2)
        yield from R.move(grasp,PICKUP_ROT,1.2)
        self.phase = 'grasp bottle without dispensing'
        yield from R.move(grasp,PICKUP_ROT,1.,grip=normalized_grip(CARRY_COMPRESSION))
        yield from self.wait(.4)
        self.note_grasp('right','glue_bottle')
        self.phase = 'lift bottle clear of table'
        yield from R.move(grasp+[0.,0.,.08],PICKUP_ROT,2.)
        self.bottle_lift_q = R.q.copy()
        self.phase = 'turn bottle upside down'
        inverted = Arm('right',[.1,1.8,1.8,-1.3,-1.5,1.7])
        inverted.aim_tool([.40,-.48,TOP+.25],[0.,0.,-1.],*self.tool_calibration())
        self.inverted_q = inverted.q.copy()
        yield from R.move_joints(self.inverted_q,4.)
        self.phase = 'hold bottle upside down'
        yield from self.wait(2. if self.mode == 'pickup-flip' else .3)

    def pick_leg(self):
        L = self.left
        pose = self.poses()[LEG.name]
        grip = pose.world(LEG_GRASP_LOCAL)
        rot = grip_rotation(LEG_GRIP_YAW)
        self.phase = 'approach loose miter'
        yield from L.move(grip+[0.,0.,.08],rot,2.5,grip=1.)
        yield from L.move(grip,rot,1.5)
        self.phase = 'grasp loose miter'
        self.wood_contacts['left'] = LEG.name
        yield from L.move(grip,rot,1.,grip=.053/.0939)
        yield from self.wait(.5)
        self.note_grasp('left',LEG.name)
        self.phase = 'lift loose miter'
        yield from L.move(grip+[0.,0.,.09],rot,3.)
        self.phase = 'raise miter for turning clearance'
        yield from L.move([.24,.15,TOP+.22],rot,2.)

    def tool_calibration(self):
        """Measured bottle-to-hand transform; no artificial grasp attachment is created."""
        if self.emb is None:
            return PICK_ROT.T,PICK_ROT.T@(TIP_LOCAL-GRASP_LOCAL)
        from .runtime import arr
        from yam_arena.constants import ARM_JOINTS
        robot = self.emb._base.scene["right_robot"]
        q = dict(zip(robot.joint_names,arr(robot.data.joint_pos)[0]))
        tcp,r = ik.tcp_pose(np.array([q[n] for n in ARM_JOINTS]),station.BASES["right"])
        pose = self.poses()["glue_bottle"]
        return r.T@pose.rot,r.T@(pose.world(TIP_LOCAL)-tcp)

    def held_leg_pose(self):
        return self.poses()[LEG.name]

    def glue_target(self,surface,uv,offset=0.):
        pose = self.held_leg_pose()
        normal = pose.rot@surface.normal
        point = pose.world(surface.origin+uv[0]*surface.u+uv[1]*surface.v)
        return point+normal*offset,-normal

    def tool_move(self,point,axis,seconds,calibration):
        start = self.right.pos+self.right.rot@calibration[1]
        axis0 = self.right.rot@calibration[0][:,2]
        for i in range(1,round(seconds*HZ)+1):
            s = ik.smoothstep(i/round(seconds*HZ))
            direction = (1-s)*axis0+s*axis
            direction /= np.linalg.norm(direction)
            self.track_tool(start+s*(point-start),direction,calibration)
            yield

    def track_tool(self,point,axis,calibration):
        """Compensate measured grasp drift and joint tracking error without moving objects directly."""
        if self.emb is not None:
            actual = self.poses()["glue_bottle"].world(TIP_LOCAL)
            predicted = self.right.pos+self.right.rot@calibration[1]
            error = predicted-actual
            norm = np.linalg.norm(error)
            if norm > .006:
                error *= .006/norm
            self.tip_bias = .8*self.tip_bias+.2*error
        self.right.aim_tool(np.asarray(point)+self.tip_bias,axis,*calibration)

    def hold_and_glue(self):
        L,R = self.left,self.right
        self.phase = "present cut face upward"
        yield from L.move_joints(PRESENT_LEFT,4.)
        yield from self.wait(1.)
        surface = next(s for s in surfaces() if s.owner == LEG.name and s.target)
        self.phase = "bring bottle above held miter"
        # Enter the precomputed shared-workspace IK branch with the nozzle above the face.
        calibration = self.tool_calibration()
        point,axis = self.glue_target(surface,[FACE_LENGTH/2,.008],.025)
        approach = Arm("right",PRESENT_RIGHT)
        approach.aim_tool(point,axis,*calibration)
        yield from R.move_joints(approach.q,4.)
        yield from self.wait(.5)
        calibration = self.tool_calibration()
        # Work from the actual held face, not the initial board pose or a scripted coating mask.
        point,axis = self.glue_target(surface,[FACE_LENGTH/2,.008],.025)
        yield from self.tool_move(point,axis,2.,calibration)
        # A single metered bead along the center of the cut. Broad film coverage is unnecessary.
        for row,v in enumerate((.008,)):
            a,b = FACE_LENGTH-.005,.005
            if row % 2: a,b = b,a
            self.phase = f"approach held edge row {row+1}"
            point,axis = self.glue_target(surface,[a,v],.012)
            yield from self.tool_move(point,axis,1.5,calibration)
            point,axis = self.glue_target(surface,[a,v])
            yield from self.tool_move(point,axis,1.5,calibration)
            self.phase = "apply single glue bead"
            # Bring the measured panel compression to the dispensing deadband without a free dose.
            yield from R.move(R.pos,R.rot,.7,grip=normalized_grip(.004))
            command_c = .004
            for _ in range(12):
                # Real finger/collider compliance shifts the stroke zero. Approach the measured
                # dispensing threshold gently, while the nozzle is already over the cut face.
                command_c = float(np.clip(command_c+.5*(.004-self.compression()),.0035,.0052))
                R.grip = normalized_grip(command_c)
                yield
            baseline_c = command_c
            actual_c0 = self.compression()
            if self.emb is not None and actual_c0 < .00398:
                raise RuntimeError("bottle panels did not reach the dispensing threshold")
            emitted0 = 0. if self.emb is None else self.emb._wood_glue_model.emitted*1e6
            previous_demand = 0.
            amount_ml = .05
            for i in range(1,51):
                s = ik.smoothstep(i/50)
                point,axis = self.glue_target(surface,[a+s*(b-a),v])
                self.track_tool(point,axis,calibration)
                observed = (max(0.,self.compression()-actual_c0)*100. if self.emb is None else
                            self.emb._wood_glue_model.emitted*1e6-emitted0)
                corrected = baseline_c+s*amount_ml/100.+.5*(previous_demand-observed)/100.
                command_c = max(command_c,min(corrected,baseline_c+amount_ml/100.+.0002))
                R.grip = normalized_grip(command_c)
                previous_demand = s*amount_ml
                yield
            self.phase = "stop glue flow and lift nozzle"
            yield from R.move(R.pos,R.rot,.7,grip=normalized_grip(CARRY_COMPRESSION))
            point,axis = self.glue_target(surface,[b,v],.012)
            yield from self.tool_move(point,axis,1.,calibration)
        self.phase = "hold coated miter for inspection"
        point,axis = self.glue_target(surface,[FACE_LENGTH/2,.008],.030)
        yield from self.tool_move(point,axis,1.5,calibration)
        yield from self.wait(.3 if self.mode == "demo" else 2.)

    def object_calibration(self,side,name):
        """Actual object-to-TCP transform, estimated after contact has settled."""
        arm = self.left if side == "left" else self.right
        p,r = arm.pos,arm.rot
        if self.emb is not None:
            from .runtime import arr
            from yam_arena.constants import ARM_JOINTS
            robot = self.emb._base.scene[side+"_robot"]
            q = dict(zip(robot.joint_names,arr(robot.data.joint_pos)[0]))
            p,r = ik.tcp_pose(np.array([q[n] for n in ARM_JOINTS]),arm.base)
        obj = self.poses()[name]
        return r.T@(obj.pos-p),r.T@obj.rot

    def move_object(self,side,name,target,duration,calibration,feedback=False):
        arm = self.left if side == "left" else self.right
        offset,rotation = calibration
        rot = target.rot@rotation.T
        if feedback and self.emb is not None:
            # Compensate actual PD tracking and grip compliance. Re-estimating the rigid grasp
            # alone cannot remove a persistent TCP tracking error at a tight mating surface.
            start = self.poses()[name]
            p0,r0 = start.pos.copy(),start.rot.copy()
            bias = self.object_bias.get(name,np.zeros(3))
            rbias = self.object_rot_bias.get(name,np.eye(3))
            for i in range(1,round(duration*HZ)+1):
                s = ik.smoothstep(i/round(duration*HZ))
                observed = self.poses()[name]
                error = arm.pos+arm.rot@offset-observed.pos
                norm = np.linalg.norm(error)
                limit = .015
                if norm > limit: error *= limit/norm
                bias = .8*bias+.2*error
                rerror = arm.rot@rotation@observed.rot.T
                angle = np.arccos(np.clip((np.trace(rerror)-1)/2,-1.,1.))
                angular_limit = np.deg2rad(5 if name == "glue_bottle" else 2)
                if angle > angular_limit:
                    rerror = ik.slerp_matrix(np.eye(3),rerror,angular_limit/angle)
                rbias = ik.slerp_matrix(rbias,rerror,.2)
                object_rot = ik.slerp_matrix(r0,target.rot,s)
                command_rot = rbias@object_rot@rotation.T
                command_pos = p0+s*(target.pos-p0)-command_rot@offset+bias
                arm.q,pe,re = arm.solve(command_pos,command_rot)
                if pe > .002 or re > np.deg2rad(2):
                    raise RuntimeError(f'{side}: measured-object alignment target is unreachable')
                arm.pos,arm.rot = command_pos,command_rot
                self.object_bias[name],self.object_rot_bias[name] = bias,rbias
                yield
            return
        yield from arm.move(target.pos-rot@offset,rot,duration)

    def parallel(self,*motions):
        active = list(map(iter,motions))
        while active:
            remaining = []
            for motion in active:
                try:
                    next(motion)
                    remaining.append(motion)
                except StopIteration:
                    pass
            active = remaining
            if active: yield

    def press_on_table(self):
        """A single normal-only closing stroke; tangential alignment happens before this."""
        from .assembly import FACE_CENTER
        from .observable import joint_status
        L = self.left
        head0,leg0 = self.poses()[HEAD.name],self.poses()[LEG.name]
        normal = head0.rot@next(s for s in HEAD.mesh()[2] if s.target).normal
        normal[2] = 0.
        normal /= np.linalg.norm(normal)
        toward = -normal
        head_center = head0.world(FACE_CENTER)
        gap0 = float((leg0.world(FACE_CENTER)-head_center)@normal)
        geometry = joint_status(self.poses())
        # Lateral mismatch and face step are independent physical gauges. Leave
        # margin below the 1 mm final tip-mismatch limit without treating height
        # as lateral error or introducing a shear correction after wet contact.
        if geometry['tangent_error_mm'] > .75 or abs(geometry['angle_deg']-90.) > .15:
            raise RuntimeError('Align the separated cut faces before starting the normal press.')
        calibration = self.object_calibration('left',LEG.name)
        offset,grasp_rotation = calibration
        rotation = L.rot.copy()
        p0 = L.pos.copy()
        bias0 = p0+rotation@offset-leg0.pos
        bias = bias0.copy()
        rotation_bias = rotation@grasp_rotation@leg0.rot.T
        wet_reference = None
        self.join_motion = dict(initial_gap_mm=gap0*1000,commanded_tangent_max_mm=0.,
                                desired_tangent_max_mm=0.,wet_shear_max_mm=0.,bottom_plane_error_max_mm=0.)
        for tick in range(120):
            s = ik.smoothstep(min((tick+1)/60,1.))
            self.phase = 'slide miters straight together on table' if tick < 60 else 'hold normal pressure on table'
            poses = self.poses()
            head,leg = poses[HEAD.name],poses[LEG.name]
            # Regulate a fixed wood tangent/rotation, compensating servo tracking error at the TCP.
            # Those corrections hold the wood on its normal path rather than intentionally wiping it.
            raw = L.pos+L.rot@offset-leg.pos
            norm = np.linalg.norm(raw)
            if norm > .015: raw *= .015/norm
            bias = .8*bias+.2*raw
            rerror = L.rot@grasp_rotation@leg.rot.T
            angle = np.arccos(np.clip((np.trace(rerror)-1)/2,-1.,1.))
            if angle > np.deg2rad(2):
                rerror = ik.slerp_matrix(np.eye(3),rerror,np.deg2rad(2)/angle)
            rotation_bias = ik.slerp_matrix(rotation_bias,rerror,.2)
            head_shift = float((head.world(FACE_CENTER)-head_center)@toward)
            desired = leg0.pos+toward*(s*(gap0-.00005)+head_shift)
            rotation = rotation_bias@leg0.rot@grasp_rotation.T
            command = desired+bias-rotation@offset
            L.q,pe,re = L.solve(command,rotation)
            if pe > .002 or re > np.deg2rad(2):
                raise RuntimeError('The table press target is unreachable.')
            L.pos,L.rot = command,rotation
            delta = command-p0
            tangent = float(np.linalg.norm(delta-toward*(delta@toward))*1000)
            self.join_motion['commanded_tangent_max_mm'] = max(self.join_motion['commanded_tangent_max_mm'],tangent)
            d = desired-leg0.pos
            self.join_motion['desired_tangent_max_mm'] = max(self.join_motion['desired_tangent_max_mm'],
                float(np.linalg.norm(d-toward*(d@toward))*1000))
            bottom_error = max(float(np.abs(poses[spec.name].world(spec.mesh()[0][:4])[:,2]-TOP).max()*1000)
                               for spec in (HEAD,LEG))
            self.join_motion['bottom_plane_error_max_mm'] = max(self.join_motion['bottom_plane_error_max_mm'],bottom_error)
            if bottom_error > 1.:
                raise RuntimeError('Both pieces must rest flat on the table throughout the press.')
            relative = leg.world(FACE_CENTER)-head.world(FACE_CENTER)
            actual_gap = abs(float(relative@normal))
            lateral = relative-normal*(relative@normal)
            if actual_gap <= .00025:
                if wet_reference is None: wet_reference = lateral.copy()
                shear = float(np.linalg.norm(lateral-wet_reference)*1000)
                self.join_motion['wet_shear_max_mm'] = max(self.join_motion['wet_shear_max_mm'],shear)
                if shear > .5:
                    raise RuntimeError('The glued faces moved tangentially after making contact.')
            yield

    def assemble(self):
        from .assembly import ASSEMBLY_CENTER,ASSEMBLY_HEAD_YAW,FACE_CENTER,planar_mating_leg
        from .scene import BOTTLE_YAW
        L,R = self.left,self.right
        self.phase = 'clear nozzle before moving coated wood'
        surface = next(s for s in surfaces() if s.owner == LEG.name and s.target)
        calibration = self.tool_calibration()
        point,axis = self.glue_target(surface,[FACE_LENGTH/2,.008],.10)
        yield from self.tool_move(point,axis,1.,calibration)
        self.phase = 'move coated wood clear of bottle arm'
        pose = self.poses()[LEG.name]
        calibration = self.object_calibration('left',LEG.name)
        yield from self.move_object('left',LEG.name,Pose(pose.pos+[0.,.20,0.],pose.rot),1.5,calibration)
        self.phase = "turn coated leg flat for assembly"
        q,pe,re = L.solve([.24,.15,TOP+.22],grip_rotation(LEG_GRIP_YAW+np.pi/2),restarts=True)
        if pe > .001 or re > .02: raise RuntimeError("no reachable flat assembly staging pose")
        yield from L.move_joints(q,4.)
        hr = yaw_matrix(ASSEMBLY_HEAD_YAW)
        head_target = Pose(ASSEMBLY_CENTER-hr@FACE_CENTER,hr)
        leg_target = planar_mating_leg(head_target,gap=.025)
        calibration = self.object_calibration('left',LEG.name)
        self.phase = 'place coated casing flat on table'
        yield from self.move_object('left',LEG.name,Pose(leg_target.pos+[0.,0.,.08],leg_target.rot),2.,calibration)
        yield from self.move_object('left',LEG.name,leg_target,2.,calibration,feedback=True)
        yield from self.wait(.3)

        self.phase = "return bottle upright"
        yield from R.move_joints(self.inverted_q,2.5)
        yield from R.move_joints(self.bottle_lift_q,4.)
        yield from self.wait(.4)
        calibration = self.object_calibration("right","glue_bottle")
        target = Pose(BOTTLE_BASE+[0.,0.,.08],yaw_matrix(BOTTLE_YAW))
        yield from self.move_object("right","glue_bottle",target,1.,calibration,feedback=True)
        self.phase = "park bottle"
        yield from self.move_object("right","glue_bottle",Pose(BOTTLE_BASE,yaw_matrix(BOTTLE_YAW)),3.,calibration,feedback=True)
        yield from self.move_object("right","glue_bottle",Pose(BOTTLE_BASE,yaw_matrix(BOTTLE_YAW)),1.,calibration,feedback=True)
        yield from self.wait(.4)
        yield from R.move(R.pos,R.rot,.6,grip=1.)
        self.note_release("right")
        yield from R.move(R.pos-PICKUP_ROT[:,2]*.065,R.rot,1.)
        self.phase = 'clear parked bottle before changing wrist orientation'
        yield from R.move(R.pos+[0.,0.,.12],R.rot,1.2)

        self.phase = "pick up head casing from table"
        head = self.poses()[HEAD.name]
        grip = head.world(LEG_GRASP_LOCAL)
        head_rot = grip_rotation(HEAD.yaw-np.pi)
        parked_before = self.poses()["glue_bottle"].pos.copy()
        yield from R.joints_to_pose(grip+[0.,0.,.08],head_rot,2.,grip=1.)
        if self.emb is not None:
            bottle = self.poses()['glue_bottle']
            if np.linalg.norm(bottle.pos[:2]-parked_before[:2]) > .005 or bottle.rot[2,2] < .95:
                raise RuntimeError('The approach to the head disturbed the parked bottle.')
        yield from R.move(grip,head_rot,1.5)
        self.wood_contacts["right"] = HEAD.name
        yield from R.move(grip,head_rot,.7,grip=.053/.0939)
        yield from self.wait(.4)
        self.note_grasp('right',HEAD.name)
        yield from R.move(grip+[0.,0.,.06],head_rot,1.5)
        calibration = self.object_calibration('right',HEAD.name)
        self.phase = 'place head casing flat on table'
        yield from self.move_object('right',HEAD.name,Pose(head_target.pos+[0.,0.,.06],hr),2.,calibration)
        yield from self.move_object('right',HEAD.name,head_target,2.,calibration,feedback=True)
        calibration = self.object_calibration('right',HEAD.name)
        yield from self.move_object('right',HEAD.name,head_target,1.,calibration,feedback=True)
        yield from self.wait(.4)
        self.phase = 'align separated miters on table'
        for _ in range(4):
            target = planar_mating_leg(self.poses()[HEAD.name],gap=.025)
            calibration = self.object_calibration('left',LEG.name)
            yield from self.move_object('left',LEG.name,target,1.,calibration,feedback=True)
        # Approach while dry, then let servo/friction transients settle before wet contact.
        # The final 4 mm push uses the same normal-only closing law at much lower speed.
        self.phase = 'approach aligned miters before final press'
        calibration = self.object_calibration('left',LEG.name)
        target = planar_mating_leg(self.poses()[HEAD.name],gap=.004)
        yield from self.move_object('left',LEG.name,target,2.,calibration,feedback=True)
        self.phase = 'settle separated miters before final press'
        for _ in range(2):
            target = planar_mating_leg(self.poses()[HEAD.name],gap=.004)
            calibration = self.object_calibration('left',LEG.name)
            yield from self.move_object('left',LEG.name,target,.5,calibration,feedback=True)
        yield from self.press_on_table()
        if self.accelerated_bond and self.emb is not None:
            for _ in range(60):
                if self.emb._wood_glue_bond.alive: break
                yield
            if not self.emb._wood_glue_bond.alive:
                raise RuntimeError('Miters never achieved the continuous wet contact required for adhesion.')

        # Keep both ordinary joint/gripper targets fixed at the seated pressing pose.
        self.phase = 'hold assembled corner in place'
        yield from self.wait(2.)

    def action(self):
        try: next(self.generator)
        except StopIteration: self.done = True
        self.steps += 1
        return np.r_[self.left.command,self.right.command]

    def diagnostics(self):
        return dict(steps=self.steps,phase=self.phase,arm_phases=self.arm_phases,
                    proof_lift_clearance_mm=self.proof_lift_mm,
                    table_join=self.join_motion,
                    worst_ik_mm={"left":self.left.worst_mm,"right":self.right.worst_mm},
                    worst_ik_deg={"left":self.left.worst_deg,"right":self.right.worst_deg})
