"""Finish mounting flaps with a tip push or perpendicular jaw squeezing.

Checkpoint trials are development resets, not continuous demonstrations.
Production native trials do the pickup and preceding passes. Explicit native
finish-only fixtures restore their initial state before the first policy step.
Only ordinary robot targets are commanded; the sheet remains passive.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
import fold_box as f


class IncompleteFinish(RuntimeError):
    """A safe, released attempt fell short of its measured bend target."""


class Finishing(f.Forming):
    def __init__(self,args):
        self.active_finish_side=None;self.finish_peak_forces=None
        super().__init__(args)
        self.finish_metrics=None;self.finishing_by_flap={};self.finishing_attempts={};self.previous_return_error=None

    @staticmethod
    def snapshot_bend(snapshot,side):
        sign=-1 if side=='front' else 1
        return sign*sum(v for n,v in snapshot['angles_deg'].items() if n.startswith(side+'_'))

    def bend(self,side='rear'):
        return self.snapshot_bend(self.sim.snapshot(),side)

    def tick(self):
        try:super().tick()
        finally:
            if self.active_finish_side is not None:
                self.finish_peak_forces=np.maximum(self.finish_peak_forces,self.sim.last_finger_peaks)

    def tail_point(self,station,side='rear'):
        b=mujoco.mj_name2id(self.sim.model,mujoco.mjtObj.mjOBJ_BODY,side+'_strap_14')
        return self.sim.data.xpos[b]+self.sim.data.xmat[b].reshape(3,3)@np.array([0.,station-f.a.FOLD_STATIONS[-2],0.])

    def form(self,side):
        finish_after=side in self.args.finish_flaps.split(',') and self.active_goal>100
        if finish_after:self.active_goal=self.args.rear_return_degrees
        try:super().form(side)
        except RuntimeError as exc:
            # Only a clean, force-safe but insufficient return may proceed
            # to a third contact. Every other guard/error stops the trial.
            if not finish_after or self.active_goal<=100 or 'retained wrap only' not in str(exc):raise
            self.previous_return_error=str(exc)
            print('FINISH_AFTER_INSUFFICIENT_RETURN',str(exc),flush=True)
        if finish_after:self.finish_with_retries(side)

    def run(self):
        if self.args.finish_only:
            if not self.args.checkpoint or (self.args.engine=='isaac' and not self.args.native_development_reset):
                raise ValueError('--finish-only requires a development checkpoint; native use must explicitly opt into a development reset')
            self.prepare_holder()
            for side in self.args.finish_flaps.split(','):self.finish_with_retries(side)
            self.stage='finish_held';self.hold(1.);self.report()
        else:super().run()

    def finish_with_retries(self,side):
        limit=self.args.squeeze_attempts if self.args.finish_method=='squeeze' else 1
        for attempt in range(limit):
            try:
                self.finish(side)
                return
            except IncompleteFinish:
                if attempt+1>=limit:raise
                # Only a completed, open-jaw withdrawal can reach this point.
                # Recompute entry from the new physical shape, without resets.
                print('SQUEEZE_REAPPROACH',side,'attempt',attempt+2,flush=True)

    def finish(self,side='rear'):
        start=len(self.trace);before=self.bend(side)
        self.active_finish_side=side;self.finish_peak_forces=np.zeros(4)
        self.finish_metrics={'flap':side,'method':self.args.finish_method,'starting_bend_degrees':before,
            'start_simulated_second':float(self.sim.data.time),
            'goal_degrees':self.args.finish_goal_degrees,'retained_min_degrees':self.args.finish_retained_degrees}
        self.finishing_by_flap[side]=self.finish_metrics
        attempts=self.finishing_attempts.setdefault(side,[])
        self.finish_metrics['attempt']=len(attempts)+1;attempts.append(self.finish_metrics)
        released=False
        try:
            if self.args.finish_method=='tip':self.tip_push(side)
            else:self.squeeze(side)
            released=True
        finally:
            after=self.bend(side)
            peak=max([before]+[self.snapshot_bend(row,side) for row in self.trace[start:]])
            self.finish_metrics.update(peak_bend_degrees=peak,
                retained_bend_degrees=after if released else None,
                bend_at_stop_degrees=after,withdrawal_completed=released,
                end_simulated_second=float(self.sim.data.time),
                control_steps=len(self.trace)-start,peak_measured_finger_forces_n=self.finish_peak_forces.tolist())
            self.active_finish_side=None
        if peak<self.args.finish_goal_degrees or after<self.args.finish_retained_degrees:
            raise IncompleteFinish(f'{side} finishing incomplete: peak {peak:.1f}, retained {after:.1f} degrees')

    def tip_push(self,side='rear'):
        self.args.contact_station=self.args.tip_station_mm/1000.
        self.args.walk_contact=False
        self.active_goal=self.args.finish_goal_degrees
        self.args.retained_min_degrees=self.args.finish_retained_degrees
        self.finish_metrics['contact_station_mm']=self.args.tip_station_mm
        self.finish_metrics['initial_contact_point_world_m']=self.tail_point(self.args.contact_station,side).tolist()
        # Explicit base call: a failed third push must never recursively retry.
        super().form(side)

    def squeeze_pose(self,half_gap,clearance=0.,side='rear'):
        _,tool,_,_=self.material_contact(side)
        if abs(tool[0,0])<.99:raise ValueError('Squeeze requires jaws closing across world X')
        contact=self.tail_point(self.args.squeeze_station_mm/1000.,side)
        g=mujoco.mj_name2id(self.sim.model,mujoco.mjtObj.mjOBJ_GEOM,'stud_'+side)
        if abs(self.sim.data.geom_xmat[g].reshape(3,3)[0,0])<.99:
            raise ValueError('Squeeze prototype requires the current unrotated channel')
        outward=-1 if side=='front' else 1
        wall=self.sim.data.geom_xpos[g,0]+outward*self.sim.model.geom_size[g,0]
        goal=contact.copy()
        # The CAD's inner fingertip face is 0.03 mm from the nominal jaw origin.
        # Keep the outside jaw against the stud as the jaws close, moving the
        # TCP with the measured opening rather than teleporting/attaching it.
        goal[0]=wall-outward*(half_gap+.00003-clearance)
        goal[1]+=self.args.squeeze_pad_depth_mm/1000.
        return goal,tool,contact

    def squeeze(self,side='rear'):
        opening=self.args.squeeze_open_mm/1000.
        if self.args.squeeze_entry_overlap_mm>0.:
            # Catch the return tail near its free end with the inside finger.
            # The outside finger stays just outside the steel. Close BEFORE
            # approaching; sliding horizontally then advances the flap instead
            # of letting both open fingers bypass it and compress it end-on.
            g=mujoco.mj_name2id(self.sim.model,mujoco.mjtObj.mjOBJ_GEOM,'stud_'+side)
            outward=-1 if side=='front' else 1
            wall=self.sim.data.geom_xpos[g,0]+outward*self.sim.model.geom_size[g,0]
            tip=self.tail_point(self.args.squeeze_station_mm/1000.,side)
            span=-outward*(tip[0]-wall)
            opening=min(opening,max(.003,(span-self.args.squeeze_entry_overlap_mm/1000.)/2.))
        goal,tool,contact=self.squeeze_pose(opening,.0006,side)
        self.finish_metrics.update(contact_station_mm=self.args.squeeze_station_mm,
            initial_contact_point_world_m=contact.tolist(),closing_axis_world=[1.,0.,0.],
            entry_opening_mm=opening*2000.,entry_tip_overlap_mm=self.args.squeeze_entry_overlap_mm)
        self.stage=side+'_squeeze_approach'
        self.move_left(goal+[0,.060,.025],tool,3.,grip=-opening,joint=True)
        if self.args.squeeze_entry_overlap_mm>0.:
            # Finish lowering while clear of the tail, then move only along Y.
            self.move_left(goal+[0,.035,0.],tool,2.,grip=-opening)
            self.stage=side+'_squeeze_slide'
            start,_=self.pose_side('left');point=start.copy()
            distance=max(0.,start[1]-goal[1]);advanced=0.;elapsed=0.
            while elapsed<distance/(self.args.contact_speed_mm_s/1000.)+4.:
                recent=max(self.sim.last_finger_peaks[:2])
                if recent<self.args.soft_force_n:
                    advanced=min(distance,advanced+self.args.contact_speed_mm_s/1000.*f.p.CONTROL_DT)
                point[1]=start[1]-advanced
                hand,hand_R=self.pose_side('left')
                correction=np.clip(point-hand,-.0015,.0015)
                er=f.ik._rot_error(tool,hand_R)*1.5
                if np.linalg.norm(er)>np.deg2rad(2):er*=np.deg2rad(2)/np.linalg.norm(er)
                self.sim.command('left',point+correction,f.rotation_vector(er)@tool)
                self.sim.targets['left'][6:]=-opening
                self.tick();elapsed+=f.p.CONTROL_DT
                if advanced>=distance:break
            self.finish_metrics.update(slide_distance_mm=advanced*1000.,
                bend_after_horizontal_slide_degrees=self.bend(side))
            self.report()
        else:
            self.move_left(goal,tool,2.,grip=-opening)
            goal,_,_=self.squeeze_pose(opening,0.,side)
            self.stage=side+'_squeeze_seat';self.move_left(goal,tool,1.,grip=-opening)
        point,_=self.pose_side('left');best=self.bend(side);progress=0.;command_gap=opening
        hold_start=None;hold_seconds=self.args.squeeze_hold_seconds
        self.finish_metrics.update(requested_hold_seconds=hold_seconds,actual_hold_seconds=0.,
            requested_finger_drive_n=self.args.squeeze_drive_n,force_backoff_n=self.args.squeeze_force_n,
            fixed_tcp_target_during_hold=True,hold_setting=self.args.squeeze_hold_setting)
        for step in range(round((self.args.squeeze_seconds+hold_seconds)/f.p.CONTROL_DT)):
            elapsed=(step+1)*f.p.CONTROL_DT
            if hold_start is None and elapsed>=self.args.squeeze_seconds:hold_start=elapsed
            qa,_=self.sim.addresses['left'];gaps=np.maximum(0.,-self.sim.data.qpos[qa[6:]])
            actual=float(gaps.mean());outside=float(gaps[1 if side=='front' else 0])
            if hold_start is None:
                goal,_,_=self.squeeze_pose(outside,-.00005,side)
                delta=goal-point;distance=np.linalg.norm(delta)
                maximum=self.args.contact_speed_mm_s/1000.*f.p.CONTROL_DT
                if distance>maximum:delta*=maximum/distance
                point+=delta
            # Once pressure is established, hold the arm target stationary.
            # Following the moving tail during the hold can drive the jaw
            # farther into the steel even after the flap is already flat.
            hand,_=self.pose_side('left');correction=np.clip(point-hand,-.0015,.0015)
            self.sim.command('left',point+correction,tool)
            # Advance from the previous command, including any force backoff.
            # A wall-clock ramp would jump closed again after a force pause.
            recent=max(self.sim.last_finger_peaks[:2])
            if recent>self.args.squeeze_force_n:
                command_gap+=min(.0003,(recent-self.args.squeeze_force_n)/self.sim.kp[6]*.8+.00002)
            elif recent<self.args.squeeze_force_n-.5 and (hold_start is None or self.args.squeeze_hold_setting=='full-close'):
                command_gap-=opening*f.p.CONTROL_DT/self.args.squeeze_close_seconds
            command_gap=max(command_gap,float(gaps.max())-self.args.squeeze_drive_n/self.sim.kp[6])
            command_gap=float(np.clip(command_gap,0.,opening))
            self.sim.targets['left'][6:]=-command_gap
            self.stage=side+('_squeeze_hold' if hold_start is not None else '_squeeze_close');self.tick()
            angle=self.bend(side)
            if angle>best+.5:best=angle;progress=elapsed
            if step%20==19:print('SQUEEZE_PROGRESS',side,round(elapsed,2),round(angle,2),
                'opening_mm',round(actual*2000,2),'close_target_mm',round(command_gap*2000,2),
                'force_n',np.round(self.sim.last_finger_peaks[:2],2).tolist(),flush=True)
            at_maximum_close=command_gap<=.00001 or recent>=self.args.squeeze_force_n-.5
            if hold_start is None and angle>=self.args.finish_goal_degrees and (self.args.squeeze_hold_setting=='goal' or at_maximum_close):hold_start=elapsed
            if hold_start is not None:
                self.finish_metrics['actual_hold_seconds']=elapsed-hold_start
                if elapsed-hold_start>=hold_seconds:break
            if hold_start is None and elapsed-progress>3. and command_gap<.0001:hold_start=elapsed
        self.report()
        self.release_jaws(side,opening)

    def release_jaws(self,side,opening):
        # Open before withdrawal so a blocked jaw cannot drag or tear the flap.
        self.stage=side+'_squeeze_open'
        release_opening=max(opening,self.args.squeeze_open_mm/1000.)
        qa,_=self.sim.addresses['left']
        release_gap=float(np.maximum(0.,-self.sim.targets['left'][6:]).min())
        release_arm_targets=self.sim.targets['left'][:6].copy()
        release_rate=self.args.squeeze_release_mm_s/1000.
        release_seconds=0.;release_timeout=max(0.,release_opening-release_gap)/release_rate+4.
        while release_seconds<release_timeout:
            # Ramp from the existing close command, not the measured opening:
            # jumping to the latter would abruptly release stored sheet energy.
            # Keep the loaded arm targets unchanged until the jaws are clear.
            if max(self.sim.last_finger_peaks[:2])<self.args.soft_force_n:
                release_gap=min(release_opening,release_gap+release_rate*f.p.CONTROL_DT)
            self.sim.targets['left'][:6]=release_arm_targets
            self.sim.targets['left'][6:]=-release_gap
            self.tick();release_seconds+=f.p.CONTROL_DT
            actual_gap=float(np.maximum(0.,-self.sim.data.qpos[qa[6:]]).min())
            if release_gap>=release_opening and actual_gap>=release_opening-.0005:break
        self.finish_metrics['release_seconds']=release_seconds
        if actual_gap<release_opening-.0005:raise RuntimeError('Unable to open jaws safely before withdrawal')
        self.report()
        hand,release_rotation=self.pose_side('left')
        self.stage=side+'_squeeze_retract';self.move_left(hand+[0,.055,.025],release_rotation,2.,grip=-release_opening)
        self.hold(.5);self.report()

    def save(self,error):
        super().save(error)
        path=self.out/'result.json';record=json.loads(path.read_text())
        record.update(finishing=self.finish_metrics,previous_return_error=self.previous_return_error,
            finishing_by_flap=self.finishing_by_flap,
            finishing_attempts=self.finishing_attempts,
            finishing_demonstrated=error is None and self.finish_metrics is not None,
            task_stage='precrimp' if self.args.finish_method=='squeeze' else 'forming',
            precrimp_completed_while_supported=(error is None and self.args.finish_method=='squeeze'
                and set(self.finishing_by_flap)=={'front','rear'}
                and all(self.bend(side)>=self.args.finish_retained_degrees for side in ['front','rear'])),
            full_installation_demonstrated=False)
        path.write_text(json.dumps(record,indent=2)+'\n')


def main():
    parser=f.argument_parser()
    parser.add_argument('--finish-method',choices=['tip','squeeze'],required=True)
    parser.add_argument('--finish-only',action='store_true')
    parser.add_argument('--finish-flaps',default='rear',help='Flaps to finish: front, rear, or front,rear')
    parser.add_argument('--finish-goal-degrees',type=float,default=166.)
    parser.add_argument('--finish-retained-degrees',type=float,default=140.)
    parser.add_argument('--rear-return-degrees','--return-degrees',dest='rear_return_degrees',type=float,default=130.)
    parser.add_argument('--tip-station-mm',type=float,default=53.5)
    parser.add_argument('--squeeze-station-mm',type=float,default=52.)
    parser.add_argument('--squeeze-pad-depth-mm',type=float,default=8.)
    parser.add_argument('--squeeze-open-mm',type=float,default=16.)
    parser.add_argument('--squeeze-entry-overlap-mm',type=float,default=1.5,
        help='Preclose to catch this far inside the tail tip, then approach horizontally; 0 uses the historical open approach')
    parser.add_argument('--squeeze-seconds',type=float,default=18.)
    parser.add_argument('--squeeze-close-seconds',type=float,default=8.)
    parser.add_argument('--squeeze-force-n',type=float,default=22.)
    parser.add_argument('--squeeze-drive-n',type=float,default=18.)
    parser.add_argument('--squeeze-hold-seconds',type=float,default=0.)
    parser.add_argument('--squeeze-hold-setting',choices=['goal','full-close'],default='full-close',
        help='Hold the jaw setting reached at the precrimp angle; full-close reproduces the maximum-closure experiment')
    parser.add_argument('--squeeze-release-mm-s',type=float,default=2.,help='Half-gap opening speed while unloading at the measured hand pose')
    parser.add_argument('--squeeze-attempts',type=int,choices=[1,2],default=2,
        help='At most one physical re-approach after safe withdrawal if the measured finish is incomplete')
    args=parser.parse_args();args.stage='place';args.closed=-.034;args.forming_material=True
    if not set(args.finish_flaps.split(','))<={'front','rear'}:parser.error('Unknown flap name')
    if not 0.<args.squeeze_drive_n<=40.:parser.error('Finger drive must stay within the 40 N actuator limit')
    if not 0.<args.squeeze_force_n<40.:parser.error('Force backoff must be below the 40 N sustained contact threshold')
    if args.squeeze_hold_seconds<0.:parser.error('Hold duration must be nonnegative')
    if args.squeeze_entry_overlap_mm<0.:parser.error('Entry overlap must be nonnegative')
    if args.squeeze_release_mm_s<=0.:parser.error('Release speed must be positive')
    args.precrimp_mode=args.finish_method=='squeeze' and set(args.finish_flaps.split(','))=={'front','rear'}
    if not 0.<args.tip_station_mm<f.a.STRAP_LENGTH*1000.:parser.error('Tip station must lie on the flap')
    if not 0.<args.squeeze_station_mm<f.a.STRAP_LENGTH*1000.:parser.error('Squeeze station must lie on the flap')
    trial=Finishing(args);error=None
    try:trial.run()
    except Exception as exc:error=str(exc)
    trial.save(error)
    if hasattr(trial.sim,'close'):trial.sim.close()
    if error:raise SystemExit(1)


if __name__=='__main__':main()
