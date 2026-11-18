"""Geometry-based supported-precrimp score, shared by live and saved trials.

No controller stage names, targets, forces, plastic memory or success flags are
used. The same conditions can be checked with an end-on photo and a ruler.
"""
from collections import deque
from types import SimpleNamespace
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from yam_arena import ik
import fold_box
import iberville_assets as a
import work_layout as w


class Scorer:
    def __init__(self):
        self.model,_,_=fold_box.build_model(SimpleNamespace(flap_sheet_mm=.3,yield_scale=1.,dt=.001,
            flap_damping=.03,triangular_colliders=False))
        self.data=mujoco.MjData(self.model);m=self.model
        def joint(name):return int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)])
        self.free=joint('box_free')
        self.robot_joints={s:[joint(f'{s}_joint{k}') for k in range(1,9)] for s in ['left','right']}
        self.folds={f'{s}_fold_{k}':joint(f'{s}_fold_{k}') for s in ['front','rear'] for k in range(15)}
        self.tail_bodies={s:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'{s}_strap_14') for s in ['front','rear']}
        self.housing=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'housing')
        self.flap_geoms=[];self.tip_geoms=[]
        for g in range(m.ngeom):
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
            body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or ''
            if '_strap_' in body and name.endswith('_visual'):self.flap_geoms.append(g)
            if body in [s+'_tip_'+f for s in ['left','right'] for f in ['left','right']]:self.tip_geoms.append(g)
        self.history=deque();self.last_time=None;self.last_report=None

    def geometry(self,snapshot):
        m,d=self.model,self.data
        d.qpos[self.free:self.free+7]=snapshot['box_position']+snapshot['box_quat_wxyz']
        for side,ids in self.robot_joints.items():d.qpos[ids]=snapshot['robot_joint_positions'][side]
        for name,adr in self.folds.items():d.qpos[adr]=np.deg2rad(snapshot['angles_deg'][name])
        if not np.all(np.isfinite(d.qpos)):return {'valid':False,'instant_score':0.,'reason':'non-finite state'}
        mujoco.mj_forward(m,d)
        root=d.xpos[self.housing].copy();R=d.xmat[self.housing].reshape(3,3)
        relative=w.CHANNEL_ROTATION.T@R
        rotation_error=float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
        horizontal_error=float(np.linalg.norm((w.CHANNEL_ROTATION.T@(root-w.BOX_TARGET_POSITION))[:2]))
        height_error=abs(float(root[2]-(a.TOP+.200)))
        clearance=min(mujoco.mj_geomDistance(m,d,g,h,.020,None) for g in self.tip_geoms for h in self.flap_geoms)
        valid=horizontal_error<=.005+1e-9 and rotation_error<=5.+1e-9 and height_error<=.020+1e-9 and clearance>=.010-1e-9
        flaps={};total=0.
        for side,b in self.tail_bodies.items():
            tail_R=d.xmat[b].reshape(3,3);axis=R.T@tail_R[:,1]
            sign=-1 if side=='front' else 1
            angle=sign*float(np.degrees(np.arctan2(-axis[0],axis[1])))
            total_turn=sign*sum(value for name,value in snapshot['angles_deg'].items() if name.startswith(side+'_'))
            endpoint=d.xpos[b]+tail_R@np.array([0.,a.STRAP_LENGTH-a.FOLD_STATIONS[-2],0.])
            local=w.CHANNEL_ROTATION.T@(endpoint-np.asarray(w.CHANNEL_POSITION))
            own_half=local[0]<=a.STUD_DEPTH/2 if side=='front' else local[0]>=a.STUD_DEPTH/2
            inward=local[0]-a.STUD_SHEET if side=='front' else a.STUD_DEPTH-a.STUD_SHEET-local[0]
            captured=bool(own_half and inward>=.003 and a.STUD_SHEET<=local[1]<=a.STUD_FLANGE-.005)
            points=0.
            if 60.-1e-9<=angle<=180.+1e-9 and 0.<=total_turn<=180.+1e-9:
                points=.25
                if angle>=150.-1e-9 and captured:points+=.25
            total+=points
            flaps[side]={'angle_degrees':angle,'total_turn_degrees':total_turn,'endpoint_in_channel_mm':(local*1000).tolist(),
                'tip_captured':captured,'points':points}
        return {'valid':bool(valid),'instant_score':total if valid else 0.,
            'horizontal_error_mm':horizontal_error*1000,'orientation_error_degrees':rotation_error,
            'height_error_from_mark_mm':height_error*1000,'minimum_finger_flap_clearance_mm':clearance*1000,
            'flaps':flaps,'box_position':root.tolist(),'box_rotation':R.tolist()}

    def update(self,snapshot):
        time=float(snapshot['t'])
        if self.last_time is not None and time<self.last_time-1e-8:self.history.clear()
        if self.last_time is not None and abs(time-self.last_time)<1e-8:return self.last_report
        self.last_time=time;report=self.geometry(snapshot);self.history.append((time,report))
        while len(self.history)>1 and self.history[1][0]<=time-2.-1e-8:self.history.popleft()
        held=bool(self.history and time-self.history[0][0]>=2.-1e-8)
        position_range=0.;rotation_range=0.
        if held and all(item['valid'] for _,item in self.history):
            positions=np.asarray([item['box_position'] for _,item in self.history])
            position_range=float(np.linalg.norm(np.ptp(positions,axis=0))*1000)
            reference=np.asarray(self.history[0][1]['box_rotation'])
            rotation_range=max(float(np.linalg.norm(ik._rot_error(np.asarray(item['box_rotation']),reference))*180/np.pi) for _,item in self.history)
        else:held=False
        stable=held and position_range<=5.+1e-8 and rotation_range<=2.+1e-8
        report.update(score=min(item['instant_score'] for _,item in self.history) if stable else 0.,
            hold_seconds=2. if stable else 0.,position_range_mm=position_range,rotation_range_degrees=rotation_range)
        if time>240.+1e-8:report['score']=0.
        report['success']=report['score']>=1.-1e-9
        self.last_report=report;return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    parser.add_argument('--output');args=parser.parse_args();run=Path(args.run)
    scorer=Scorer();trace=json.loads((run/'trace.json').read_text());first_full=None
    for snapshot in trace:
        result=scorer.update(snapshot)
        if result['success'] and first_full is None:first_full=snapshot['t']
    result=dict(result,first_full_score_second=first_full,control_steps=len(trace),
        scope='Supported precrimp: body support permitted; fingertip contact with flaps prohibited.')
    text=json.dumps(result,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text)
    print(text,end='')
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()
