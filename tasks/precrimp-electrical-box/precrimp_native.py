"""Process-local articulation scoring adapter for the standard YAM SDK."""
from pathlib import Path
import numpy as np


def array(value):return (value.torch if hasattr(value,'torch') else value).cpu().numpy()


def install(folder):
    import yam_arena.inspect_embodiment as implementation
    from inspect_robots import StepResult
    cls=implementation.YamArenaEmbodiment
    if getattr(cls,'_electrical_precrimp_scoring',False):return
    cls._electrical_precrimp_scoring=True
    old_step=cls._task_step;old_reset=cls.reset

    def reset(self,*args,**kwargs):
        self._electrical_precrimp_cache=None
        result=old_reset(self,*args,**kwargs)
        if self.task is not None and self.task.path.parent.resolve()==Path(folder).resolve():self.task.module.reset_score()
        return result

    def step(self):
        if self.task is None or self.task.path.parent.resolve()!=Path(folder).resolve():return old_step(self)
        from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS
        if getattr(self,'_electrical_precrimp_cache',None) is None:
            box=self._base.scene['device_box']
            folds=[f'{s}_fold_{k}' for s in ['front','rear'] for k in range(15)]
            fold_ids=box.find_joints(folds,preserve_order=True)[0]
            robot_ids={s:self._base.scene[s+'_robot'].find_joints(ARM_JOINTS+FINGER_JOINTS,preserve_order=True)[0] for s in ['left','right']}
            self._electrical_precrimp_cache=(folds,fold_ids,robot_ids)
        folds,fold_ids,robot_ids=self._electrical_precrimp_cache
        box=self._base.scene['device_box'];pose=array(box.data.root_link_pose_w)[0]
        snapshot={'t':float(self._t),'box_position':pose[:3].tolist(),
            'box_quat_wxyz':pose[[6,3,4,5]].tolist(),
            'angles_deg':dict(zip(folds,np.degrees(array(box.data.joint_pos)[0,fold_ids]).tolist())),
            'robot_joint_positions':{s:array(self._base.scene[s+'_robot'].data.joint_pos)[0,ids].tolist() for s,ids in robot_ids.items()}}
        value=self.task.module.score(snapshot);report=self.task.module.scoring_report()
        self._precrimp_report=report
        return StepResult(observation=self._observation(),terminated=value>=1.-1e-9,
            termination_reason='success' if value>=1.-1e-9 else None,
            info={'score':value,'success':value>=1.-1e-9,'precrimp':report})

    cls.reset=reset;cls._task_step=step
