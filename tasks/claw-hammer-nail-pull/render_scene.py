#!/usr/bin/env python3
"""Render and validate the initial native scene; no motion policy or paid API."""
import argparse
import atexit
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    args=parser.parse_args()
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in ('0','1'):parser.error('Check load and pin one GPU with CUDA_VISIBLE_DEVICES')
    root=Path(os.environ.get('ROBOCURVE_HOME','/workspace/davidrm3/robocurve'))
    scratch=root/'playground/claw-hammer-nail-pull/generated'
    scratch.mkdir(parents=True,exist_ok=True)
    os.environ['TMPDIR']=str(scratch)
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['OMP_NUM_THREADS']='4'
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import cv2
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from yam_arena.inspect_embodiment import yam_arena_embodiment
    import yam_arena.inspect_embodiment as impl
    from isaaclab.app import AppLauncher
    from nail_scene.scene import INSTRUCTION,TOP,BOARD_SIZE
    from nail_scene import mechkit as m
    from nail_scene.runtime import TRIAL
    from nail_scene.scoring import score
    started=time.monotonic()
    emb=yam_arena_embodiment(task=str(Path(__file__).with_name('task.py')),
        device='cuda:0',object_jitter=0,control_hz=20,
        cam_width=args.width,cam_height=args.height)
    if impl._APP is None:
        impl._APP=AppLauncher(headless=True,device='cuda:0',enable_cameras=True,
            kit_args='--/carb/tasking/plugin/threadCount=4 --/omni/tbb/globalcontrol/maxThreadCount=4').app
        atexit.register(impl._close_app_at_exit)
    obs=emb.reset(Scene(id='claw-hammer-nail-pull',instruction=INSTRUCTION),seed=0)
    target=np.array(obs.state['joint_pos'],float)
    for _ in range(40):obs=emb.step(Action(data=target)).observation
    print('SETTLED_STATES',json.dumps({name:{'pos':m.state(emb._env,name).pos.tolist(),
        'quat_xyzw':m.state(emb._env,name).quat_xyzw.tolist()}
        for name in ('board','nail','hammer','tray')}),flush=True)
    images=dict(obs.images)
    from nail_scene.preview_camera import PreviewCamera
    camera=PreviewCamera(emb,args.width,args.height)
    for name,eye,target,focal in [
            ('nail-closeup',(.095,-.150,1.095),(.325,.080,TOP+.050),50.),
            ('hammer-closeup',(.540,-.500,1.160),(.315,-.305,TOP+.012),35.),
            ('scene-angle',(.900,-.700,1.450),(.310,.010,TOP+.025),26.)]:
        camera.aim(eye,target,focal);images[name]=camera.frame()
    for name,frame in images.items():
        cv2.imwrite(str(args.output/(name+'.png')),cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR))
    states={name:m.state(emb._env,name) for name in ('board','nail','hammer','tray')}
    board,nail=states['board'],states['nail']
    tip=(nail.pos-board.pos)@board.rotation()
    embedded_mm=(BOARD_SIZE[2]/2-tip[2])*1000
    checks={'initial_score_zero':score(TRIAL.outcome)==0.,
        'nail_embedment_20mm':abs(embedded_mm-20.)<.5,
        'board_on_table':abs(board.pos[2]-BOARD_SIZE[2]/2-TOP)<.004,
        'board_flat':board.tilt_deg()<1.,
        'nail_still':float(np.linalg.norm(nail.lin_vel))<.002}
    from pxr import UsdPhysics
    material_joint=UsdPhysics.Joint.Get(emb._base.sim.stage,'/World/envs/env_0/nail/wood_joint')
    checks['joint_bound_to_board']=[str(p) for p in material_joint.GetBody0Rel().GetTargets()]==['/World/envs/env_0/board']
    receipt={'stage':'Step 2 scene approval; no completion trajectory yet',
        'physical_gpu':int(gpu),'seed':0,'settle_steps':40,'paid_api_used':False,
        'elapsed_seconds':time.monotonic()-started,'setup_checks':checks,
        'embedment_mm':embedded_mm,'outcome':asdict(TRIAL.outcome),
        'material':TRIAL.final_state,
        'states':{n:{'pos':s.pos.tolist(),'quat_xyzw':s.quat_xyzw.tolist(),
                     'velocity':s.lin_vel.tolist()} for n,s in states.items()},
        'source':{str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in Path(__file__).parent.rglob('*.py')}}
    def scalar(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(f'Unsupported receipt value: {type(value)}')
    serialized=json.dumps(receipt,indent=2,default=scalar)
    (args.output/'receipt.json').write_text(serialized+'\n')
    camera.close();emb.close()
    print(json.dumps(receipt,default=scalar),flush=True)
    if not all(checks.values()):raise RuntimeError('Scene setup verification failed')


if __name__=='__main__':main()
