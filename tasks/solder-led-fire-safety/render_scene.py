#!/usr/bin/env python3
"""Stationary native Isaac/PhysX scene preview. No policy or paid API calls."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import time

SETTLE_STEPS=40


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    args=parser.parse_args()
    physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical_gpu not in ('0','1'):
        parser.error('Check current load and pin one GPU with CUDA_VISIBLE_DEVICES=0 or 1.')
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['OMP_NUM_THREADS']='4'
    scratch=Path('/workspace/davidrm3/robocurve/playground/solder-led-fire-safety/generated')
    scratch.mkdir(parents=True,exist_ok=True)
    os.environ['TMPDIR']=str(scratch)
    args.output.mkdir(parents=True,exist_ok=False)

    import numpy as np
    import cv2
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from yam_arena.inspect_embodiment import yam_arena_embodiment
    import yam_arena.inspect_embodiment as impl
    from isaaclab.app import AppLauncher
    from solder_scene.scene import INSTRUCTION

    started=time.monotonic()
    emb=yam_arena_embodiment(task=str(Path(__file__).with_name('task.py')),device='cuda:0',
          object_jitter=0,control_hz=20,cam_width=args.width,cam_height=args.height)
    if impl._APP is None:
        impl._APP=AppLauncher(headless=True,device='cuda:0',enable_cameras=True,
             kit_args='--/carb/tasking/plugin/threadCount=4 --/omni/tbb/globalcontrol/maxThreadCount=4').app
        atexit.register(impl._close_app_at_exit)
    obs=emb.reset(Scene(id='solder-led-fire-safety',instruction=INSTRUCTION),seed=0)
    target=np.array(obs.state['joint_pos'],float)
    for _ in range(SETTLE_STEPS):
        obs=emb.step(Action(data=target)).observation
    images=dict(obs.images)
    from solder_scene.preview_camera import PreviewCamera
    camera=PreviewCamera(emb,args.width,args.height)
    for name,eye,lookat,focal in [
        ('joint-closeup',(.155,-.145,1.335),(.391,.008,1.050),45.),
        ('scene-angle',(.950,-.720,1.550),(.470,.015,.960),22.),
        ('cardboard-closeup',(.265,-.320,1.430),(.580,-.120,1.080),30.),
    ]:
        camera.aim(eye,lookat,focal)
        images[name]=camera.frame()
    for name,frame in images.items():
        cv2.imwrite(str(args.output/f'{name}.png'),cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR))
    from solder_scene import mechkit as m
    states={}
    raw_states={}
    for name in ('soldering_mat','cardboard','led_light','soldering_iron','iron_power_switch'):
        s=m.state(emb._env,name)
        raw_states[name]=s
        states[name]={'position':s.pos.tolist(),'quat_xyzw':s.quat_xyzw.tolist(),
                      'linear_velocity':s.lin_vel.tolist(),'joints':s.joints}
    from solder_scene.scene import JOINT_LOCAL,CARDBOARD_SIZE,MAT_SIZE,WORK_TOP,TOP
    from solder_scene.scoring import cardboard_distance_xy,mat_supports_joint,mat_is_unobstructed
    light=raw_states['led_light']
    joint=light.pos+light.rotation()@JOINT_LOCAL
    cardboard=raw_states['cardboard']
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*CARDBOARD_SIZE/2
    footprint=corners@cardboard.rotation().T+cardboard.pos
    mat=raw_states['soldering_mat']
    setup_checks=dict(joint_position=joint.tolist(),
        cardboard_clearance_mm=1000*cardboard_distance_xy(footprint,joint),
        mat_under_joint=any(mat_supports_joint(mat.pos,mat.rotation(),MAT_SIZE,joint,z) for z in (TOP,WORK_TOP)),
        mat_unobstructed=bool(mat_is_unobstructed(mat.pos,mat.rotation(),MAT_SIZE,footprint)),
        iron_switch_on=abs(raw_states['iron_power_switch'].joint('power'))<math.radians(1.))
    if not setup_checks['iron_switch_on']:
        raise RuntimeError('Iron switch did not settle in its initial ON state.')
    receipt=dict(stage='scene approval only',instruction=INSTRUCTION,seed=0,settle_steps=SETTLE_STEPS,
                 physical_gpu=int(physical_gpu),cuda_visible_devices=physical_gpu,
                 paid_api_used=False,score=0.,elapsed_seconds=time.monotonic()-started,
                 files=list(images),states=states,setup_checks=setup_checks,
                 source={str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path(__file__).parent.rglob('*.py')})
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    camera.close()
    emb.close()
    print(json.dumps({'output':str(args.output),'elapsed_seconds':receipt['elapsed_seconds'],'states':states}),flush=True)


if __name__=='__main__':
    main()
