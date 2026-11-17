#!/usr/bin/env python3
"""Run and record native YAM/PhysX execution. Runs scripted or Dream-1 actions; never uploads submission media."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback


def choose_gpu(selection):
    raw = subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.used,utilization.gpu,uuid",
                                   "--format=csv,noheader,nounits"],text=True)
    rows = []
    for line in raw.splitlines():
        idx,name,memory,util,uuid = [s.strip() for s in line.split(",")]
        rows.append(dict(index=int(idx),name=name,memory_mb=int(memory),utilization=int(util),uuid=uuid))
    if selection == "auto":
        selected = min(rows,key=lambda r:(r["utilization"],r["memory_mb"]))
    else:
        selected = next(r for r in rows if r["index"] == int(selection))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected["index"])
    return selected,rows


def fingerprint():
    root = Path(__file__).resolve().parent
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py','.png','.svg')}


def run(args):
    selected,loads = choose_gpu(args.gpu)
    print(json.dumps(dict(selected_gpu=selected,loads=loads)),flush=True)
    # A single small scene does not need a machine-sized BLAS/Kit/encoder thread pool.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = str(args.cpu_workers)
    import numpy as np
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from yam_arena.inspect_embodiment import yam_arena_embodiment
    from wood_glue.scene import INSTRUCTION
    from wood_glue.runtime import measured_poses,configure
    configure(accelerated_bond=args.accelerated_bond)

    instruction = ("Pick up the labelled glue bottle and turn it upside down, holding it clear of the bench."
                   if args.mode == "pickup-flip" else INSTRUCTION)
    if args.mode in ("hold-and-glue","bead"):
        instruction = ("Pick up the loose frame rail with one hand and use the labelled bottle in the other "
                       "hand to apply one long bead of glue to its 45-degree cut face. Hold the coated piece clear of the bench.")

    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    receipt = dict(status="starting",mode=args.mode,seed=args.seed,max_steps=args.steps,
                   control_hz=20,physical_gpu=selected,gpu_load_at_start=loads,
                   cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],source=fingerprint(),
                   instruction=instruction,dream_api_used=args.mode=="dream",prototype_uncalibrated=True,
                   accelerated_cure=args.accelerated_bond)
    receipt['cpu_workers'] = args.cpu_workers
    receipt['encoder_threads_per_camera'] = 2
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    emb,writers,metrics,inspection = None,{},{},None
    policy = session = None
    steps,inverted_hold = 0,0.
    try:
        emb = yam_arena_embodiment(task=str(Path(__file__).with_name("task.py")),device="cuda:0",
                                  object_jitter=0,control_hz=20,cam_width=args.width,cam_height=args.height)
        import yam_arena.inspect_embodiment as arena_impl
        from isaaclab.app import AppLauncher
        if arena_impl._APP is None:
            kit_args = (f'--/carb/tasking/plugin/threadCount={args.cpu_workers} '
                        f'--/omni/tbb/globalcontrol/maxThreadCount={args.cpu_workers}')
            arena_impl._APP = AppLauncher(headless=True,device='cuda:0',enable_cameras=True,kit_args=kit_args).app
            atexit.register(arena_impl._close_app_at_exit)
        obs = emb.reset(Scene(id="wood-glue",instruction=instruction),seed=args.seed)
        from wood_glue.runtime import current
        import cv2
        if not args.no_record:
            pictures = dict(obs.images)
            if args.inspection_camera:
                from wood_glue.inspection import InspectionCamera,EYE,TARGET,ASSEMBLY_EYE,ASSEMBLY_TARGET
                inspection = InspectionCamera(emb,args.width,args.height)
                if args.mode == 'bottle-preview': inspection.show_bottle()
                pictures["inspection_cam"] = inspection.frame()
                receipt["inspection_camera"] = dict(eye=EYE,target=TARGET,focal_length_mm=40.,
                    horizontal_aperture_mm=20.955,auxiliary=True,part_of_robot_observation=False)
                receipt['inspection_camera']['assembly_view'] = dict(eye=ASSEMBLY_EYE,target=ASSEMBLY_TARGET,focal_length_mm=24.)
            for name,frame in pictures.items():
                cv2.imwrite(str(out/f"setup-{name}.png"),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                h,w = frame.shape[:2]
                writers[name] = subprocess.Popen(["ffmpeg","-nostdin","-loglevel","error","-f","rawvideo",
                    "-pixel_format","rgb24","-video_size",f"{w}x{h}","-framerate","20","-i","pipe:0","-an",
                    "-c:v","libx264","-preset","fast","-crf","23","-pix_fmt","yuv420p","-movflags",
                    "+faststart","-threads","2",str(out/f"{name}.mp4")],stdin=subprocess.PIPE)
                writers[name].stdin.write(np.ascontiguousarray(frame).tobytes())
        if args.mode in ("smoke","bottle-preview","dream"):
            action = np.asarray(obs.state["joint_pos"],float)
            controller = None
        else:
            from wood_glue.controller import Controller
            controller = Controller(emb,mode=args.mode,accelerated_bond=args.accelerated_bond)
        arm_clearance_min = 50.
        arm_clearance_samples = 0
        wood_clearance_min = 50.
        retreat_clearance = 50.
        retreat_samples = 0
        bead_frame_tick = 0
        bead_frames = []
        if args.mode == 'dream':
            from inspect_robots_dream.policy import DreamPolicy
            from wood_glue.dream_run import AuditedDreamSession
            session = AuditedDreamSession(out)
            policy = DreamPolicy(model=args.model,rows_per_request=args.rows_per_request,record=False,session=session)
            policy.bind(emb.info)
            policy.reset(Scene(id='wood-glue',instruction=instruction))
            receipt['model'] = args.model
            receipt['server_recording'] = False
        with (out/"trace.jsonl").open("w") as trace:
            for step in range(args.steps):
                if policy is not None:
                    chunk = policy.act(obs)
                    action = np.asarray(chunk.actions[0].data,float)
                elif controller is not None:
                    action = controller.action()
                result = emb.step(Action(data=action))
                obs = result.observation
                steps = step+1
                metrics = current().metrics()
                if hasattr(emb,'_wood_glue_bond'):
                    metrics['adhesive_bond'] = emb._wood_glue_bond.metrics()
                poses,states = measured_poses(emb)
                if args.mode == "pickup-flip":
                    from wood_glue.milestone import evaluate
                    status = evaluate(emb,poses,states)
                    inverted_hold = inverted_hold+.05 if status["qualified"] else 0.
                    metrics.update(pickup_flip=status,inverted_hold_s=inverted_hold,
                                   milestone_success=inverted_hold >= 1.-1e-9)
                elif args.mode in ("hold-and-glue","bead"):
                    from wood_glue.milestone import evaluate_hold_glue
                    status = evaluate_hold_glue(emb,poses,states,current())
                    inverted_hold = inverted_hold+.05 if status["qualified"] else 0.
                    metrics.update(held_edge_glue=status,glue_hold_s=inverted_hold,
                                   milestone_success=inverted_hold >= 1.-1e-9)
                if args.mode == 'demo':
                    metrics['milestone_success'] = bool(metrics['success'])
                from wood_glue.clearance import measured_joints,robot_hulls,arm_clearance,wood_clearance
                joints = measured_joints(emb)
                hulls = robot_hulls(joints)
                clearance,pair = arm_clearance(joints,hulls)
                if controller is not None:
                    contacts = controller.wood_contacts.copy()
                else:
                    holding = metrics.get('held_corner') or {}
                    contacts = {side:hand['piece'] for side,hand in holding.get('hands',{}).items() if hand['holding']}
                wood_gap,wood_pair = wood_clearance(joints,poses,contacts,hulls)
                wood_clearance_min = min(wood_clearance_min,wood_gap)
                metrics['arm_wood_clearance_lower_bound_mm'] = wood_clearance_min
                if wood_gap < 5. and args.mode != "dream":
                    raise RuntimeError(f'Unintended arm-to-wood clearance below 5 mm: {wood_gap:.3f}, {wood_pair}')
                arm_clearance_min = min(arm_clearance_min,clearance)
                arm_clearance_samples += 1
                metrics['arm_clearance_lower_bound_mm'] = arm_clearance_min
                metrics['arm_clearance_samples'] = arm_clearance_samples
                if clearance < 5. and args.mode != "dream":
                    raise RuntimeError(f'Inter-arm clearance below 5 mm: {clearance:.3f}, {pair}')
                if controller is not None and controller.phase.startswith(('release corner','withdraw right arm','withdraw left arm','inspect released corner')):
                    retreat_clearance = min(retreat_clearance,clearance)
                    retreat_samples += 1
                    if clearance < 5. and args.mode != "dream":
                        raise RuntimeError(f'Inter-arm retreat clearance below 5 mm: {clearance:.3f}, {pair}')
                if retreat_samples:
                    metrics['retreat_arm_clearance_lower_bound_mm'] = retreat_clearance
                    metrics['retreat_clearance_samples'] = retreat_samples
                row = dict(joints=joints,wood_contacts=contacts,arm_phases={} if controller is None else controller.arm_phases.copy(),step=steps,time_s=steps/20,action=action.tolist(),
                           phase="hold" if controller is None else controller.phase,metrics=metrics,
                           poses={name:dict(pos=p.pos.tolist(),rotation=p.rot.tolist(),
                                           linear=p.linear.tolist(),angular=p.angular.tolist())
                                  for name,p in poses.items()},sliders=states["glue_bottle"].sliders)
                trace.write(json.dumps(row)+"\n")
                pictures = dict(result.observation.images)
                if inspection is not None:
                    if controller is not None and controller.phase in (
                            'turn coated leg flat for assembly','pick up head casing from table',
                            'position head casing for corner','bring coated miter to matching face',
                            'press miters together at 90 degrees','release assembled corner','inspect released corner','finished'):
                        inspection.show_assembly()
                    pictures["inspection_cam"] = inspection.frame()
                if controller is not None and controller.phase == 'apply single glue bead':
                    bead_frame_tick += 1
                    label = {28:'bead-start-top.png',52:'bead-mid-top.png',76:'bead-end-top.png'}.get(bead_frame_tick)
                    if label and 'top_cam' in pictures:
                        cv2.imwrite(str(out/label),cv2.cvtColor(pictures['top_cam'],cv2.COLOR_RGB2BGR))
                        bead_frames.append(dict(file=label,step=steps,time_s=steps/20,camera='standard top_cam',phase_tick=bead_frame_tick))
                        (out/'bead-frames.json').write_text(json.dumps(bead_frames,indent=2)+'\n')
                for name,writer in writers.items():
                    writer.stdin.write(np.ascontiguousarray(pictures[name]).tobytes())
                if steps % 20 == 0:
                    print(json.dumps(dict(step=steps,phase=row["phase"],metrics=metrics)),flush=True)
                    trace.flush()
                if metrics['violations'] and args.mode != 'dream':
                    raise RuntimeError(f"Native task violation: {metrics['violations']}")
                if (result.terminated and args.mode != 'demo') or (controller is not None and controller.done):
                    break
        receipt.update(status="completed",steps=steps,final_metrics=metrics,
                       success=bool(metrics.get("milestone_success") if args.mode in ("pickup-flip","hold-and-glue","bead")
                                    else metrics.get("success")))
        if args.mode in ('demo','dream'):
            receipt['completion_criterion'] = 'held_corner_task'
        if controller is not None:
            receipt["controller"] = controller.diagnostics()
    except BaseException:
        receipt.update(status="error",steps=steps,final_metrics=metrics,error=traceback.format_exc())
        raise
    finally:
        if policy is not None:
            receipt['policy'] = policy.transcript()
            policy.close()
        errors = {}
        for name,writer in writers.items():
            writer.stdin.close()
            code = writer.wait()
            if code: errors[name] = code
        receipt.update(elapsed_wall_s=time.monotonic()-start,video_errors=errors,
                       source_unchanged=receipt["source"] == fingerprint())
        (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
        if inspection is not None: inspection.close()
        if emb is not None: emb.close()
    print(json.dumps(receipt),flush=True)
    if args.mode == 'demo' and not metrics.get('success'):
        raise SystemExit('The held-corner demo endpoint did not pass; inspect the native trace and videos.')
    if args.mode == "pickup-flip" and not receipt["success"]:
        raise SystemExit("Pickup-and-flip milestone did not pass; inspect the native trace and videos.")
    if args.mode in ("hold-and-glue","bead") and not receipt["success"]:
        raise SystemExit("Held-edge glue milestone did not pass; inspect the native trace and videos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["dream","demo","bead","hold-and-glue","pickup-flip","smoke","bottle-preview"],default="demo")
    parser.add_argument("--model",default="dreamlabs/dream-1-beta")
    parser.add_argument("--rows-per-request",type=int,default=20)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--gpu",default="auto",help="physical GPU index, or auto for least utilized/allocated")
    parser.add_argument("--steps",default=2400,type=int)
    parser.add_argument("--seed",default=0,type=int)
    parser.add_argument("--width",default=640,type=int)
    parser.add_argument("--height",default=360,type=int)
    parser.add_argument("--no-record",action="store_true")
    parser.add_argument('--cpu-workers',type=int,default=8,help='process-local Kit/OpenMP worker limit')
    parser.add_argument("--inspection-camera",action="store_true",help="also record a fixed auxiliary close-up; not a policy observation")
    parser.add_argument("--accelerated-bond",action=argparse.BooleanOptionalAction,default=False,
                        help="optional diagnostic shortcut: form a breakable joint after two seconds; disabled for the submission task, which ends with both pieces physically held")
    def terminated(signum,frame):
        raise InterruptedError(f'Run received signal {signum}; preserving its partial artifacts.')
    signal.signal(signal.SIGTERM,terminated)
    try:
        run(parser.parse_args())
    except BaseException as error:
        # Kit's registered atexit hook exits the process with zero even after SystemExit(1).
        # Artifacts/video writers have already been finalized by run(); preserve the failure status.
        if isinstance(error,SystemExit):
            code = error.code if isinstance(error.code,int) else 1
            if error.code and not isinstance(error.code,int): print(error,file=sys.stderr)
        else:
            traceback.print_exc()
            code = 130 if isinstance(error,KeyboardInterrupt) else 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
