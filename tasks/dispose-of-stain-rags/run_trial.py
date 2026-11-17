"""Run one scored trial of the scripted demo and record it.

    CUDA_VISIBLE_DEVICES=0 python tasks/dispose-of-stain-rags/run_trial.py --out runs/demo

Uses the yam_arena embodiment's own reset() and step(), so the trial is scored exactly as
`inspect-robots run --scorer task_score` scores it: 1.0 if success() holds for 3 s (the episode then
ends), otherwise score() at the end. Writes result.json, H.264 MP4s of the top and wrist cameras
(640 x 360, 20 fps) and the first and last top-camera frames. Dream-1 runs go through inspect-robots
(see README.md).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=2400, help="run limit in control steps (20 Hz)")
    p.add_argument("--no-video", action="store_true", help="skip camera recording (faster, for debugging)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app

    import imageio.v2 as imageio
    from inspect_robots import Scene
    from inspect_robots.types import Action

    import can as C
    import cloth_bridge
    import scoring
    import task as T
    from demo import Demo
    from yam_arena import inspect_embodiment as wrapper

    wrapper._APP = app
    t0 = time.monotonic()
    emb = wrapper.YamArenaEmbodiment(task=str(HERE / "task.py"), object_jitter=0.0, control_hz=20,
                                     cam_width=a.width, cam_height=a.height)
    obs = emb.reset(Scene(id="dispose-of-stain-rags", instruction=T.INSTRUCTION))
    demo = Demo(cloth_bridge.state(emb._env, T.RAGS[0]).nodes)   # plan the pinch on the rag as measured
    writers = {}
    if not a.no_video:
        for cam in obs.images:
            writers[cam] = imageio.get_writer(a.out / f"{cam}.mp4", fps=20, codec="libx264",
                                              pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"])

    def record(o):
        for cam, w in writers.items():
            w.append_data(np.asarray(o.images[cam], dtype=np.uint8))

    from PIL import Image
    Image.fromarray(np.asarray(obs.images["top_cam"], dtype=np.uint8)).save(a.out / "first-top_cam.png")
    record(obs)
    log, ever_in_bin, info, steps, terminated = [], False, {}, 0, False
    for k in range(a.max_steps):
        res = emb.step(Action(data=demo.action(k / 20.0)))
        steps, info, terminated = k + 1, res.info, res.terminated
        record(res.observation)
        rags = [cloth_bridge.state(emb._env, n).nodes for n in T.RAGS]
        ever_in_bin |= any(scoring.rag_in_bin(n) for n in rags)
        if k % 20 == 0 or terminated:
            log.append(dict(step=k + 1, score=info.get("score"), success=info.get("success"),
                            lid_deg=float(C.can_state(emb._env, "oily_waste_can").lid_angle_deg),
                            rags_in_can=[scoring.rag_in_can(n) for n in rags],
                            rags_held=[bool(cloth_bridge.state(emb._env, nm).held) for nm in T.RAGS],
                            rag_centres=[n.mean(0).round(4).tolist() for n in rags],
                            rag_speed_mean_max=[[round(float(np.mean(np.linalg.norm(v, axis=1))), 4),
                                                 round(float(np.max(np.linalg.norm(v, axis=1))), 4)]
                                                for v in (cloth_bridge.state(emb._env, nm).velocities for nm in T.RAGS)],
                            rag_z_range_mm=[[round(1000 * (n[:, 2].min() - 0.7475)), round(1000 * (n[:, 2].max() - 0.7475))]
                                            for n in rags]))
            e = log[-1]
            print(f"step {e['step']:4d}  {time.monotonic() - t0:6.0f} s  score {e['score']}  lid {e['lid_deg']:5.1f}  "
                  f"held {e['rags_held']}  in_can {e['rags_in_can']}  centre {e['rag_centres']}", flush=True)
        if terminated:
            break
    np.savez(a.out / "final_rag_nodes.npz", **{n: cloth_bridge.state(emb._env, n).nodes for n in T.RAGS})
    obs = res.observation
    Image.fromarray(np.asarray(obs.images["top_cam"], dtype=np.uint8)).save(a.out / "last-top_cam.png")
    for w in writers.values():
        w.close()
    task_score = 1.0 if terminated and info.get("success") else float(info.get("score", 0.0))
    grips = {name: {side: [{k2: v for k2, v in e.items() if k2 != "indices"} for e in g.events]
                    for side, g in hands.items()} for name, hands in cloth_bridge.GRIPS.items()}
    result = dict(policy="scripted demo", instruction=T.INSTRUCTION, task_score=task_score,
                  terminated_by_success=bool(terminated and info.get("success")), steps=steps,
                  run_limit_steps=a.max_steps, any_rag_ever_in_bin=ever_in_bin,
                  ik_worst_mm={s: arm.worst_mm for s, arm in demo.arms.items()},
                  grip_events=grips, log=log, wall_s=time.monotonic() - t0,
                  videos=sorted(f"{c}.mp4" for c in writers))
    (a.out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: result[k] for k in ("task_score", "terminated_by_success", "steps", "wall_s")}))


if __name__ == "__main__":
    main()
