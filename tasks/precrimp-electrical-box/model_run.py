"""Run one stock Dream-1 trial and record native top-camera pixels and state.

The observer below only reads state after ordinary SDK actions. It never changes
the policy's observations, actions, physics, termination or score.
"""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer-gpu', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.renderer_gpu):
        parser.error('Check GPU availability, then pin CUDA_VISIBLE_DEVICES to --renderer-gpu.')
    args.output.mkdir(parents=True, exist_ok=False)
    folder = Path(__file__).resolve().parent
    sources = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(folder.rglob('*'))
               if p.is_file() and (p.suffix in {'.py', '.usda'} or p.name == 'task.json')}
    (args.output / 'source-hashes.json').write_text(json.dumps(sources, indent=2) + '\n')

    import imageio.v2 as imageio
    import numpy as np
    from isaaclab.app import AppLauncher
    import yam_arena.inspect_embodiment as implementation
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    implementation._APP = AppLauncher(
        headless=True, device='cpu', enable_cameras=True,
        kit_args=f'--/renderer/activeGpu={args.renderer_gpu} '
                 '--/renderer/multiGpu/enabled=false '
                 '--/carb/tasking/plugin/threadCount=4 '
                 '--/omni/tbb/globalcontrol/maxThreadCount=4').app
    atexit.register(implementation._close_app_at_exit)
    original_step = implementation.YamArenaEmbodiment.step
    writer = imageio.get_writer(str(args.output / 'model_run.mp4'), fps=20,
        codec='libx264', quality=8, macro_block_size=2,
        ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    trace = []
    actions = []
    reports = []
    indices = None
    peak_effort = {side: np.zeros(8) for side in ['left', 'right']}
    peak_contact = {side: np.zeros(2) for side in ['left', 'right']}

    def array(value):
        return (value.torch if hasattr(value, 'torch') else value).cpu().numpy()

    def observe(self, action):
        nonlocal indices
        result = original_step(self, action)
        box = self._base.scene['device_box']
        if indices is None:
            names = [f'{side}_fold_{k}' for side in ['front', 'rear'] for k in range(15)]
            indices = (names, box.find_joints(names, preserve_order=True)[0],
                {side: self._base.scene[side + '_robot'].find_joints(
                    ARM_JOINTS + FINGER_JOINTS, preserve_order=True)[0]
                 for side in ['left', 'right']})
        names, folds, robots = indices
        pose = array(box.data.root_link_pose_w)[0]
        frame = {'t': float(self._t), 'box_position': pose[:3].tolist(),
            'box_quat_wxyz': pose[[6, 3, 4, 5]].tolist(),
            'angles_deg': dict(zip(names, np.degrees(array(box.data.joint_pos)[0, folds]).tolist())),
            'robot_joint_positions': {side: array(self._base.scene[side + '_robot'].data.joint_pos)[0, ids].tolist()
                                      for side, ids in robots.items()},
            'task_score': float(result.info.get('score', 0.))}
        trace.append(frame)
        actions.append(np.asarray(action.data).tolist())
        reports.append(result.info.get('precrimp', {}))
        for side, ids in robots.items():
            robot = self._base.scene[side + '_robot']
            peak_effort[side] = np.maximum(peak_effort[side], np.abs(array(robot.data.applied_torque)[0, ids]))
            sensor = self._base.scene[side + '_contacts']
            forces = array(sensor.data.net_forces_w_history)[0]
            tips = [sensor.body_names.index('tip_' + finger) for finger in ['left', 'right']]
            peak_contact[side] = np.maximum(peak_contact[side], np.linalg.norm(forces[:, tips], axis=-1).max(axis=0))
        rgb = array(self._base.scene['top_cam'].data.output['rgb'])[0, :, :, :3].astype('uint8')
        writer.append_data(rgb)
        if len(trace) % 100 == 0:
            print(f'MODEL_PROGRESS steps={len(trace)} time={self._t:.2f} score={frame["task_score"]}', flush=True)
        return result

    implementation.YamArenaEmbodiment.step = observe
    instruction = json.loads((folder / 'task.json').read_text())['instruction']
    start = time.monotonic()
    error = None
    exit_code = None
    try:
        from inspect_robots.cli import main as run_cli
        exit_code = run_cli(['run', '--embodiment', 'yam_arena', '--policy', 'dream',
            '-E', f'task={folder / "task.py"}', '-E', 'device=cpu',
            '-E', 'object_jitter=0', '-E', 'control_hz=20',
            '-E', 'cam_width=320', '-E', 'cam_height=176',
            '--instruction', instruction, '--scorer', 'task_score', '--max-steps', '4800',
            '--store-frames', '--seed', '0', '--no-prompt', '--log-dir', str(args.output)])
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        writer.close()
        (args.output / 'trace.json').write_text(json.dumps(trace) + '\n')
        (args.output / 'actions.json').write_text(json.dumps(actions) + '\n')
        (args.output / 'scoring-history.json').write_text(json.dumps(reports) + '\n')
        receipt = {'model': 'dreamlabs/dream-1-beta', 'seed': 0,
            'physical_renderer_gpu': args.renderer_gpu, 'physics_device': 'cpu',
            'control_hz': 20, 'control_steps': len(trace),
            'simulated_seconds': trace[-1]['t'] if trace else 0.,
            'wall_seconds': time.monotonic() - start, 'cli_exit_code': exit_code,
            'error': error, 'task_score': trace[-1]['task_score'] if trace else None,
            'maximum_score': max((item['task_score'] for item in trace), default=0.),
            'final_scoring_report': reports[-1] if reports else None,
            'peak_measured_joint_efforts': {side: value.tolist() for side, value in peak_effort.items()},
            'peak_finger_normal_contact_n': {side: value.tolist() for side, value in peak_contact.items()},
            'video': 'model_run.mp4', 'camera': 'native calibrated top_cam; original steel materials'}
        (args.output / 'native_result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
