"""Save the native calibrated top-camera setup screenshot before any actions."""
import argparse,atexit
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer-gpu',type=int,required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    from isaaclab.app import AppLauncher
    from yam_arena import inspect_embodiment as implementation
    implementation._APP=AppLauncher(headless=True,device='cpu',enable_cameras=True,
        kit_args=f'--/renderer/activeGpu={args.renderer_gpu} --/renderer/multiGpu/enabled=false --/carb/tasking/plugin/threadCount=4 --/omni/tbb/globalcontrol/maxThreadCount=4').app
    atexit.register(implementation._close_app_at_exit)
    from inspect_robots import Scene
    import imageio.v2 as imageio
    embodiment=implementation.YamArenaEmbodiment(task=str(Path(__file__).parent/'task.py'),
        device='cpu',control_hz=20,object_jitter=0)
    embodiment.reset(Scene(id='electrical-box-setup',instruction='Prebend both mounting flaps.'))
    data=embodiment._base.scene['top_cam'].data.output['rgb']
    rgb=(data.torch if hasattr(data,'torch') else data).cpu().numpy()[0,:,:,:3]
    imageio.imwrite(args.output,rgb.astype('uint8'));embodiment.close()

if __name__=='__main__':main()
