"""Regression for the measured return collision that an arm-only monitor missed."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wood_glue.clearance import arm_clearance,wood_clearance
from wood_glue.geometry import Pose

JOINTS = {'left': {'joint1': -1.2682840824127197, 'joint2': 2.7353780269622803, 'joint3': 2.1119613647460938, 'joint4': 0.7978162169456482, 'joint5': -1.2728893756866455, 'joint6': -0.7009979486465454, 'joint7': -0.028605399653315544, 'joint8': -0.028287090361118317}, 'right': {'joint1': -0.1640104502439499, 'joint2': 2.356372594833374, 'joint3': 1.3122838735580444, 'joint4': 1.002339243888855, 'joint5': -0.16809958219528198, 'joint6': 0.0014225280610844493, 'joint7': -0.027457041665911674, 'joint8': -0.027508169412612915}}
POSES = {'head_casing': {'pos': [0.24484744668006897, -0.03984750062227249, 0.7474995255470276], 'rotation': [[0.7071069942666119, 0.7071065681041055, -1.8087543720777737e-06], [-0.7071065681050948, 0.7071069942679248, 1.265696363970044e-07], [1.3684810886277528e-06, 1.1894838214268964e-06, 0.9999999999983562]]}, 'loose_leg': {'pos': [0.348111093044281, -0.10137908160686493, 1.011694312095642], 'rotation': [[0.034293884017298604, -0.03236232278941831, -0.9988876861703129], [-0.7587420400643321, -0.651372440178271, -0.004945787623807041], [-0.6504878524292543, 0.7580676910671486, -0.046892745730001906]]}}

def test_recorded_return_collision_is_detected_even_when_arms_are_clear(monkeypatch):
    # Preserve the recording's square-ended casing geometry. The new frame
    # rails have a second miter, which removes the corner that collided here.
    from scipy.spatial import ConvexHull
    from wood_glue.geometry import prism
    def recorded_geometry(name):
        vertices=prism(name,[(0.,-.0285),(.300,-.0285),(.300,.0285),(.057,.0285)],.016)[0]
        equations=ConvexHull(vertices).equations
        return vertices,equations[:,:3],-equations[:,3]
    monkeypatch.setattr('wood_glue.clearance.wood_geometry',recorded_geometry)
    poses = {n:Pose(p['pos'],np.array(p['rotation'])) for n,p in POSES.items()}
    arm_gap,_ = arm_clearance(JOINTS)
    wood_gap,pair = wood_clearance(JOINTS,poses,{'left':'loose_leg'})
    assert arm_gap >= 5.
    assert wood_gap == 0.
    assert pair == ('right','link3','loose_leg')
