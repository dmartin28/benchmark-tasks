"""Checks for the current pickup-and-invert milestone, independent of GPU execution."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wood_glue.controller import Controller, PICKUP_ROT, PICK_ROT
from wood_glue.geometry import Pose
from wood_glue.milestone import pose_status
from wood_glue.scene import BOTTLE, BOTTLE_RADII, BOTTLE_EMPTY_MASS, GLUE_DENSITY_G_ML, GRASP_LOCAL, TOP, yaw_matrix


def test_commercial_bottle_envelope_mass_and_round_outlet():
    from wood_glue.scene import NOZZLE_DIAMETER, BOTTLE_UNCAPPED_HEIGHT, BOTTLE_CAP_HEIGHT
    from wood_glue.bottle_visual import bottle_meshes
    assert 2*BOTTLE_RADII == pytest.approx(np.array([.973,1.976])*.0254)
    assert BOTTLE_UNCAPPED_HEIGHT == pytest.approx(3.686*.0254)
    assert BOTTLE_CAP_HEIGHT == pytest.approx(1.25*.0254)
    assert NOZZLE_DIAMETER == pytest.approx(.000762)
    assert BOTTLE.mass+sum(s.mass for s in BOTTLE.sliders) == pytest.approx(BOTTLE_EMPTY_MASS+.020*GLUE_DENSITY_G_ML)
    shell=next(points for _,name,points,_,_ in bottle_meshes() if name=='shell')
    assert np.ptp(shell[:,:2],axis=0) == pytest.approx(2*BOTTLE_RADII)


def test_fingers_close_across_the_squeeze_panels():
    # URDF finger travel is along the TCP x axis. This caught the original incorrect grasp.
    expected = yaw_matrix(BOTTLE.yaw)[:,1]
    assert abs(PICKUP_ROT[:,0] @ expected) == pytest.approx(1.)
    assert abs(PICKUP_ROT[2,2]) < 1e-10


def test_panel_springs_stay_within_the_grippers_effort_limit():
    from yam_arena.constants import GRIPPER
    for panel in BOTTLE.sliders:
        assert panel.rate*max(abs(panel.lower),abs(panel.upper)) < GRIPPER["effort"]


def test_complete_offline_pickup_path_and_inversion():
    from yam_arena import ik
    controller = Controller()
    minimum = 10.
    for _ in range(500):
        action = controller.action()
        assert np.all(action[7:13] >= ik.LOWER)
        assert np.all(action[7:13] <= ik.UPPER)
        if controller.phase in ("turn bottle upside down","hold bottle upside down"):
            arm = controller.right
            bottle_rot = arm.rot @ PICK_ROT.T
            origin = arm.pos-bottle_rot@GRASP_LOCAL
            # Conservative box includes core, cap, nozzle and squeeze-panel travel envelope.
            points = np.array([[x,y,z] for x in (-.0175,.0175) for y in (-.029,.029) for z in (0.,.114)])
            minimum = min(minimum,(points@bottle_rot.T+origin)[:,2].min()-TOP)
        if controller.done: break
    assert controller.done and controller.steps <= 500
    assert controller.right.worst_mm < .1
    assert minimum >= .03
    assert bottle_rot[2,2] < -np.cos(np.deg2rad(5))


@pytest.mark.parametrize("change",["upright","bench","distant_hand","open_hand","falling","spinning"])
def test_false_milestones_do_not_pass(change):
    r = np.diag([1.,-1.,-1.])
    p = Pose([.3,-.3,TOP+.2],r)
    low = np.array([.28,-.33,TOP+.08])
    tcp = p.world(GRASP_LOCAL)
    gap = .055
    if change == "upright": p.rot = np.eye(3)
    if change == "bench": low[2] = TOP
    if change == "distant_hand": tcp += [.03,0.,0.]
    if change == "open_hand": gap = .09
    if change == "falling": p.linear = np.array([0.,0.,-.1])
    if change == "spinning": p.angular = np.array([0.,1.,0.])
    assert not pose_status(p,low,tcp,gap)["qualified"]


def test_measured_inverted_hold_qualifies():
    p = Pose([.3,-.3,TOP+.2],np.diag([1.,-1.,-1.]))
    result = pose_status(p,np.array([.28,-.33,TOP+.08]),p.world(GRASP_LOCAL),.055)
    assert result["qualified"]
    assert result["inversion_deg"] == pytest.approx(180.)


def test_original_height_bottle_pickup_clears_the_table():
    from yam_arena.analysis import world_hulls,fk_link_poses_world
    from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS
    from wood_glue.scene import BOTTLE_BASE,TIP_LOCAL
    assert TIP_LOCAL[2] == pytest.approx(.114)
    assert BOTTLE_BASE[2] == TOP
    c = Controller()
    minimum = 1.
    while not c.done:
        c.action()
        q = dict(zip(ARM_JOINTS,c.right.q))
        q.update(dict.fromkeys(FINGER_JOINTS,-.0939*c.right.grip/2))
        hulls = world_hulls(fk_link_poses_world(q,c.right.base))
        minimum = min(minimum,*[h.vertices[:,2].min()-TOP for name,h in hulls if name not in ('base','link1')])
    assert minimum >= .005
