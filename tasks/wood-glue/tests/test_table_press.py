"""Joining uses one table-supported normal stroke, then both arms hold the wood in place."""
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wood_glue.bottle_visual import bottle_meshes
from wood_glue.controller import Controller
from wood_glue.scene import HEAD,LEG,TOP
from yam_arena import ik,station
from yam_arena.constants import READY


def test_bottle_visual_meshes_are_finite_with_valid_usd_names():
    for parent,name,vertices,faces,_ in bottle_meshes():
        assert re.fullmatch('[A-Za-z][A-Za-z0-9_]*',parent)
        assert re.fullmatch('[A-Za-z][A-Za-z0-9_]*',name)
        assert np.isfinite(vertices).all()
        assert faces.min() >= 0 and faces.max() < len(vertices)
        assert vertices[:,2].max() <= (.114 if parent == 'body' else .022)


def test_table_press_actions_have_no_commanded_shear_or_lift():
    c = Controller(mode='demo',accelerated_bond=True)
    from wood_glue.clearance import arm_clearance,wood_clearance,robot_hulls,commanded_joints
    points = []
    minimum_arm_clearance = minimum_wood_clearance = 50.
    pickup_overlap_frames = 0
    previous = None
    wood_placed = False
    bottle_returned = False
    terminal_actions = []
    for _ in range(2400):
        action = c.action()
        if c.phase in ('slide miters straight together on table','hold normal pressure on table'):
            tcp,_ = ik.tcp_pose(action[:6],station.BASES['left'])
            points.append(tcp)
            for spec in (HEAD,LEG):
                bottom = c.poses()[spec.name].world(spec.mesh()[0][:4])[:,2]
                assert np.max(np.abs(bottom-TOP)) < .001
        joints = commanded_joints(c)
        hulls = robot_hulls(joints)
        clearance,_ = arm_clearance(joints,hulls)
        wood_gap,_ = wood_clearance(joints,c.poses(),c.wood_contacts,hulls)
        minimum_wood_clearance = min(minimum_wood_clearance,wood_gap)
        if c.phase == 'pick up bottle and loose miter concurrently' and previous is not None:
            pickup_overlap_frames += int(np.linalg.norm(action[:6]-previous[:6])>1e-6 and
                                         np.linalg.norm(action[7:13]-previous[7:13])>1e-6)
        if c.phase == 'place coated casing flat on table': wood_placed = True
        if c.phase == 'return bottle upright':
            assert wood_placed
            for spec in (LEG,):
                assert np.max(np.abs(c.poses()[spec.name].world(spec.mesh()[0][:4])[:,2]-TOP)) < .001
            bottle_returned = True
        if c.phase == 'hold assembled corner in place': terminal_actions.append(action.copy())
        assert c.phase not in ('release corner on table','withdraw right arm for clear view','withdraw left arm for clear view')
        previous = action.copy()
        minimum_arm_clearance = min(minimum_arm_clearance,clearance)
        if c.done: break
    assert c.done and c.steps < 2400
    assert minimum_arm_clearance >= 10.
    assert minimum_wood_clearance >= 10.
    assert pickup_overlap_frames >= 100 and bottle_returned
    assert c.steps < 2100
    points = np.array(points)
    assert len(points) == 120
    assert .003 < np.ptp(points[:,1]) < .005
    assert np.ptp(points[:,0]) < .0001
    assert np.ptp(points[:,2]) < .0001
    assert c.join_motion['commanded_tangent_max_mm'] < 1e-8
    assert c.join_motion['wet_shear_max_mm'] < 1e-8
    assert len(terminal_actions) == 40
    assert np.ptp(np.array(terminal_actions),axis=0).max() < 1e-12
    assert c.left.grip < .65 and c.right.grip < .65
    assert c.wood_contacts == {'left':LEG.name,'right':HEAD.name}


def test_bottle_base_is_directly_on_the_table_without_support_objects():
    from wood_glue.scene import BOTTLE_BASE,OBJECTS
    assert BOTTLE_BASE[2] == TOP
    assert {spec.name for spec in OBJECTS} == {'head_casing','loose_leg','glue_bottle','bench_cover'}


def test_assembly_seam_runs_front_to_back():
    from wood_glue.assembly import ASSEMBLY_HEAD_YAW
    from wood_glue.geometry import yaw_matrix
    surface = next(s for s in HEAD.mesh()[2] if s.target)
    seam = yaw_matrix(ASSEMBLY_HEAD_YAW)@surface.u
    assert abs(seam[0]) > .999999
    assert abs(seam[1]) < 1e-10
