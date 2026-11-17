"""Two-arm path and conserved deposition on the lifted cut face (CPU only)."""
from pathlib import Path
import copy
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wood_glue.controller import Controller, PICK_ROT, REST_GAP, LEG_GRIP_YAW, GRIP_FACTOR
from wood_glue.milestone import held_glue_status
from wood_glue.model import GlueModel
from wood_glue.assembly import bead_status
from wood_glue.scene import (LEG, TOP, TIP_LOCAL, GRASP_LOCAL, all_initial_poses, surfaces)


def test_left_fingers_close_across_the_width_of_the_leg():
    from yam_arena.demo import grip_rotation
    from wood_glue.geometry import yaw_matrix
    width_axis = yaw_matrix(LEG.yaw)[:,1]
    assert abs(grip_rotation(LEG_GRIP_YAW)[:,0]@width_axis) == pytest.approx(1.)


@pytest.fixture(scope="module")
def offline_coating():
    controller = Controller(mode="hold-and-glue")
    model = GlueModel(surfaces())
    poses = all_initial_poses()
    minimum = 10.
    for tick in range(2400):
        controller.action()
        if controller.phase == "present cut face upward":
            minimum = min(minimum,controller.held_leg_pose().world(LEG.mesh()[0])[:,2].min()-TOP)
        if controller.phase == "bring bottle above held miter" or model.last_tick is not None:
            poses[LEG.name] = controller.held_leg_pose()
            r = controller.right.rot@PICK_ROT.T
            tip = controller.right.pos+r@(TIP_LOCAL-GRASP_LOCAL)
            compression = (REST_GAP-.0939*controller.right.grip)/GRIP_FACTOR
            model.advance(tick,.05,tip,r,compression,poses)
        if controller.done:
            break
    return controller,model,poses,minimum


def test_two_arm_path_clears_bench_and_finishes_in_budget(offline_coating):
    controller,_,_,minimum = offline_coating
    assert controller.done and controller.steps < 2400
    assert minimum > .015
    assert max(controller.left.worst_mm,controller.right.worst_mm) < .1


def test_only_held_miter_gets_glue_and_volume_is_conserved(offline_coating):
    _,model,_,_ = offline_coating
    assert not model.failed
    assert .03 < model.emitted*1e6 < .08
    assert abs(model.metrics()["conservation_error_ml"]) < 1e-9
    assert model.films["head_casing/miter"].volume.sum() == 0
    film = model.films[LEG.name+"/miter"]
    assert bead_status(film)['qualified']
    assert film.coverage(model.settings) < .5  # A long bead is sufficient; a full film is not needed.


@pytest.mark.parametrize("failure",[None,"dry","on_support","not_grasped","spill"])
def test_held_edge_milestone_requires_grasp_lift_and_deposition(offline_coating,failure):
    controller,original,poses,_ = offline_coating
    model = copy.deepcopy(original)
    leg = poses[LEG.name]
    low = leg.world(LEG.mesh()[0]).min(0)
    tcp = controller.left.pos.copy()
    if failure == "dry": model.reset()
    if failure == "on_support": low[2] = TOP
    if failure == "not_grasped": tcp += [.05,0.,0.]
    if failure == "spill": model.invalidate("off_target_glue")
    status = held_glue_status(leg,low,tcp,.057,{"qualified":True},model)
    assert status["qualified"] == (failure is None)


def test_glue_stroke_runs_across_top_camera_view(offline_coating):
    controller,_,poses,_ = offline_coating
    surface = next(s for s in LEG.mesh()[2] if s.target)
    pose = poses[LEG.name]
    along = pose.rot@surface.u
    normal = pose.rot@surface.normal
    assert abs(along[1]) > .99
    assert abs(along[0]) < .001
    assert normal[0] < -.03  # The cut faces slightly toward the policy camera.
    assert normal[1] > .08  # This tilt keeps the dispensing wrist within its working range.
    assert normal[2] > np.cos(np.deg2rad(8))
