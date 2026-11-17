"""Reject short blobs, dry joints, wrong angles, offsets, and premature release credit."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wood_glue.assembly import AssemblyModel,ContactDwell,bead_status,joint_geometry,mating_leg,wet_contact_length
from wood_glue.controller import Controller,LEG_GRASP_LOCAL
from wood_glue.geometry import Pose,yaw_matrix
from wood_glue.scene import HEAD,LEG,TOP,all_initial_poses,surfaces


def model_and_joint():
    model = AssemblyModel(surfaces(),require_bond=False)
    poses = all_initial_poses()
    poses[LEG.name] = mating_leg(poses[HEAD.name])
    film = model.films[LEG.name+'/miter']
    film.volume[4:-4,6:10] = film.area*.00017
    return model,poses


def test_long_bead_qualifies_without_broad_film():
    m,_ = model_and_joint()
    f = m.films[LEG.name+'/miter']
    assert bead_status(f)['qualified']
    assert f.coverage(m.settings) < .3
    assert m.coating_score() == .5


def test_adhesion_requires_glue_to_touch_not_just_nearby_faces():
    m,p = model_and_joint()
    assert joint_geometry(p)['seated']
    assert wet_contact_length(m,p) == 0.  # 0.5 mm gap is wider than the 0.17 mm deposited bead.
    p[LEG.name] = mating_leg(p[HEAD.name],gap=.00010)
    assert wet_contact_length(m,p) >= .065


def test_contact_dwell_uses_pose_stability_instead_of_substep_velocity_noise():
    _,p = model_and_joint()
    dwell = ContactDwell()
    for t in range(40):
        p[LEG.name].angular = np.array([.15,0.,0.])  # A noisy solver endpoint with a stationary pose.
        dwell.advance(.05,True,p)
    assert dwell.time == pytest.approx(2.)
    dwell.advance(.05,False,p)
    assert dwell.time == 0.


@pytest.mark.parametrize('motion',['translation','rotation'])
def test_contact_dwell_rejects_accumulated_drift(motion):
    _,p = model_and_joint()
    dwell = ContactDwell()
    for t in range(80):
        if motion == 'translation': p[LEG.name].pos += [.0001,0.,0.]
        else: p[LEG.name].rot = yaw_matrix(np.deg2rad(.1))@p[LEG.name].rot
        dwell.advance(.05,True,p)
        assert dwell.time < 2.


def test_same_volume_in_short_blob_fails_bead_requirement():
    m,_ = model_and_joint()
    f = m.films[LEG.name+'/miter']
    volume = f.volume.sum()
    f.volume.fill(0.)
    f.volume[35:45,:] = volume/160
    assert f.volume.sum() == pytest.approx(volume)
    assert not bead_status(f)['qualified']


def test_large_break_in_bead_does_not_qualify():
    m,_ = model_and_joint()
    f = m.films[LEG.name+'/miter']
    f.volume[32:43] = 0
    assert bead_status(f)['largest_gap_mm'] > 8.
    assert not bead_status(f)['qualified']


@pytest.mark.parametrize('error',['angle','gap','flush','tangent','dry','spill'])
def test_wrong_or_dry_joint_cannot_pass(error):
    m,p = model_and_joint()
    if error == 'angle': p[LEG.name].rot = yaw_matrix(.15)@p[LEG.name].rot
    if error == 'gap': p[LEG.name].pos += [-.01,0.,0.]
    if error == 'flush': p[LEG.name].pos += [0.,0.,.005]
    if error == 'tangent': p[LEG.name].pos += [0.,.01,0.]
    if error == 'dry': m.reset()
    if error == 'spill': m.invalidate('off_target_glue')
    for t in range(25): m.joint_step(t,.05,p,True,True)
    assert m.score() < .75


def test_release_credit_requires_one_continuous_second():
    m,p = model_and_joint()
    assert joint_geometry(p)['seated']
    m.joint_step(0,.05,p,False,True)
    assert m.score() == .75
    for t in range(1,20): m.joint_step(t,.05,p,True,True)
    assert m.score() == .75
    m.joint_step(20,.05,p,False,True)
    assert m.release_s == 0
    for t in range(21,41): m.joint_step(t,.05,p,True,True)
    assert m.score() == 1.


def test_accelerated_mode_requires_an_intact_native_bond():
    m,p = model_and_joint()
    m.require_bond = True
    for t in range(25): m.joint_step(t,.05,p,True,True,bonded=False)
    assert m.score() == .75
    for t in range(25,45): m.joint_step(t,.05,p,True,True,bonded=True)
    assert m.score() == 1.
    m.joint_step(45,.05,p,True,True,bonded=False)
    assert m.score() == .75


def test_boards_start_flat_on_table_and_fingertips_clear_it():
    p = all_initial_poses()
    for spec in (HEAD,LEG):
        assert spec.free
        assert p[spec.name].world(spec.mesh()[0])[:,2].min() == pytest.approx(TOP)
    assert LEG_GRASP_LOCAL[2]-.013 >= .002


@pytest.mark.parametrize('bond',[False,True])
def test_entire_kinematic_assembly_path_within_budget(bond):
    c = Controller(mode='demo',accelerated_bond=bond)
    for _ in range(2400):
        c.action()
        if c.done: break
    assert c.done and c.steps < 2400
    assert joint_geometry(c.poses())['seated']
    assert max(c.left.worst_mm,c.right.worst_mm) < .1
