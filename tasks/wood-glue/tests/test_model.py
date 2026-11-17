"""Physical ledger, geometry and failure scenarios; no Isaac Sim or GPU."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wood_glue.geometry import Pose, Surface, first_hit, rotation, yaw_matrix
from wood_glue.model import GlueModel, Settings
from wood_glue.scene import (FACE_LENGTH, HEAD, LEG, OBJECTS, TOP, all_initial_poses, surfaces)


def setup(cell=.001):
    return GlueModel(surfaces(),Settings(cell=cell)),all_initial_poses()


def aim(model,poses,index=0,uv=None):
    film = list(model.films.values())[index]
    s,p = film.surface,poses[film.surface.owner]
    uv = np.array([FACE_LENGTH/2,.008]) if uv is None else np.array(uv)
    point = s.origin+s.u*uv[0]+s.v*uv[1]
    # Lip x spans the height; lip y spans along the miter, emitting toward the wood.
    rot = p.rot @ np.column_stack((-s.v,-s.u,-s.normal))
    assert np.linalg.det(rot) == pytest.approx(1.)
    return p.world(point),rot


def fill(model,index,coverage=1.,h=.00016):
    f = list(model.films.values())[index]
    indices = np.flatnonzero(f.mask)
    f.volume.flat[indices[:round(len(indices)*coverage)]] = f.area*h


def test_cut_geometry_closed_and_correct_area():
    for spec in (HEAD,LEG):
        v,f,ss = spec.mesh()
        edges = np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
        _,counts = np.unique(edges,axis=0,return_counts=True)
        assert np.all(counts == 2)
        s = next(s for s in ss if s.target)
        assert np.linalg.norm(s.corners[1]-s.corners[0]) == pytest.approx(.057*np.sqrt(2))
        assert np.linalg.norm(s.corners[-1]-s.corners[0]) == pytest.approx(.016)
        assert all_initial_poses()[spec.name].rot @ s.normal == pytest.approx([-1,0,0])
        volume = np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6
        assert volume == pytest.approx((.300*.057-.057**2)*.016)


def test_pose_and_quaternion_validation():
    p = Pose([1.,2.,3.],yaw_matrix(.7))
    pts = np.array([[.1,.2,.3],[-1.,0.,1.]])
    assert p.local(p.world(pts)) == pytest.approx(pts)
    assert rotation([0,0,0,2]) == pytest.approx(np.eye(3))
    with pytest.raises(ValueError): rotation([0,0,0,0])


def test_first_surface_occludes_target():
    m,poses = setup()
    tip,r = aim(m,poses)
    n = r[:,2]
    assert first_hit(tip-.001*n,tip+.001*n,m.surfaces,poses)[1].target
    s = Surface("blocker","blocker",np.array([[0,-1,-1],[0,1,-1],[0,1,1],[0,-1,1]])[::-1])
    poses["blocker"] = Pose(tip-.0005*n)
    hit = first_hit(tip-.001*n,tip+.001*n,[s,*m.surfaces],poses)
    assert hit[1].name == "blocker"


def test_dry_nozzle_motion_and_carry_grip_never_dispense():
    m,p = setup()
    tip,r = aim(m,p)
    for k in range(5): m.advance(k,.05,tip+[0,k*.001,0],r,.003,p)
    assert m.emitted == 0
    assert m.score() == 0


def test_initial_compression_is_baseline_not_free_glue():
    m,p = setup()
    tip,r = aim(m,p)
    m.advance(0,.05,tip,r,.009,p)
    assert m.emitted == 0


def test_squeeze_release_and_stationary_hold():
    m,p = setup()
    tip,r = aim(m,p)
    m.advance(0,.05,tip,r,0.,p)
    m.advance(1,.05,tip,r,.006,p)
    emitted = m.emitted
    assert emitted*1e6 == pytest.approx(.2)
    for k,c in enumerate([.006,.006,0.],2): m.advance(k,.05,tip,r,c,p)
    assert m.emitted == emitted
    m.advance(5,.05,tip,r,.006,p)
    assert m.emitted == pytest.approx(2*emitted)
    assert abs(m.metrics()["conservation_error_ml"]) < 1e-10


def test_reservoir_cannot_create_glue_after_depletion():
    m,p = setup()
    m.remaining = .01e-6
    tip,r = aim(m,p)
    m.advance(0,.05,tip,r,0.,p)
    m.advance(1,.05,tip,r,.012,p)
    assert m.remaining == 0
    assert m.emitted == pytest.approx(.01e-6)


def test_spreading_preserves_volume_and_requires_contact():
    m,p = setup()
    tip,r = aim(m,p)
    f = list(m.films.values())[0]
    i,j = np.array(f.shape)//2
    f.volume[i,j] = .02e-6
    before = f.volume.sum()
    m._spread(tip-r[:,2]*.02,r,p)
    assert np.count_nonzero(f.volume) == 1
    m._spread(tip,r,p)
    assert np.count_nonzero(f.volume) > 1
    assert f.volume.sum() == pytest.approx(before)


def test_duplicate_ticks_and_scoring_are_read_only():
    m,p = setup()
    tip,r = aim(m,p)
    m.advance(0,.05,tip,r,0.,p)
    m.advance(1,.05,tip,r,.006,p)
    snapshot = m.metrics()
    m.advance(1,.05,tip,r,.012,p)
    for _ in range(10): m.score(); m.metrics()
    assert m.metrics() == snapshot
    with pytest.raises(ValueError): m.advance(0,.05,tip,r,0.,p)


@pytest.mark.parametrize("a,b,h,want",[(0,0,.00016,0),(.81,0,.00016,.25),
    (.81,.81,.00016,.5),(.96,.96,.00016,.75),(1,1,.00005,0),(1,1,.0003,0)])
def test_score_tiers_and_starved_or_thick_film(a,b,h,want):
    m,_ = setup()
    fill(m,0,a,h);fill(m,1,b,h)
    assert m.score() == want


def test_puddle_disqualifies_only_its_face():
    m,_ = setup()
    fill(m,0);fill(m,1)
    list(m.films.values())[0].volume[0,0] = .00051*list(m.films.values())[0].area
    assert m.score() == .25


def test_release_hold_resets_when_endpoint_is_lost():
    m,p = setup()
    fill(m,0);fill(m,1)
    tip,r = aim(m,p)
    # Keep the nozzle away from the film so the test doesn't redistribute the fixture.
    tip -= r[:,2]*.1
    for k in range(19): m.advance(k,.05,tip,r,0.,p,ready=True)
    assert m.score() == .75
    m.advance(19,.05,tip,r,0.,p,ready=False)
    assert m.held == 0
    for k in range(20,40): m.advance(k,.05,tip,r,0.,p,ready=True)
    assert m.score() == 1


def test_contamination_is_permanent_and_volume_accounted():
    m,p = setup()
    fill(m,0);fill(m,1)
    surface = next(s for s in m.surfaces if s.name == "bench/top")
    m._land(surface,np.array([.1,.1,TOP]),1e-12)
    assert m.failed and m.score() == 0
    m.held = 5
    assert m.score() == 0
    assert "off_target_glue" in m.violations


def test_missed_emission_falls_to_bench():
    m,p = setup()
    tip = np.array([.1,-.4,TOP+.01])
    r = np.diag([1.,-1.,-1.])
    m.advance(0,.05,tip,r,0.,p)
    m.advance(1,.05,tip,r,.006,p)
    for k in range(2,5): m.advance(k,.05,tip,r,.006,p)
    assert m.failed
    assert m.contamination_volume > 0
    assert not m.drops
    assert abs(m.metrics()["conservation_error_ml"]) < 1e-9


def test_local_film_moves_with_leg_without_spawning_or_losing_glue():
    m,p = setup()
    f = m.films["loose_leg/miter"]
    f.volume[2,3] = 1e-9
    point = f.points[2,3]
    moved = Pose(p[LEG.name].pos+[0,.1,.1],yaw_matrix(.4)@p[LEG.name].rot)
    assert moved.local(moved.world(point)) == pytest.approx(point)
    assert f.volume.sum() == 1e-9


def test_reset_removes_every_trial_dependent_value():
    m,p = setup()
    tip,r = aim(m,p)
    m.advance(0,.05,tip,r,0.,p)
    m.advance(1,.05,tip,r,.012,p)
    m.invalidate("object_dropped")
    m.held = 2
    m.reset()
    assert m.metrics() == GlueModel(surfaces()).metrics()
    assert m.previous is None and m.last_tick is None


def test_task_loads_without_simulator():
    from yam_arena.task_file import load
    t = load(Path(__file__).resolve().parents[1]/"task.py")
    assert set(t.free_objects) == {"head_casing","loose_leg","glue_bottle"}
    assert t.module.score({}) == 0
    assert "isaaclab.sim" not in sys.modules
    assert len(OBJECTS) == 4
    # Arena's Pose.to_tensor infers float64 from NumPy scalars, which Warp rejects on reset.
    assert all(type(v) is float for obj in OBJECTS for v in obj.pos)
