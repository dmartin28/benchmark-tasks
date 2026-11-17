"""Physical-judging rubric: dose scale, visible geometry, recoverable wood, and real pad locations."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import ConvexHull

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wood_glue.observable import ObservableAssemblyModel, bead_status, joint_status, hand_status
from wood_glue.assembly import mating_leg
from wood_glue.scene import HEAD,LEG,WIDTH,THICK,TOP,COVER,COVER_THICKNESS,GLUE_DENSITY_G_ML,all_initial_poses,surfaces


def fixture():
    model = ObservableAssemblyModel(surfaces())
    poses = all_initial_poses()
    poses[LEG.name] = mating_leg(poses[HEAD.name],gap=.0001)
    film = model.films[LEG.name+'/miter']
    film.volume[4:-4,6:10] = film.area*.00017
    model.emitted = film.volume.sum()
    model.remaining -= model.emitted
    return model,poses


def settle(model,poses,start=0,steps=40,held=True,parked=True):
    for tick in range(start,start+steps):
        model.joint_step(tick,.05,poses,False,parked,held=held)


def test_full_score_without_internal_wet_contact_or_chemical_bond():
    model,poses = fixture()
    # A narrow gauge gap is judged by the physical rubric, not the hidden film-contact solver.
    poses[LEG.name] = mating_leg(poses[HEAD.name],gap=.0004)
    settle(model,poses)
    assert model.score()==1 and model.held_s==pytest.approx(2)
    assert model.metrics()['scale_dose_g']==.05
    assert not model.bonded


def test_visible_length_and_mass_are_both_required():
    model,poses = fixture()
    film = model.films[LEG.name+'/miter']
    assert bead_status(film)['qualified']
    assert model.coating_score()==.5
    film.volume *= .01
    model.emitted = film.volume.sum()
    assert bead_status(film)['qualified']  # A rendered trace alone cannot satisfy the dose check.
    assert model.coating_score()==.25
    film.volume.fill(0)
    assert model.coating_score()==0


def test_same_dose_puddle_and_dry_gap_fail_useful_bead():
    model,_ = fixture()
    film = model.films[LEG.name+'/miter']
    volume = film.volume.sum()
    film.volume.fill(0)
    film.volume[32:43,:] = volume/(11*16)
    assert not bead_status(film)['qualified']
    model,_ = fixture()
    film = model.films[LEG.name+'/miter']
    film.volume[32:39] = 0
    assert bead_status(film)['largest_gap_mm']>5
    assert not bead_status(film)['qualified']


def test_zigzag_is_accepted_and_path_length_does_not_replace_cut_coverage():
    model,_ = fixture()
    film = model.films[LEG.name+'/miter']
    film.volume.fill(0)
    for i in range(4,77):
        j=3+abs((i//4)%8-4)
        film.volume[i,j:j+3] = film.area*.0002
    assert bead_status(film)['qualified']


def test_stray_glue_caps_credit_and_small_spots_do_not_zero_progress():
    model,poses = fixture()
    cover = next(s for s in model.surfaces if s.name=='bench_cover/front')
    for i in range(26):model._land(cover,np.array([-.4+i*.002,.5,COVER_THICKNESS/2]),1e-12)
    settle(model,poses)
    assert not model.failed
    assert model.cleanliness_mm2==pytest.approx(26)
    assert model.score()==.5
    model,poses = fixture()
    cover = next(s for s in model.surfaces if s.name=='bench_cover/front')
    model._land(cover,np.array([-.4,.5,COVER_THICKNESS/2]),1e-12)
    settle(model,poses)
    assert model.score()==1 and model.cleanliness_mm2==pytest.approx(1)


def test_squeezeout_band_is_only_exempt_at_a_seated_joint():
    model,poses = fixture()
    front = next(s for s in model.surfaces if s.name==LEG.name+'/front')
    cut = model.films[LEG.name+'/miter'].surface
    p = cut.origin+.03*cut.u
    p[2]=THICK
    p -= cut.normal*.001
    model._land(front,p,1e-12)
    assert model.stray_area()>0
    settle(model,poses)
    assert model.cleanliness_mm2==0


def test_bottle_off_table_zeroes_all_credit_and_reset_clears_it():
    model,poses = fixture()
    settle(model,poses)
    model.invalidate('bottle_off_table')
    assert model.score()==0
    model.reset()
    assert not model.failed and model.held_s==0 and model.cleanliness_mm2==0


@pytest.mark.parametrize('failure',['gap','step','tip_offset','airborne','unheld','unparked','drift'])
def test_incomplete_final_states_cannot_get_full_score(failure):
    model,poses = fixture()
    if failure=='gap':poses[LEG.name]=mating_leg(poses[HEAD.name],gap=.001)
    if failure=='step':poses[LEG.name].pos += [0,0,.0015]
    if failure=='tip_offset':poses[LEG.name].pos += [.002,0,0]
    if failure=='airborne':
        for spec in (HEAD,LEG):poses[spec.name].pos += [0,0,.01]
    for tick in range(60):
        if failure=='drift':poses[LEG.name].pos += [.00015,0,0]
        model.joint_step(tick,.05,poses,False,failure!='unparked',held=failure!='unheld')
    assert model.score()<1


def finger_hulls(pose,x):
    result=[]
    for sign,name in ((-1,'tip_left'),(1,'tip_right')):
        centre=np.array([x,sign*(WIDTH/2+.003),THICK/2])
        v=np.array([[a,b,c] for a in (-.008,.008) for b in (-.003,.003) for c in (-.006,.006)])+centre
        v=pose.world(v)
        result.append((name,SimpleNamespace(vertices=v,planes=ConvexHull(v).equations)))
    return result


def test_hand_contact_has_no_prescribed_tcp_or_hand_assignment():
    _,poses=fixture()
    hulls={'left':finger_hulls(poses[LEG.name],.09),'right':finger_hulls(poses[HEAD.name],.22)}
    gaps={'left':.057,'right':.057}
    assert all(x['holding'] for x in hand_status(poses,hulls,gaps).values())
    hulls['left'],hulls['right']=hulls['right'],hulls['left']
    status=hand_status(poses,hulls,gaps)
    assert status['left']['piece']==HEAD.name and all(x['holding'] for x in status.values())
    hulls['right']=finger_hulls(poses[LEG.name],.09)
    hulls['right'][0][1].vertices += [0,0,.03]
    assert not hand_status(poses,hulls,gaps)['right']['holding']


def test_cover_is_thin_and_new_frame_rails_have_two_miters():
    assert COVER_THICKNESS==pytest.approx(.0002)
    assert COVER.pos[2]+COVER.size[2]/2==pytest.approx(TOP)
    for spec in (HEAD,LEG):
        v=spec.mesh()[0][:4]
        assert v[1,0]!=v[2,0] and v[0,0]!=v[3,0]


def test_all_convex_parts_of_each_finger_are_checked():
    _,poses=fixture()
    hulls={'left':finger_hulls(poses[LEG.name],.18),'right':finger_hulls(poses[HEAD.name],.18)}
    # Real YAM fingertips have many convex parts. A distant last part must not
    # overwrite the pad that is actually in contact.
    for side,spec in (('left',LEG),('right',HEAD)):
        distant=finger_hulls(poses[spec.name],.18)
        for _,h in distant:h.vertices += [0,0,.1]
        hulls[side].extend(distant)
    assert all(s['holding'] for s in hand_status(poses,hulls,{'left':.057,'right':.057}).values())


def test_step_and_lateral_mismatch_are_independent_gauge_checks():
    _,poses=fixture()
    cut=next(s for s in HEAD.mesh()[2] if s.target)
    poses[LEG.name].pos += poses[HEAD.name].rot@cut.u*.00095+[0,0,.00045]
    assert joint_status(poses)['precision_seated']


def test_round_outlet_stays_narrow_in_air_and_conserves_local_wetting_on_wood():
    from wood_glue.scene import NOZZLE_DIAMETER, all_initial_poses, surfaces
    model=ObservableAssemblyModel(surfaces())
    poses=all_initial_poses()
    tip=np.array([0.,0.,TOP+.5])
    model._emit(tip,np.eye(3),.05e-6,poses)
    assert len(model.drops)==4
    offsets=np.array([p-tip for p,_,_ in model.drops])
    assert np.linalg.norm(offsets[:,:2],axis=1).max()<=NOZZLE_DIAMETER/2
    assert sum(v for _,_,v in model.drops)==pytest.approx(.05e-6)
    model.reset()
    film=next(iter(model.films.values()))
    point=film.surface.corners.mean(axis=0)
    model._land(film.surface,point,.05e-6)
    wet=film.points[film.volume>0]
    assert len(wet)>1
    assert np.linalg.norm(wet-point,axis=1).max()<=model.settings.wetting_radius
    assert film.volume.sum()==pytest.approx(.05e-6)
    assert model.settings.tip_width==pytest.approx(NOZZLE_DIAMETER)
