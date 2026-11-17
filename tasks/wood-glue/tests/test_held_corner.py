"""The held endpoint requires both actual grasp poses and leaves release scoring distinct."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wood_glue.assembly import AssemblyModel,mating_leg
from wood_glue.controller import LEG_GRASP_LOCAL
from wood_glue.milestone import held_corner_status
from wood_glue.scene import HEAD,LEG,surfaces,all_initial_poses
from yam_arena.demo import grip_rotation


def held_state():
    model=AssemblyModel(surfaces(),require_bond=True)
    poses=all_initial_poses()
    poses[LEG.name]=mating_leg(poses[HEAD.name],gap=.0001)
    film=model.films[LEG.name+'/miter']
    film.volume[4:-4,6:10]=film.area*.00017
    tcps={side:poses[spec.name].world(LEG_GRASP_LOCAL) for side,spec in (('left',LEG),('right',HEAD))}
    rotations={side:grip_rotation(np.arctan2(poses[spec.name].rot[1,0],poses[spec.name].rot[0,0]))
               for side,spec in (('left',LEG),('right',HEAD))}
    return model,poses,tcps,rotations,{'left':.057,'right':.057}


def test_both_hands_holding_qualifies():
    m,p,t,r,g=held_state()
    assert held_corner_status(p,t,r,g,m,True,True)['qualified']


@pytest.mark.parametrize('failure',['left_open','right_open','left_away','right_away','misaligned','unparked','dry','unbonded'])
def test_incomplete_held_endpoint_does_not_qualify(failure):
    m,p,t,r,g=held_state()
    if failure.endswith('_open'):g[failure.split('_')[0]]=.09
    if failure.endswith('_away'):t[failure.split('_')[0]] += [.03,0.,0.]
    if failure=='misaligned':r['right']=np.eye(3)
    if failure=='dry':m.reset()
    status=held_corner_status(p,t,r,g,m,failure!='unparked',failure!='unbonded')
    assert not status['qualified']


def test_held_milestone_does_not_claim_released_task_success():
    m,p,t,r,g=held_state()
    for tick in range(40):m.joint_step(tick,.05,p,False,True,bonded=True)
    assert held_corner_status(p,t,r,g,m,True,True)['qualified']
    assert m.score()==.75 and m.release_s==0.


def submission_state():
    _,poses,tcps,rotations,gaps=held_state()
    model=AssemblyModel(surfaces(),require_bond=False,completion='held')
    film=model.films[LEG.name+'/miter']
    film.volume[4:-4,6:10]=film.area*.00017
    return model,poses,tcps,rotations,gaps


def test_submission_scores_a_stable_hold_without_curing_or_release():
    m,p,t,r,g=submission_state()
    assert held_corner_status(p,t,r,g,m,True,False)['qualified']
    for tick in range(20):m.joint_step(tick,.05,p,False,True,bonded=False,held=True)
    assert m.score()==1. and m.held_s>=1.-1e-9 and m.release_s==0.
    m.joint_step(20,.05,p,False,True,held=False)
    assert m.score()==.75


def test_submission_allows_different_grasp_positions_and_hand_assignment():
    m,p,t,r,g=submission_state()
    for side,spec in (('left',LEG),('right',HEAD)):
        t[side]=p[spec.name].world([.24,0.,.0155])
    assert held_corner_status(p,t,r,g,m,True,False)['qualified']
    t['left'],t['right']=t['right'],t['left']
    r['left'],r['right']=r['right'],r['left']
    status=held_corner_status(p,t,r,g,m,True,False)
    assert status['qualified'] and status['hands']['left']['piece']==HEAD.name


@pytest.mark.parametrize('error',['airborne','gap','lateral','drifting'])
def test_submission_rejects_loose_or_unstable_held_corners(error):
    m,p,t,r,g=submission_state()
    if error=='airborne':
        for spec in (HEAD,LEG):p[spec.name].pos += [0.,0.,.02]
    if error=='gap':p[LEG.name]=mating_leg(p[HEAD.name],gap=.0006)
    if error=='lateral':p[LEG.name].pos += [0.,.0009,0.]
    for tick in range(30):
        if error=='drifting':
            for spec in (HEAD,LEG):p[spec.name].pos += [.0001,0.,0.]
        m.joint_step(tick,.05,p,False,True,held=True)
    assert m.score()<1.
