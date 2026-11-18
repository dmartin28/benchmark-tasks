"""Outcome tests independent of the demo's phase names and control targets."""
import copy,json,unittest
from pathlib import Path
from yam_arena.constants import READY,FINGER_OPEN
from precrimp_scoring import Scorer

FIXTURES=json.loads((Path(__file__).parent/'tests/geometry_cases.json').read_text())

def held_score(frame):
    scorer=Scorer()
    for k in range(42):
        sample=copy.deepcopy(frame);sample['t']=k*.05;report=scorer.update(sample)
    return report

def clear_hands(frame):
    frame=copy.deepcopy(frame)
    for side in ['left','right']:frame['robot_joint_positions'][side]=[*READY,FINGER_OPEN,FINGER_OPEN]
    return frame

class ScoringTests(unittest.TestCase):
    def test_demonstrated_supported_endpoint_is_full_score(self):
        self.assertEqual(held_score(FIXTURES['final'])['score'],1.)

    def test_displaced_box_is_zero(self):
        frame=copy.deepcopy(FIXTURES['final']);frame['box_position'][1]-=.030
        self.assertEqual(held_score(frame)['score'],0.)

    def test_straight_flaps_are_zero(self):
        frame=clear_hands(FIXTURES['final']);frame['angles_deg']={n:0. for n in frame['angles_deg']}
        self.assertEqual(held_score(frame)['score'],0.)

    def test_one_completed_flap_is_half_credit(self):
        frame=clear_hands(FIXTURES['final'])
        for n in frame['angles_deg']:
            if n.startswith('rear_'):frame['angles_deg'][n]=0.
        self.assertEqual(held_score(frame)['score'],.5)

    def test_wrong_direction_is_not_credited(self):
        frame=clear_hands(FIXTURES['final'])
        for n in frame['angles_deg']:
            if n.startswith('front_'):frame['angles_deg'][n]*=-1.
        self.assertLessEqual(held_score(frame)['score'],.5)

    def test_gripper_still_pressing_flap_is_zero(self):
        self.assertEqual(held_score(FIXTURES['pressing'])['score'],0.)

    def test_extra_full_turn_cannot_masquerade_as_correct_angle(self):
        frame=clear_hands(FIXTURES['final']);frame['angles_deg']['rear_fold_14']+=360.
        self.assertEqual(held_score(frame)['score'],.5)

    def test_short_hold_does_not_score(self):
        scorer=Scorer()
        for k in range(20):
            frame=copy.deepcopy(FIXTURES['final']);frame['t']=k*.05;result=scorer.update(frame)
        self.assertEqual(result['score'],0.)

    def test_moving_box_is_zero_even_while_seated(self):
        scorer=Scorer()
        for k in range(42):
            frame=copy.deepcopy(FIXTURES['final']);frame['t']=k*.05
            frame['box_position'][0]+=.003 if k%2 else -.003
            result=scorer.update(frame)
        self.assertEqual(result['score'],0.)

    def test_score_after_action_limit_is_zero(self):
        scorer=Scorer()
        for k in range(42):
            frame=copy.deepcopy(FIXTURES['final']);frame['t']=239.+k*.05
            result=scorer.update(frame)
        self.assertEqual(result['score'],0.)


if __name__=='__main__':unittest.main()
