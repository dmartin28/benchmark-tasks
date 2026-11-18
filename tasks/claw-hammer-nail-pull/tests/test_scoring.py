from dataclasses import replace
import unittest
import numpy as np
from nail_scene.scoring import Outcome,score,StabilityWindow,fits_tray
from nail_scene.scene import NAIL_VERTICES,TRAY_SIZE
from nail_scene.material import NAIL_HEAD_DIAMETER
from scipy.spatial.transform import Rotation


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.good=Outcome(extraction_m=.021,claw_work_m=.010,nail_free=True,
            nail_in_tray=True,board_on_table=True,hammer_on_table=True,
            tray_on_table=True,hands_clear=True,stable_s=1.)

    def test_initial_and_direct_hand_pull_score_zero(self):
        self.assertEqual(score(Outcome()),0)
        self.assertEqual(score(replace(self.good,claw_work_m=0)),0)

    def test_endpoint_tiers(self):
        self.assertEqual(score(self.good),1)
        self.assertEqual(score(replace(self.good,nail_in_tray=False)),.75)
        partial=replace(self.good,nail_free=False,nail_in_tray=False)
        self.assertEqual(score(replace(partial,extraction_m=.010)),.5)
        self.assertEqual(score(replace(partial,extraction_m=.005)),.25)
        self.assertEqual(score(replace(partial,extraction_m=.004)),0)

    def test_held_or_unstable_endpoints_do_not_score(self):
        for kwargs in [{'hands_clear':False},{'stable_s':.95},
                       {'board_on_table':False},{'hammer_on_table':False},
                       {'tray_on_table':False},{'invalid':True}]:
            self.assertEqual(score(replace(self.good,**kwargs)),0)

    def test_flat_nail_entirely_inside_tray(self):
        R=Rotation.from_euler('y',90,degrees=True).as_matrix()
        vertices=NAIL_VERTICES@R.T+[0,0,.0015+NAIL_HEAD_DIAMETER/2]
        self.assertTrue(fits_tray(vertices,TRAY_SIZE))

    def test_tilted_head_above_rim_does_not_count(self):
        R=Rotation.from_euler('y',60,degrees=True).as_matrix()
        vertices=NAIL_VERTICES@R.T+[0,0,.0015]
        # Both shaft endpoints are below 22 mm, but the head protrudes.
        self.assertFalse(fits_tray(vertices,TRAY_SIZE))

    def test_slow_drift_is_not_a_stable_second(self):
        window=StabilityWindow()
        for i in range(30):
            stable=window.update({'nail':(np.array([i*.0005,0.,0.]),
                np.array([0.,0.,0.,1.]))},.05,True)
        self.assertEqual(stable,0)

    def test_clear_stationary_second_and_hand_return(self):
        window=StabilityWindow()
        poses={'nail':(np.zeros(3),np.array([0.,0.,0.,1.]))}
        for _ in range(25):stable=window.update(poses,.05,True)
        self.assertGreaterEqual(stable,.999999)
        self.assertEqual(window.update(poses,.05,False),0)
