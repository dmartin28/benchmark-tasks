from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solder_scene.mechkit import PartState
from solder_scene.runtime import unsafe_geometry,cardboard_supported,workpiece_supported,tip_surface_distance,Trial
from solder_scene.scene import TOP,WORK_TOP,CARDBOARD_START,MAT_START,IRON_START


def state(p):
    p=np.asarray(p,float)
    return PartState(pos=p,quat_xyzw=np.array([0.,0.,0.,1.]),
        lin_vel=np.zeros(3),ang_vel=np.zeros(3),low=p,high=p)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.box=state(CARDBOARD_START)
        self.mat=state(MAT_START)

    def test_iron_in_rest_is_clear_of_wood_and_cardboard(self):
        self.assertFalse(unsafe_geometry(state(IRON_START),self.box,self.mat,[.39,.008,WORK_TOP],False))

    def test_hot_iron_on_bare_bench_is_unsafe(self):
        self.assertTrue(unsafe_geometry(state([.30,.30,TOP]),self.box,self.mat,[.39,.008,WORK_TOP],False))

    def test_box_supported_on_cabinet_but_not_floor(self):
        self.assertTrue(cardboard_supported(self.box))
        self.assertFalse(cardboard_supported(state([.58,-.12,.04])))

    def test_discarded_light_cannot_earn_clearance_credit(self):
        self.assertFalse(workpiece_supported(state([.10,-.02,.01]),self.mat))
        self.assertFalse(workpiece_supported(state([2.,-2.,TOP]),self.mat))
        self.assertTrue(workpiece_supported(state([.10,-.02,TOP+.003]),self.mat))

    def test_tip_side_contact_counts_but_hovering_does_not(self):
        # Physical contact on the cone need not use the extreme endpoint.
        self.assertLess(tip_surface_distance([[.157,.0012,0.]])[0],1e-8)
        self.assertAlmostEqual(tip_surface_distance([[.160,0.,0.]])[0],0.)
        self.assertGreater(tip_surface_distance([[.163,0.,0.]])[0],.0029)
        self.assertGreater(tip_surface_distance([[.157,.0032,0.]])[0],.0018)

    def test_trial_starts_preheated_and_powered_without_actions(self):
        trial=Trial()
        self.assertEqual(trial.heat.tip_c,350.)
        self.assertTrue(trial.switch_on)
        self.assertFalse(trial.heat.bonded)

    def test_relocating_joint_changes_the_packaging_heat_hazard(self):
        iron=state(IRON_START)
        self.assertTrue(unsafe_geometry(iron,self.box,self.mat,[.3902,.008,1.04895],True))
        self.assertFalse(unsafe_geometry(iron,self.box,self.mat,[.098,.0798,.75195],True))


if __name__=='__main__':unittest.main()
