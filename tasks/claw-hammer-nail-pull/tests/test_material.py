import unittest
from nail_scene.material import Withdrawal,withdrawal_force,INITIAL_EMBEDMENT,ELASTIC_STIFFNESS


class MaterialTests(unittest.TestCase):
    def test_handbook_units_and_reference_force(self):
        self.assertAlmostEqual(withdrawal_force(.020),157.665,delta=.1)
        self.assertAlmostEqual(withdrawal_force(.010)*2,withdrawal_force(.020))

    def test_no_force_after_point_clears(self):
        self.assertEqual(withdrawal_force(0),0)
        self.assertEqual(withdrawal_force(-.01),0)

    def test_sub_yield_loading_does_not_ratchet(self):
        model=Withdrawal()
        for _ in range(100):
            rest,force=model.update(.000050)
            self.assertEqual(rest,0)
            self.assertFalse(model.released)

    def test_plastic_withdrawal_persists_after_elastic_unloading(self):
        model=Withdrawal();rest,force=model.update(.006)
        self.assertAlmostEqual(rest,.006-force/ELASTIC_STIFFNESS)
        self.assertGreater(rest,.005)
        self.assertEqual(model.update(rest)[0],rest)

    def test_constraint_releases_only_after_full_extraction(self):
        model=Withdrawal();model.update(INITIAL_EMBEDMENT-.001)
        self.assertFalse(model.released)
        model.update(INITIAL_EMBEDMENT+.001)
        self.assertTrue(model.released)

    def test_collected_nail_does_not_regain_wood_resistance(self):
        model=Withdrawal()
        model.update(INITIAL_EMBEDMENT+.001)
        # A removed nail can end below the board's top, inside the tray.
        _,force=model.update(-.020)
        self.assertTrue(model.released)
        self.assertEqual(force,0.)
