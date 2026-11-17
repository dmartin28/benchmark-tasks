from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solder_scene.thermal import ThermalState


class ThermalTests(unittest.TestCase):
    def test_warming_alone_does_not_solder(self):
        s=ThermalState()
        for _ in range(600):s.advance(.05,True,False)
        self.assertGreater(s.tip_c,330.)
        self.assertEqual(s.joint_c,22.)
        self.assertFalse(s.bonded)

    def test_cold_contact_does_not_solder(self):
        s=ThermalState()
        for _ in range(80):s.advance(.05,False,True)
        self.assertFalse(s.wetted)

    def test_contact_reflow_then_cooling_required(self):
        s=ThermalState(tip_c=340.)
        for _ in range(30):s.advance(.05,True,True)
        self.assertTrue(s.wetted)
        self.assertFalse(s.bonded)
        for _ in range(30):s.advance(.05,True,False)
        self.assertTrue(s.bonded)

    def test_short_separate_taps_do_not_accumulate(self):
        s=ThermalState(tip_c=340.)
        for _ in range(8):
            for _ in range(5):s.advance(.05,True,True)
            for _ in range(30):s.advance(.05,True,False)
        self.assertFalse(s.wetted)

    def test_unsafe_heat_latches_even_after_power_off(self):
        s=ThermalState(tip_c=340.)
        s.advance(.05,False,False,True)
        for _ in range(2000):s.advance(.05,False,False,False)
        self.assertTrue(s.unsafe)
        self.assertLess(s.tip_c,30.)

    def test_invalid_timestep_rejected(self):
        for dt in (-.05,0.,float('nan')):
            with self.assertRaises(ValueError):ThermalState().advance(dt,True,True)


if __name__=='__main__':unittest.main()
