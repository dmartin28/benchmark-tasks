"""Check overload timing, including loads crossing policy-step boundaries."""
import unittest
import numpy as np
from install_physics import ContactForceMonitor


class ForceMonitorTests(unittest.TestCase):
    def test_isolated_millisecond_spikes_do_not_stop(self):
        m=ContactForceMonitor();samples=np.zeros((100,4))
        samples[7,1]=117.;samples[66,1]=42.
        m.observe(samples,.001)
        self.assertFalse(m.triggered.any())
        self.assertAlmostEqual(m.longest_s[1],.001)

    def test_sustained_load_crossing_control_boundary_stops(self):
        m=ContactForceMonitor();first=np.zeros((50,4));first[-9:,2]=45.
        m.observe(first,.001);self.assertFalse(m.triggered.any())
        second=np.zeros((11,4));second[:,2]=45.
        m.observe(second,.001)
        self.assertTrue(m.triggered[2]);self.assertAlmostEqual(m.longest_s[2],.020)

    def test_below_threshold_resets_duration(self):
        m=ContactForceMonitor();samples=np.zeros((39,4));samples[:,0]=50.;samples[19,0]=40.
        m.observe(samples,.001)
        self.assertFalse(m.triggered.any());self.assertAlmostEqual(m.longest_s[0],.019)

    def test_fingers_do_not_share_overload_time(self):
        m=ContactForceMonitor();samples=np.zeros((30,4));samples[:15,0]=80.;samples[15:,1]=80.
        m.observe(samples,.001);self.assertFalse(m.triggered.any())

    def test_zero_duration_preserves_instantaneous_stop(self):
        m=ContactForceMonitor(duration_s=0.);m.observe([[0.,0.,0.,0.]],.001)
        self.assertFalse(m.triggered.any())
        m.observe([[0.,41.,0.,0.]],.001);self.assertTrue(m.triggered[1])


if __name__=='__main__':unittest.main()
