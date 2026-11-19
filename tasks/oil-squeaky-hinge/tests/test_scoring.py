"""Offline tests for layout.py, scoring.py and task.py: the stream's geometry, the gaps, the door's blocking, the
spray recorder, every score tier and the task file's declarations. No simulator.

    PYTHONPATH=<yam-isaac-lab-arena> python3 -m unittest discover -s tests -v

Every number here is a design value; nothing was measured on the simulator."""

import math
import pathlib
import sys
import unittest

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

try:
    import layout as L  # noqa: E402
    import scoring as S  # noqa: E402
except ModuleNotFoundError as e:  # pragma: no cover
    raise unittest.SkipTest(str(e))

MM = L.MM
DT = 0.05
I3 = np.eye(3)
AX, AY = L.axis_world()


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def can_for_tip(tip_world, rot=I3, valve=0.0):
    """The can origin that puts the straw's tip at ``tip_world`` with the can turned ``rot``."""
    local = np.array(L.STRAW_TIP) - np.array([0, 0, valve])
    return np.asarray(tip_world, float) - rot @ local


def tip_at_gap(k, standoff=12 * MM, dz=0.0):
    return np.array([AX - L.BARREL_R - standoff, AY, S.seam_heights_world()[k] + dz])


class Geometry(unittest.TestCase):
    def test_four_gaps_12_7mm_apart_on_the_top_hinge(self):
        z = L.seams()
        self.assertEqual(len(z), 4)
        for a, b in zip(z, z[1:]):
            self.assertAlmostEqual(b - a, 12.7 * MM, places=6)
        self.assertTrue(L.DOOR_Z0 < z[0] < z[-1] < L.DOOR_Z1)
        self.assertGreater(z[0], L.BOTTOM_HINGE_Z + L.HINGE_LEN / 2)

    def test_knuckles_alternate_frame_and_door(self):
        ks = L.knuckles(L.TOP_HINGE_Z)
        self.assertEqual([k[2] for k in ks], [False, True, False, True, False])
        for (lo, hi, _), s in zip(ks, L.seams() + [None]):
            if s is not None:
                self.assertLess(hi, s)

    def test_straw_tip_at_rest(self):
        tip, d = S.straw_jet(I3, (0, 0, 0), 0.0)
        np.testing.assert_allclose(tip, L.STRAW_TIP)
        np.testing.assert_allclose(d, (1, 0, 0))
        tip2, _ = S.straw_jet(I3, (0, 0, 0), 2 * MM)
        self.assertAlmostEqual(tip2[2], L.STRAW_TIP[2] - 2 * MM)

    def test_straw_turns_with_the_can(self):
        tip, d = S.straw_jet(rot_z(math.pi / 2), (0.1, 0.2, 0.3), 0.0)
        np.testing.assert_allclose(d, (0, 1, 0), atol=1e-12)
        self.assertAlmostEqual(tip[1], 0.2 + L.STRAW_TIP[0])


class Stream(unittest.TestCase):
    def test_square_shot_lands_in_each_gap(self):
        for k in range(4):
            hit = S.barrel_hit(tip_at_gap(k), (1, 0, 0))
            self.assertIsNotNone(hit)
            self.assertAlmostEqual(hit[0], AX - L.BARREL_R, places=6)
            self.assertEqual(S.seam_at(hit[2]), k)

    def test_within_3mm_counts_beyond_does_not(self):
        hit = S.barrel_hit(tip_at_gap(1, dz=2.9 * MM), (1, 0, 0))
        self.assertEqual(S.seam_at(hit[2]), 1)
        hit = S.barrel_hit(tip_at_gap(1, dz=3.5 * MM), (1, 0, 0))
        self.assertIsNotNone(hit)
        self.assertIsNone(S.seam_at(hit[2]))

    def test_mid_knuckle_is_no_gap(self):
        hit = S.barrel_hit(tip_at_gap(1, dz=6.35 * MM), (1, 0, 0))
        self.assertIsNone(S.seam_at(hit[2]))

    def test_beside_the_barrel_misses(self):
        p = tip_at_gap(0)
        p[1] += L.BARREL_R + 1 * MM
        self.assertIsNone(S.barrel_hit(p, (1, 0, 0)))

    def test_away_from_the_barrel_misses(self):
        self.assertIsNone(S.barrel_hit(tip_at_gap(0), (-1, 0, 0)))

    def test_bottom_hinge_and_door_face_get_nothing(self):
        p = np.array([AX - 0.03, AY, L.TOP + L.BOTTOM_HINGE_Z])
        self.assertIsNone(S.barrel_hit(p, (1, 0, 0)))
        p = np.array([AX - 0.03, AY, L.TOP + L.TOP_HINGE_Z + 0.05])
        self.assertIsNone(S.barrel_hit(p, (1, 0, 0)))

    def test_beyond_reach(self):
        self.assertIsNotNone(S.barrel_hit(tip_at_gap(2, standoff=0.45), (1, 0, 0)))
        self.assertIsNone(S.barrel_hit(tip_at_gap(2, standoff=0.55), (1, 0, 0)))

    def test_slanted_shot_from_the_side(self):
        z = S.seam_heights_world()[3]
        p = np.array([AX - 0.03, AY - 0.03, z])
        d = np.array([AX, AY, z]) - p
        hit = S.barrel_hit(p, d / np.linalg.norm(d))
        self.assertEqual(S.seam_at(hit[2]), 3)

    def test_tip_touching_the_barrel(self):
        p = np.array([AX - L.BARREL_R + 1 * MM, AY, S.seam_heights_world()[2]])
        hit = S.barrel_hit(p, (0, 0, -1))
        self.assertEqual(S.seam_at(hit[2]), 2)

    def test_open_door_blocks_a_shot_along_its_face(self):
        z = S.seam_heights_world()[0]
        p = np.array([AX - 2 * MM, AY + 0.06, z])  # from the robot's left, across the front of the barrel
        self.assertIsNotNone(S.barrel_hit(p, (0, -1, 0), door_angle=0.0))
        self.assertIsNone(S.barrel_hit(p, (0, -1, 0), door_angle=math.radians(90)))

    def test_open_door_does_not_block_a_square_shot(self):
        for a in (0.0, 30.0, 70.0):
            self.assertIsNotNone(S.barrel_hit(tip_at_gap(1), (1, 0, 0), door_angle=math.radians(a)))


def run(spray, steps, valve=0.0, tip=None, rot=I3, door=0.0):
    origin = can_for_tip(tip if tip is not None else tip_at_gap(0, standoff=0.3), rot, valve)
    for _ in range(steps):
        spray.step(valve, rot, origin, door, DT)


def oil_all(spray, steps_each=6):
    for k in range(4):
        run(spray, steps_each, valve=2 * MM, tip=tip_at_gap(k))


def swing(spray, n=1, top=70.0, bottom=0.0):
    for _ in range(n):
        for a in np.linspace(bottom, top, 10).tolist() + np.linspace(top, bottom, 10).tolist():
            run(spray, 1, door=math.radians(a))


class Recorder(unittest.TestCase):
    def test_a_short_burst_oils_a_gap(self):
        s = S.Spray()
        run(s, 1, valve=2 * MM, tip=tip_at_gap(1))  # 0.05 s: one frame passing over it, not yet
        self.assertEqual(s.oiled, [False, False, False, False])
        run(s, 1, valve=2 * MM, tip=tip_at_gap(1))  # 0.1 s: a burst
        self.assertEqual(s.oiled, [False, True, False, False])
        self.assertAlmostEqual(s.spray_time, 0.1)

    def test_valve_below_threshold_sprays_nothing(self):
        s = S.Spray()
        run(s, 20, valve=1.4 * MM, tip=tip_at_gap(1))
        self.assertEqual(s.n_oiled, 0)
        self.assertEqual(s.spray_time, 0.0)

    def test_spray_elsewhere_counts_toward_the_limit(self):
        s = S.Spray()
        run(s, 98, valve=2 * MM, tip=tip_at_gap(1, dz=6 * MM))  # 4.9 s: 3.97 g
        self.assertEqual(s.n_oiled, 0)
        self.assertAlmostEqual(s.sprayed_g, 4.9 * 0.81)
        self.assertFalse(s.flooded)
        run(s, 1, valve=2 * MM, tip=tip_at_gap(1, dz=6 * MM))  # 4.95 s: 4.01 g
        self.assertTrue(s.flooded)

    def test_swings_count_only_after_all_four_gaps(self):
        s = S.Spray()
        swing(s, 2)
        self.assertEqual(s.swings, 0)
        oil_all(s)
        self.assertEqual(s.n_oiled, 4)
        swing(s, 3)
        self.assertEqual(s.swings, 3)

    def test_a_swing_needs_45_open_and_10_shut(self):
        s = S.Spray()
        oil_all(s)
        swing(s, 1, top=43.0)
        self.assertEqual(s.swings, 0)
        swing(s, 1, top=70.0, bottom=15.0)
        self.assertEqual(s.swings, 0)
        run(s, 1, door=math.radians(5))
        self.assertEqual(s.swings, 1)

    def test_reset(self):
        s = S.Spray()
        oil_all(s)
        swing(s, 3)
        s.reset()
        self.assertEqual((s.n_oiled, s.swings, s.spray_time), (0, 0, 0.0))


CAN_HOME = np.array([L.CAN_START[0], L.CAN_START[1], L.TOP])


class Tiers(unittest.TestCase):
    def done(self, swings=3):
        s = S.Spray()
        oil_all(s)
        swing(s, swings)
        return s

    def test_full(self):
        r = S.judge(self.done(), 0.0, I3, CAN_HOME, True, True)
        self.assertEqual(r.score, 1.0)

    def test_one_or_two_swings(self):
        self.assertEqual(S.judge(self.done(swings=1), 0.0, I3, CAN_HOME, True, True).score, 0.75)
        self.assertEqual(S.judge(self.done(swings=2), 0.0, I3, CAN_HOME, True, True).score, 0.75)

    def test_door_left_open_or_can_held_or_moving(self):
        s = self.done()
        self.assertEqual(S.judge(s, math.radians(20), I3, CAN_HOME, True, True).score, 0.75)
        self.assertEqual(S.judge(s, 0.0, I3, CAN_HOME, False, True).score, 0.75)
        self.assertEqual(S.judge(s, 0.0, I3, CAN_HOME, True, False).score, 0.75)

    def test_can_lying_down_or_against_the_cabinet(self):
        s = self.done()
        self.assertEqual(S.judge(s, 0.0, rot_y(math.radians(90)), CAN_HOME + [0, 0, L.CAN_R], True, True).score,
                         0.75)
        against = np.array([L.CAB_X - L.CAN_R - 2 * MM, 0.1, L.TOP])
        self.assertEqual(S.judge(s, 0.0, I3, against, True, True).score, 0.75)
        on_top = np.array([L.CAB_X + 0.1, 0.1, L.TOP + L.CAB_H])
        self.assertEqual(S.judge(s, 0.0, I3, on_top, True, True).score, 0.75)

    def test_oiled_not_worked(self):
        self.assertEqual(S.judge(self.done(swings=0), 0.0, I3, CAN_HOME, True, True).score, 0.5)

    def test_two_or_three_gaps(self):
        s = S.Spray()
        run(s, 6, valve=2 * MM, tip=tip_at_gap(0))
        self.assertEqual(S.judge(s, 0.0, I3, CAN_HOME, True, True).score, 0.0)
        run(s, 6, valve=2 * MM, tip=tip_at_gap(3))
        self.assertEqual(S.judge(s, 0.0, I3, CAN_HOME, True, True).score, 0.25)
        run(s, 6, valve=2 * MM, tip=tip_at_gap(2))
        swing(s, 3)
        self.assertEqual(S.judge(s, 0.0, I3, CAN_HOME, True, True).score, 0.25)

    def test_flooding_scores_zero_whatever_else(self):
        s = self.done()
        run(s, 80, valve=2 * MM, tip=tip_at_gap(0))
        self.assertTrue(s.flooded)
        self.assertEqual(S.judge(s, 0.0, I3, CAN_HOME, True, True).score, 0.0)

    def test_nothing_done(self):
        self.assertEqual(S.judge(S.Spray(), 0.0, I3, CAN_HOME, True, True).score, 0.0)

    def test_swinging_without_oil(self):
        s = S.Spray()
        swing(s, 5)
        self.assertEqual(S.judge(s, 0.0, I3, CAN_HOME, True, True).score, 0.0)


class TaskFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import task as T  # noqa: F401
        except ModuleNotFoundError as e:
            raise unittest.SkipTest(f"yam_arena is not importable ({e}): put yam-isaac-lab-arena on PYTHONPATH")
        cls.T = T

    def test_objects(self):
        names = [o.name for o in self.T.OBJECTS]
        self.assertEqual(names, ["cabinet", "can"])
        self.assertIsNone(self.T.CABINET.mass)  # fixed
        self.assertAlmostEqual(self.T.CAN.mass, L.CAN_MASS)

    def test_can_fits_the_gripper_and_starts_clear(self):
        self.assertLess(2 * L.CAN_R, 0.094)
        self.assertTrue(S.clear_of_cabinet((L.CAN_START[0], L.CAN_START[1], L.TOP)))

    def test_start_scores_zero(self):
        cab, can = self.T.CABINET.rest_state(), self.T.CAN.rest_state()
        self.T.SPRAY.reset()
        self.assertEqual(self.T.score({"cabinet": cab, "can": can}), 0.0)

    def test_instruction(self):
        self.assertIn("WD-40", self.T.INSTRUCTION)
        self.assertIn("three times", self.T.INSTRUCTION)

    def test_demo_reads_no_scorer_state(self):
        src = (HERE / "demo.py").read_text()
        for word in ("SPRAY.", "T.SPRAY", ".dose", ".oiled", ".n_oiled", ".swings", "judge(", "Spray("):
            self.assertFalse(word in src, f"demo.py mentions {word!r}")


if __name__ == "__main__":
    unittest.main()
