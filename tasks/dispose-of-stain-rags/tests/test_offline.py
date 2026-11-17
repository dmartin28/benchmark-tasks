"""Offline checks: no GPU, no Isaac Sim.  python -m unittest discover -s tasks/dispose-of-stain-rags/tests"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import can as C  # noqa: E402
import layout as L  # noqa: E402
import scoring as S  # noqa: E402
from gather_pinch import TIP, GatherPinch  # noqa: E402
from rag_mesh import crumple, edge_strain, square  # noqa: E402

FLAT = square(side=L.RAG_SIDE, spacing=L.RAG_SPACING)
MESH = crumple(FLAT, seed=L.RAG_SEED)
NV = len(MESH.rest)


def rag_on_board():
    return MESH.rest + (*L.RAG_XY, L.BOARD_TOP + 0.0015)


def rag_in_can_bottom(dy=0.0):
    """The rag bunched on the can's floor: squeezed into a 120 mm patch, 10-60 mm up."""
    p = FLAT.rest * 0.48
    p[:, 2] = 0.010 + 0.05 * np.abs(FLAT.rest[:, 0]) / 0.125
    return p + (L.CAN_XY[0], L.CAN_XY[1] + dy, L.TOP)


def rag_in_bin():
    return FLAT.rest * 0.4 + (L.BIN_XY[0], L.BIN_XY[1], L.TOP + 0.03)


def rag_held_over_can():
    """Hanging from a gripper above the open mouth: the bottom 30 mm below the rim, the rest above."""
    return FLAT.rest[:, [0, 2, 1]] * 0.4 + (L.CAN_XY[0], L.CAN_XY[1], S.RIM_Z + 0.02)


def footprint_clear(a, ra, b, rb, margin=0.015):
    return math.dist(a, b) > ra + rb + margin


class TestMeshAndLayout(unittest.TestCase):
    def test_rag_mesh(self):
        info = MESH.audit()
        self.assertEqual(info["vertices"], 676)
        self.assertEqual(MESH.uv.shape, (676, 2))
        self.assertGreater(info["surface_area_m2"], 0.035)              # about a 200 mm rag of cloth
        self.assertLess(info["max_edge_m"], 0.03)

    def test_rag_starts_crumpled_on_the_board(self):
        p = rag_on_board()
        self.assertTrue(np.all(p[:, 0] > L.BOARD_X0) and np.all(p[:, 0] < L.BOARD_X0 + L.BOARD_L))
        self.assertTrue(np.all(abs(p[:, 1] - L.BOARD_Y) < L.BOARD_W / 2))
        z = p[:, 2] - L.BOARD_TOP
        self.assertGreater(z.min(), 0.0)
        self.assertGreater(z.max(), 0.05)                                # folds stand 5-7 cm high
        self.assertGreater(np.mean(z > 0.015), 0.25)

    def test_footprints_do_not_overlap(self):
        board_half = (L.BOARD_L / 2, L.BOARD_W / 2)
        bx, by = L.BOARD_CENTRE[:2]
        for xy, r in ((L.CAN_XY, C.LID_R), (L.BIN_XY, L.BIN_R)):
            dx = max(abs(xy[0] - bx) - board_half[0], 0.0)
            dy = max(abs(xy[1] - by) - board_half[1], 0.0)
            self.assertGreater(math.hypot(dx, dy), r)                    # clear of the board
        self.assertTrue(footprint_clear(L.CAN_XY, C.LID_R, L.BIN_XY, L.BIN_R, 0.2))
        self.assertAlmostEqual(L.CAN_XY[0], L.BIN_XY[0])                 # mirror images either side
        self.assertAlmostEqual(L.CAN_XY[1], -L.BIN_XY[1])

    def test_stain_props_clear_of_everything(self):
        props = [(L.STAIN_XY, L.STAIN_R), (L.STAIN_LID_XY, L.STAIN_LID_R)]
        for xy, r in props:
            self.assertTrue(footprint_clear(xy, r, L.BIN_XY, L.BIN_R))
            self.assertTrue(footprint_clear(xy, r, L.CAN_XY, C.LID_R))
            self.assertGreater(xy[0] - r, L.BOARD_X0 + L.BOARD_L)          # behind the board's far end
            self.assertLess(xy[0] + r, 0.91)                              # on the bench (far edge 0.91)
        self.assertTrue(footprint_clear(L.STAIN_XY, L.STAIN_R, L.STAIN_LID_XY, L.STAIN_LID_R, 0.01))

    def test_can_matches_maker_dimensions(self):
        self.assertAlmostEqual(2 * C.LID_R, 0.244, places=3)                     # 244 mm outer diameter
        self.assertAlmostEqual(C.BODY_H + C.LID_T, 0.228, delta=0.006)           # 232 mm overall
        self.assertEqual(C.OPEN_LIMIT_DEG, 60.0)


class TestScoring(unittest.TestCase):
    still = [np.zeros((NV, 3))]

    def test_full_score(self):
        self.assertEqual(S.score_from([rag_in_can_bottom()], 0.0), 1.0)
        self.assertTrue(S.success_from([rag_in_can_bottom()], self.still, 0.0))

    def test_lid_left_open(self):
        rags = [rag_in_can_bottom()]
        self.assertEqual(S.score_from(rags, 20.0), 0.5)
        self.assertEqual(S.score_from(rags, 3.5), 0.5)
        self.assertEqual(S.score_from(rags, 3.0), 1.0)

    def test_rag_in_bin_scores_zero(self):
        self.assertEqual(S.score_from([rag_in_bin()], 0.0), 0.0)
        corner = rag_in_can_bottom().copy()
        corner[0] = (L.BIN_XY[0], L.BIN_XY[1], L.TOP + 0.20)             # one point dangling in the bin
        self.assertEqual(S.score_from([corner], 0.0), 0.0)

    def test_start_state_scores_zero(self):
        self.assertEqual(S.score_from([rag_on_board()], 0.0), 0.0)
        self.assertEqual(S.score_from([rag_on_board()], 40.0), 0.0)

    def test_rag_held_over_can_does_not_count(self):
        held = rag_held_over_can()
        self.assertFalse(S.rag_in_can(held))
        self.assertEqual(S.score_from([held], 58.0), 0.0)
        self.assertEqual(S.score_from([held], 0.0), 0.0)

    def test_rag_against_inner_wall_counts_but_over_the_wall_does_not(self):
        r = rag_in_can_bottom()
        r[0, :2] = (L.CAN_XY[0] - (C.BODY_R - C.WALL) - 0.0005, L.CAN_XY[1])   # touching the inner wall
        self.assertTrue(S.rag_in_can(r))
        r[1, :2] = (L.CAN_XY[0] - C.BODY_R - 0.0005, L.CAN_XY[1])              # poked through the thin wall
        self.assertTrue(S.rag_in_can(r))
        r[1, :2] = (L.CAN_XY[0] - C.BODY_R - 0.010, L.CAN_XY[1])               # hanging outside the wall
        self.assertFalse(S.rag_in_can(r))

    def test_rag_draped_over_the_wall_does_not_count(self):
        draped = rag_in_can_bottom()
        draped[:30, 0] = L.CAN_XY[0] - C.BODY_R - 0.001                         # outside, against the wall
        draped[30:40, 2] = S.RIM_Z + 0.0005                                      # over the rim
        self.assertFalse(S.rag_in_can(draped))

    def test_rag_caught_on_rim_does_not_count(self):
        caught = rag_in_can_bottom()
        caught[:20, 2] = S.RIM_Z + 0.005
        self.assertFalse(S.rag_in_can(caught))

    def test_a_rag_still_held_never_counts(self):
        rags = [rag_in_can_bottom()]
        self.assertEqual(S.score_from(rags, 0.0, held=[True]), 0.0)
        self.assertEqual(S.score_from(rags, 40.0, held=[True]), 0.0)
        self.assertFalse(S.success_from(rags, self.still, 0.0, held=[True]))
        self.assertTrue(S.success_from(rags, self.still, 0.0, held=[False]))

    def test_moving_rag_is_not_success(self):
        rags = [rag_in_can_bottom()]
        self.assertFalse(S.success_from(rags, [np.tile((0.0, 0.0, -0.3), (NV, 1))], 0.0))
        jitter = np.zeros((NV, 3))
        jitter[5] = (0, 0, -0.3)                      # one node's numerical jitter is not motion
        self.assertTrue(S.success_from(rags, [jitter], 0.0))


class TestGripRule(unittest.TestCase):
    R, tcp = np.eye(3), np.zeros(3)
    _xy = np.meshgrid(np.arange(-0.0995, 0.1, 0.01), np.arange(-0.1, 0.1001, 0.01))
    grid = np.column_stack([_xy[0].ravel(), _xy[1].ravel(), np.zeros(_xy[0].size)])   # offset 10 mm grid

    def close(self, pts_by_gap, gaps):
        g = GatherPinch()
        for gap in gaps:
            g.update(pts_by_gap(gap), self.tcp, self.R, gap, 1 / 240)
        return g

    def test_gather_needs_pressed_tips(self):
        pressed = self.grid + (0, 0, TIP)
        g = self.close(lambda gap: pressed, np.linspace(0.061, 0.002, 80))
        self.assertTrue(len(g.indices) >= 3)
        self.assertEqual(g.events[-1]["how"], "gather")

    def test_no_contact_no_grip(self):
        far = self.grid + (0, 0, TIP + 0.02)
        g = self.close(lambda gap: far, np.linspace(0.061, 0.002, 80))
        self.assertEqual(len(g.indices), 0)

    def test_lifting_off_mid_close_disarms(self):
        pressed, far = self.grid + (0, 0, TIP), self.grid + (0, 0, TIP + 0.02)
        g = self.close(lambda gap: pressed if gap > 0.03 else far, np.linspace(0.061, 0.002, 80))
        self.assertEqual(len(g.indices), 0)
        self.assertIn("gather_disarmed", [e["event"] for e in g.events])

    def test_starting_closed_cannot_gather(self):
        pressed = self.grid + (0, 0, TIP)
        g = self.close(lambda gap: pressed, np.linspace(0.020, 0.002, 60))   # never opened to 25 mm
        self.assertEqual(len(g.indices), 0)

    def test_opening_releases(self):
        pressed = self.grid + (0, 0, TIP)
        g = self.close(lambda gap: pressed, list(np.linspace(0.061, 0.002, 80)) + [0.002, 0.009])
        self.assertEqual(len(g.indices), 0)
        self.assertEqual(g.events[-1]["event"], "opened")


class TestLid(unittest.TestCase):
    def test_lid_angle_from_xyzw(self):
        for deg in (0.0, 12.5, 40.0, 60.0):
            a = math.radians(deg)
            self.assertAlmostEqual(C.lid_angle_deg((0.0, math.sin(a / 2), 0.0, math.cos(a / 2))), deg, places=6)


class TestTaskFileAndDemo(unittest.TestCase):
    def test_task_file_loads(self):
        from yam_arena import task_file
        t = task_file.load(HERE / "task.py")
        self.assertEqual(t.scene, "station")
        self.assertEqual(t.free_objects, ())   # nothing gets the loader's reset jitter
        self.assertIsNotNone(t.success)
        self.assertIsNotNone(t.score)
        self.assertEqual([o.name for o in t.objects if o.name.startswith("rag")], ["rag"])

    def test_grasp_is_on_the_highest_fold(self):
        import demo as D
        g, crest, across = D.grasp_from_nodes(rag_on_board())
        self.assertAlmostEqual(np.linalg.norm(across), 1.0)
        self.assertGreater(crest - L.BOARD_TOP, 0.05)
        p = rag_on_board()
        d = np.hypot(p[:, 0] - g[0], p[:, 1] - g[1])
        self.assertLess(d.min(), 0.012)                                    # a cloth point under the closing point
        # The cloth falls away on both sides of the fold, across it.
        for s in (-1, 1):
            q = g + s * 0.035 * across
            i = np.argmin(np.hypot(p[:, 0] - q[0], p[:, 1] - q[1]))
            self.assertLess(p[i, 2], crest - 0.015)

    def test_demo_reaches_its_keyframes(self):
        from yam_arena import ik

        import demo as D
        frames = D.keyframes()
        worst = {}
        for side, fr in frames.items():
            q = np.array(D.READY, float)
            worst[side] = 0.0
            for end, kind, tgt, quat, _, *roll in fr:
                pts = [D.lid_arc(a) for a in np.linspace(0, tgt, 6)] if kind == "arc" else [np.asarray(tgt)]
                for p in pts:
                    q2, pe, _ = ik.solve(p, quat, q, base_pos=D.BASES[side], iters=200)
                    if pe > 0.002:
                        q2, pe, _ = ik.solve(p, quat, q, base_pos=D.BASES[side], iters=200, rot_weight=0.03)
                    if pe > 0.004:
                        q2, pe, _ = ik.solve(p, quat, q2, base_pos=D.BASES[side], iters=200, rot_weight=0.005)
                    q = q2
                    worst[side] = max(worst[side], pe)
        self.assertLess(worst["left"], 0.004)    # the lid tab is 30 mm deep
        self.assertLess(worst["right"], 0.004)

    def test_grasp_pose_is_exact(self):
        from yam_arena import ik

        import demo as D
        fr = D.keyframes()["right"]
        by_time = {f[0]: f for f in fr}
        hover, pinch = by_time[3.0], by_time[4.0]                           # above the crest, then down on it
        q, _, _ = ik.solve(hover[2], hover[3], D.READY, base_pos=D.BASES["right"], iters=300)
        for s in np.linspace(0.1, 1.0, 10):                                # come down as the demo does
            p = np.asarray(hover[2]) + s * (np.asarray(pinch[2]) - np.asarray(hover[2]))
            q, pe, re = ik.solve(p, pinch[3], q, base_pos=D.BASES["right"], iters=100)
        self.assertLess(pe, 0.001)
        self.assertLess(re, 0.02)

    def test_lid_open_clears_the_right_wrist(self):
        """At the drop point the right wrist (gripper link) stays at least 5 cm under the open lid."""
        from yam_arena import ik
        from yam_arena.constants import TCP_OFFSET

        import demo as D
        frames = {f[0]: f[2] for f in D.keyframes()["right"]}
        for t_drop in D.DROP_TIMES:
            p = frames[t_drop]
            q, pe, _ = ik.solve(p, D.DOWN_X, D.READY, base_pos=D.BASES["right"], iters=300, rot_weight=0.005)
            self.assertLess(pe, 0.002)
            tcp, R = ik.tcp_pose(q, D.BASES["right"])
            R = np.asarray(R) if np.asarray(R).shape == (3, 3) else ik.quat_xyzw_to_matrix(R)
            wrist = tcp - R @ np.asarray(TCP_OFFSET)
            under = D.HINGE[2] + (D.HINGE[0] - wrist[0]) * math.tan(math.radians(D.OPEN_DEG))
            self.assertGreater(under - wrist[2], 0.05)
            left = D.lid_arc(D.OPEN_DEG)
            self.assertGreater(np.linalg.norm(np.asarray(p) - left), 0.12)   # clear of the hand holding the lid
            self.assertLess(math.dist(p[:2], L.CAN_XY), 0.05)               # released near the can's centre


if __name__ == "__main__":
    unittest.main()
