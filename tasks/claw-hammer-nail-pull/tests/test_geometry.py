import unittest
import numpy as np
from nail_scene.scene import claw_prong,HAMMER_VERTICES,OBJECTS
from nail_scene.material import NAIL_DIAMETER,NAIL_HEAD_DIAMETER


class GeometryTests(unittest.TestCase):
    def test_prongs_do_not_fill_slot_and_fit_normal_nail(self):
        left,_=claw_prong(-1);right,_=claw_prong(1)
        self.assertTrue(np.all(left.v[:,0]<0))
        self.assertTrue(np.all(right.v[:,0]>0))
        # At least one actual section admits the shank but retains its head.
        inner=right.v[::4,0]
        gap=inner*2
        self.assertTrue(np.any((gap>NAIL_DIAMETER+.0002)&
                               (gap<NAIL_HEAD_DIAMETER-.0004)))

    def test_tool_is_full_size_with_nonzero_claw_faces(self):
        extent=np.ptp(HAMMER_VERTICES,axis=0)
        self.assertTrue(.280<extent[1]<.295)
        self.assertTrue(.105<extent[2]<.118)
        for sign in (-1,1):
            mesh,_=claw_prong(sign)
            edges=np.cross(mesh.v[mesh.f[:,1]]-mesh.v[mesh.f[:,0]],
                           mesh.v[mesh.f[:,2]]-mesh.v[mesh.f[:,0]])
            self.assertTrue(np.all(np.linalg.norm(edges,axis=1)>1e-12))

    def test_scene_has_only_four_loose_real_props(self):
        self.assertEqual({o.name for o in OBJECTS},{'board','nail','hammer','tray'})
        self.assertTrue(all(o.role=='free' for o in OBJECTS))
