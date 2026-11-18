"""Scene placement and arm roles; component dimensions stay in iberville_assets.

Channel local +y points into its opening; -y is its closed back. World +y is
left. Robot bases, geometry, material assumptions and wrap target are unchanged.
"""
import numpy as np
import iberville_assets as a
from yam_arena import ik

LAYOUT_REVISION='open-left-flat-right-v2'
CHANNEL_CENTER_XY=np.array([.320,0.])
CHANNEL_YAW_DEG=0.  # +135 degrees from the previous scene; open side is world +y.
_angle=np.deg2rad(CHANNEL_YAW_DEG)
CHANNEL_ROTATION=np.array([[np.cos(_angle),-np.sin(_angle),0.],
                           [np.sin(_angle),np.cos(_angle),0.],[0.,0.,1.]])
_root=np.r_[CHANNEL_CENTER_XY,a.TOP]-CHANNEL_ROTATION@np.array([a.STUD_DEPTH/2,a.STUD_FLANGE/2,0.])
CHANNEL_POSITION=tuple(float(v) for v in _root)
CHANNEL_ROT_XYZW=tuple(float(v) for v in ik.matrix_to_quat_xyzw(CHANNEL_ROTATION))
MOUNT_HEIGHT=.940
BOX_TARGET_POSITION=np.array(CHANNEL_POSITION)+CHANNEL_ROTATION@np.array([0.,-a.WIDTH/2-a.WEB_FACE_CLEARANCE,MOUNT_HEIGHT-a.TOP])
BOX_TARGET_ROTATION=CHANNEL_ROTATION.copy()

# Stand the loose box on its lower device ear and opposite rear body edge.
# It has no fixture or attachment. A small natural lean gives a direct top grasp.
BOX_INITIAL_LEAN=float(np.arctan(.009/(a.DEPTH-.0014)))
_c,_s=np.cos(BOX_INITIAL_LEAN),np.sin(BOX_INITIAL_LEAN)
BOX_INITIAL_ROTATION=CHANNEL_ROTATION@np.array([[_c,0.,_s],[0.,1.,0.],[-_s,0.,_c]])
_points=np.concatenate([s.mesh.vertices for s in a.body_solids() if s.physical])
_z=a.TOP+.0002-float((_points@BOX_INITIAL_ROTATION.T)[:,2].min())
BOX_INITIAL_POSITION=(.210,-.150,_z)
BOX_INITIAL_ROT_XYZW=tuple(float(v) for v in ik.matrix_to_quat_xyzw(BOX_INITIAL_ROTATION))
# Tool axes for a top grasp across the box's 50.8 mm width.
GRASP_TILT_DEG=25.
_ga=np.deg2rad(GRASP_TILT_DEG);_gc,_gs=np.cos(_ga),np.sin(_ga)
TOP_GRASP_RELATIVE=np.array([[0.,_gc,-_gs],[1.,0.,0.],[0.,-_gs,-_gc]])
TOP_GRASP_OFFSET=np.array([a.DEPTH/2,0.,.020])


def make_model(**kwargs):
    import install_physics as p
    return p.make_model(stud_front=CHANNEL_POSITION[0],stud_y=CHANNEL_POSITION[1],
        stud_yaw=np.deg2rad(CHANNEL_YAW_DEG),box_position=BOX_INITIAL_POSITION,
        box_rotation=BOX_INITIAL_ROT_XYZW,channel_profile='u',**kwargs)


def metadata():
    from yam_arena import station
    from yam_arena.constants import READY
    web=np.array(CHANNEL_POSITION)+CHANNEL_ROTATION@np.array([a.STUD_DEPTH/2,a.STUD_SHEET/2,0.])
    toward=np.array(station.BASES['left'])[:2]-web[:2];toward/=np.linalg.norm(toward)
    normal=-CHANNEL_ROTATION[:2,1]
    opening=CHANNEL_ROTATION[:2,1]
    left_hand=ik.tcp_pose(READY,station.BASES['left'])[0][:2]-web[:2];left_hand/=np.linalg.norm(left_hand)
    right_hand=ik.tcp_pose(READY,station.BASES['right'])[0][:2]-web[:2];right_hand/=np.linalg.norm(right_hand)
    return {'layout_revision':LAYOUT_REVISION,'channel_profile':'u',
        'channel_center_xy_m':CHANNEL_CENTER_XY.tolist(),'channel_origin_xyz_m':list(CHANNEL_POSITION),
        'channel_yaw_degrees':CHANNEL_YAW_DEG,'channel_quaternion_xyzw':list(CHANNEL_ROT_XYZW),
        'closed_back_normal_xy':normal.tolist(),'open_side_normal_xy':opening.tolist(),
        'rotation_from_previous_scene_degrees':135.,
        'opening_vs_left_hand_angle_degrees':float(np.degrees(np.arccos(np.clip(opening@left_hand,-1,1)))),
        'closed_back_vs_right_hand_angle_degrees':float(np.degrees(np.arccos(np.clip(normal@right_hand,-1,1)))),
        'box_flap_side_normal_xyz':BOX_INITIAL_ROTATION[:,1].tolist(),
        'closed_back_vs_left_base_angle_degrees':float(np.degrees(np.arccos(np.clip(normal@toward,-1,1)))),
        'box_initial_xyz_m':list(BOX_INITIAL_POSITION),'box_initial_quaternion_xyzw':list(BOX_INITIAL_ROT_XYZW),
        'box_natural_lean_degrees':float(np.degrees(BOX_INITIAL_LEAN)),
        'right_grasp_tilt_degrees':GRASP_TILT_DEG,
        'box_seated_xyz_m':BOX_TARGET_POSITION.tolist(),'box_seated_quaternion_xyzw':list(CHANNEL_ROT_XYZW),
        'arm_roles':{'right':'Pick up the box from the right, slide it onto the channel, and hold it.',
                     'left':'Form both return wraps, then retract for the release check.'},
        'trajectory_demonstrated':False}
