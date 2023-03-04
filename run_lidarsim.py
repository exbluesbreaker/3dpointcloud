from lidarsim import LidarScene, LidarTarget, Segmentation
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
from pytransform3d.transform_manager import TransformManager
import numpy as np
from math import sin,cos
import cv2

num_shots = 1008
num_layers = 360
ah_first = np.deg2rad(63)
ah_last = np.deg2rad(-63)
av_first = np.deg2rad(9)
av_last = np.deg2rad(-9)

lscene = LidarScene(num_shots,num_layers,ah_first,ah_last,av_first, av_last)

rot_yaw = pt.transform_from(
    pr.active_matrix_from_intrinsic_euler_xyz(np.array([0,0, np.deg2rad(25)])),
    np.array([0, 0, 0]))

tm = TransformManager()
tm.add_transform("lidar000", "lidar_yaw", rot_yaw)
transformation = tm.get_transform("lidar000", "lidar_yaw")

# target 1
# top part
dist_to_target = 480
pts = np.array([[dist_to_target, 183.5, 24.6, 1],
                [dist_to_target, 153.5, 24.6, 1],
                [dist_to_target, 153.5, 19.6, 1],
                [dist_to_target, 183.5, 19.6, 1]])
t1_top = LidarTarget(pts)
lscene.add_rect(t1_top)
# bottom part
pts = np.array([[dist_to_target, 171, 14.6, 1],
                [dist_to_target, 166, 14.6, 1],
                [dist_to_target, 166, -16.6, 1],
                [dist_to_target, 171, -16.6, 1]])
t1_bot = LidarTarget(pts)
lscene.add_rect(t1_bot)

# target 2
# top part
dist_to_target = 480
pts = np.array([[dist_to_target, -153.5, 24.6, 1],
                [dist_to_target, -183.5, 24.6, 1],
                [dist_to_target, -183.5, 19.6, 1],
                [dist_to_target, -153.5, 19.6, 1]])
t2_top   = LidarTarget(pts)
lscene.add_rect(t2_top)
# bottom part
pts = np.array([[dist_to_target, -166, 14.6, 1],
                [dist_to_target, -171, 14.6, 1],
                [dist_to_target, -171, -16.6, 1],
                [dist_to_target, -166, -16.6, 1]])
t2_bot = LidarTarget(pts)
lscene.add_rect(t2_bot)

# rotate scene
lscene.transform_scene(transformation)

# do segmentation and calculater roll misalignment 
scan = lscene.get_scan()
segm = Segmentation(dist_to_target,num_shots,num_layers,ah_first,ah_last,av_first, av_last)
segm.do_segmentation(scan)
segments = segm.get_segments()
roll = segm.calculate_roll()
print(roll)
cv2.imshow("Lidar scan",scan)
cv2.waitKey(1)
