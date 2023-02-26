from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
import math
import cv2
import numpy as np
from pytransform3d.transform_manager import TransformManager
import lidarsimutils as lsu

yaw_g2l = np.deg2rad(1.2)
pitch_g2l = 0#np.deg2rad(-2.3)
roll_g2l = 0#np.deg2rad(-0.6)
x_g2l = 1.6
y_g2l = -4.1
z_g2l = 2.2

pitch_gonio = np.deg2rad(-5)

gonio2lidar = pt.transform_from(
    pr.active_matrix_from_intrinsic_euler_xyz(np.array([roll_g2l,pitch_g2l, yaw_g2l])),
    np.array([x_g2l, y_g2l, z_g2l]))

gonio2pitch = pt.transform_from(
    pr.active_matrix_from_intrinsic_euler_xyz(np.array([0,pitch_gonio, 0])),
    np.array([0, 0, 0]))

tm = TransformManager()
tm.add_transform("goniometer000", "lidar", gonio2lidar)
tm.add_transform("goniometer000", "gonio_pitch5", gonio2pitch)

gonio5p2lidar = tm.get_transform("gonio_pitch5", "lidar")

# make 000 scene
h = 360
w = 1008
dist_to_plane = 480

hangle_first = np.deg2rad(63);
hangle_last = np.deg2rad(-63);

vangle_first = np.deg2rad(9);
vangle_last = np.deg2rad(-9);

hor_angles = np.linspace(hangle_first,hangle_last,w);
vert_angles = np.linspace(vangle_first,vangle_last,h);


# make 2 targets in for of t-shape for lidar
scan_000 = np.zeros((h,w,1), np.uint8)
scan_000[100:115,320:364] = 255
scan_000[120:170,339:345] = 255
scan_000[100:115,644:688] = 255
scan_000[120:170,663:669] = 255
cv2.imshow("Inital scan",scan_000)

points_g = lsu.get_scene_points(scan_000,hor_angles,vert_angles,dist_to_plane)
points_g_p5 = pt.transform(gonio2pitch,points_g)

# make pitch misalignment scene in gonio coordinates
scan_gonio_p5 = lsu.points2scan(points_g_p5,hor_angles,vert_angles)
cv2.imshow("Pitch rotation gonio",scan_gonio_p5)

# make pitch misalignment scene in lidar coordinates
points_l = pt.transform(gonio2lidar,points_g_p5)
scan_lidar_p5 = lsu.points2scan(points_l,hor_angles,vert_angles)
cv2.imshow("Pitch rotation lidar",scan_lidar_p5)
