import numpy as np
import math
import cv2
import lidarsimutils as lsu
import statistics as st
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt




h = 360
w = 1008
dist_to_plane = 480

hangle_first = np.deg2rad(63);
hangle_last = np.deg2rad(-63);

vangle_first = np.deg2rad(9);
vangle_last = np.deg2rad(-9);


# make 2 targets in for of t-shape for lidar
scene = np.zeros((h,w,1), np.uint8)
scene[100:115,320:364] = 255
scene[120:170,339:345] = 255

scene[100:115,644:688] = 255
scene[120:170,663:669] = 255

hor_angles = np.linspace(hangle_first,hangle_last,w);
vert_angles = np.linspace(vangle_first,vangle_last,h);

points_g = get_scene_points(scene,hor_angles,vert_angles,dist_to_plane)

# transformation parameters from lidar
# to goniometer coordinate system
yaw_g2l = 1.2
pitch_g2l = 0
roll_g2l = 0
x_g2l = 0
y_g2l = 0
z_g2l = 0

# transformation matrices
mat_y_g2l = lsu.get_yaw_mat(yaw_g2l)
mat_p_g2l = lsu.get_pitch_mat(pitch_g2l)
mat_r_g2l = lsu.get_roll_mat(roll_g2l)
mat_t_g2l = lsu.get_trans_mat(x_g2l,y_g2l,z_g2l)
mat_g2l = np.matmul(mat_p_g2l,mat_r_g2l)
mat_g2l = np.matmul(mat_y_g2l,mat_g2l)
mat_g2l = np.matmul(mat_t_g2l,mat_g2l)
mat_l2g = np.linalg.inv(mat_g2l)

# simulate goniometer pitch rotation
pitch = -5
mat_rot_p = lsu.get_pitch_mat(pitch)
# points with pitch misalignment in gonio coordinate system
points_g_r = np.matmul(mat_rot_p,points_g)

# 000 points in lidar coordinate system
points_l = np.matmul(mat_g2l,points_g)


# make 000 scene in lidar coordinates
indices_l = approximate_lidar_points(points_l,hor_angles,vert_angles)
new_scene = np.zeros((h,w,1), np.uint8)
for idx in indices_l:
    new_scene[idx[1],idx[0]] = 255

# find top segments for t-shapes
segments = lsu.get_segments(new_scene)
tb1_top, tb2_top, tb1_bot, tb2_bot = lsu.get_tshape_parts(segments)

idx_tb1_top_mid  = (round(st.mean([pt[0] for pt in tb1_top])),round(st.mean([pt[1] for pt in tb1_top])))
pt_tb1_top_mid = get_plane_xyz(hor_angles[idx_tb1_top_mid[1]],vert_angles[idx_tb1_top_mid[0]], dist_to_plane)
pt_tb1_top_mid_rdist = get_rdist(hor_angles[idx_tb1_top_mid[1]],vert_angles[idx_tb1_top_mid[0]], dist_to_plane)
idx_tb2_top_mid  = (round(st.mean([pt[0] for pt in tb2_top])),round(st.mean([pt[1] for pt in tb2_top])))
pt_tb2_top_mid = get_plane_xyz(hor_angles[idx_tb2_top_mid[1]],vert_angles[idx_tb2_top_mid[0]], dist_to_plane)
pt_tb2_top_mid_rdist = get_rdist(hor_angles[idx_tb2_top_mid[1]],vert_angles[idx_tb2_top_mid[0]], dist_to_plane)
idx_tb1_bot_mid  = (round(st.mean([pt[0] for pt in tb1_bot])),round(st.mean([pt[1] for pt in tb1_bot])))
pt_tb1_bot_mid = get_plane_xyz(hor_angles[idx_tb1_bot_mid[1]],vert_angles[idx_tb1_bot_mid[0]], dist_to_plane)
pt_tb1_bot_mid_rdist = get_rdist(hor_angles[idx_tb1_bot_mid[1]],vert_angles[idx_tb1_bot_mid[0]], dist_to_plane)
idx_tb2_bot_mid  = (round(st.mean([pt[0] for pt in tb2_bot])),round(st.mean([pt[1] for pt in tb2_bot])))
pt_tb2_bot_mid = get_plane_xyz(hor_angles[idx_tb2_bot_mid[1]],vert_angles[idx_tb2_bot_mid[0]], dist_to_plane)
pt_tb2_bot_mid_rdist = get_rdist(hor_angles[idx_tb2_bot_mid[1]],vert_angles[idx_tb2_bot_mid[0]], dist_to_plane)



# pitch misalignment points in lidar coordinate system
points_l_r = np.matmul(mat_g2l,points_g_r)

points_l_r = np.matmul(mat_l2g,points_l_r)


# make pitch misalignment scene in lidar coordinates
indices_l_r = approximate_lidar_points(points_l_r,hor_angles,vert_angles)
new_scene = np.zeros((h,w,1), np.uint8)
for idx in indices_l_r:
    new_scene[idx[1],idx[0]] = 255

# find top segments for t-shapes
segments = lsu.get_segments(new_scene)
tb1_top, tb2_top, tb1_bot, tb2_bot = lsu.get_tshape_parts(segments)

idx_tb1_top_mid  = (round(st.mean([pt[0] for pt in tb1_top])),round(st.mean([pt[1] for pt in tb1_top])))
pt_tb1_top_mid_r = get_xyz(hor_angles[idx_tb1_top_mid[1]],vert_angles[idx_tb1_top_mid[0]], pt_tb1_top_mid_rdist)
idx_tb2_top_mid  = (round(st.mean([pt[0] for pt in tb2_top])),round(st.mean([pt[1] for pt in tb2_top])))
pt_tb2_top_mid_r = get_xyz(hor_angles[idx_tb2_top_mid[1]],vert_angles[idx_tb2_top_mid[0]], pt_tb2_top_mid_rdist)
idx_tb1_bot_mid  = (round(st.mean([pt[0] for pt in tb1_bot])),round(st.mean([pt[1] for pt in tb1_bot])))
pt_tb1_bot_mid_r = get_xyz(hor_angles[idx_tb1_bot_mid[1]],vert_angles[idx_tb1_bot_mid[0]], pt_tb1_bot_mid_rdist)
idx_tb2_bot_mid  = (round(st.mean([pt[0] for pt in tb2_bot])),round(st.mean([pt[1] for pt in tb2_bot])))
pt_tb2_bot_mid_r = get_xyz(hor_angles[idx_tb2_bot_mid[1]],vert_angles[idx_tb2_bot_mid[0]], pt_tb2_bot_mid_rdist)

vector1 = (pt_tb1_top_mid_r[0]-pt_tb2_bot_mid_r[0],pt_tb1_top_mid_r[1]-pt_tb2_bot_mid_r[1],pt_tb1_top_mid_r[2]-pt_tb2_bot_mid_r[2])
vector2 = (pt_tb2_top_mid_r[0]-pt_tb1_bot_mid_r[0],pt_tb2_top_mid_r[1]-pt_tb1_bot_mid_r[1],pt_tb2_top_mid_r[2]-pt_tb1_bot_mid_r[2])

normal = np.cross(np.asarray(vector1),np.asarray(vector2))
normal = normal/np.linalg.norm(normal)

print(np.rad2deg(math.asin(normal[2])))

print(pr.active_matrix_from_angle(2,np.deg2rad(yaw_g2l)))
print(mat_y_g2l)
 

cv2.imshow("Pitch rotated",new_scene)
cv2.imwrite("result.png",new_scene)

