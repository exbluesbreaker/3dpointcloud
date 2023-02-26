import numpy as np
import math
import cv2
import lidarsimutils as lsu
import statistics as st

# get coordinates for point hit by lidar
# at hangle/vangle for plane
# which is perpendicular to X axis of lidar
def get_plane_xyz(hangle,vangle, dist_000):
    r = dist_000/(math.cos(hangle)*math.cos(vangle))
    x = dist_000
    y = r*math.cos(vangle)*math.sin(hangle)
    z = r*math.sin(vangle)
    # return homogeneous coordinates
    return (x,y,z,1)

# consider scene consist of a plane
# which is perpendicular to X axis of lidar
def get_scene_points(scene,hangles,vangles,dist_000):
    shape = scene.shape
    points = []
    for row in range(shape[0]):
        for col in range(shape[1]):
            if scene[row,col,0] != 0:
                point = get_plane_xyz(hangles[col],vangles[row],dist_000)
                points.append(point)
    return np.transpose(np.asarray(points))

# calculate vertical and horizontal angle
# for given 3D coordinates
def get_ha_va(x,y,z):
    hor_angle = math.atan(y/x)
    vert_angle = math.atan(math.cos(hor_angle)*z/x)
    return (hor_angle,vert_angle)

# approximate va ha indices for lidar
# for given points
def approximate_lidar_points(points,hangles,vangles):
    indices = []
    for i in range(points.shape[1]):
        x = points[0][i]
        y = points[1][i]
        z = points[2][i]
        (ha,va) = get_ha_va(x,y,z)
        ha_idx = (np.abs(hangles - ha)).argmin()
        va_idx = (np.abs(vangles - va)).argmin()
        indices.append((ha_idx,va_idx))
    return np.asarray(indices)


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

# points with pitch misalignment in lidar coordinate system
points_l_r = np.matmul(mat_g2l,points_g_r)


# make rotated gonio scene
indices_g = approximate_lidar_points(points_g_r,hor_angles,vert_angles)
new_scene = np.zeros((h,w,1), np.uint8)
for idx in indices_g:
    new_scene[idx[1],idx[0]] = 255

# find top segments for t-shapes
segments = lsu.get_segments(new_scene)
tb1_top, tb2_top = lsu.get_tshape_parts(segments)

tshape_tops = np.zeros((h,w,1), np.uint8)

for idx in tb1_top:
    tshape_tops[idx[0],idx[1]] = 255

for idx in tb2_top:
    tshape_tops[idx[0],idx[1]] = 255

top_points = get_scene_points(tshape_tops,hor_angles,vert_angles,dist_to_plane)

x = [x for x in top_points[1,:]]
y = [y for y in top_points[2,:]]
roll, b = np.polyfit(x, y, 1)

print("Roll: "+str(np.rad2deg(roll)))

cv2.imshow("Pitch rotated",new_scene)
cv2.imwrite("result.png",new_scene)

