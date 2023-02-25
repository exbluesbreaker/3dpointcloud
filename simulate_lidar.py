import numpy as np
import math
import cv2
import lidarsimutils as lsu

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
    return np.asarray(points)

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
    for point in points:
        (ha,va) = get_ha_va(point[0],point[1],point[2])
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


# make 2 targets for lidar  
scene = np.zeros((h,w,1), np.uint8)
scene[100:115,320:364] = 255
scene[120:170,339:345] = 255

scene[100:115,644:688] = 255
scene[120:170,663:669] = 255

hor_angles = np.linspace(hangle_first,hangle_last,w);
vert_angles = np.linspace(vangle_first,vangle_last,h);

points = get_scene_points(scene,hor_angles,vert_angles,dist_to_plane)

# transformation parameters from lidar
# to goniometer coordinate system
l2g_yaw = 1.2
l2g_pitch = -1.8
l2g_roll = 0.7
l2g_x = 3.2
l2g_y = -2.7
l2g_z = 1.4

# transformation matrices
l2g_y_mat = lsu.get_yaw_mat(l2g_yaw)
l2g_p_mat = lsu.get_pitch_mat(l2g_pitch)
l2g_r_mat = lsu.get_roll_mat(l2g_roll)
l2g_t_mat = lsu.get_trans_mat(l2g_x,l2g_y,l2g_z)
l2g_mat = np.matmul(l2g_p_mat,l2g_r_mat)
l2g_mat = np.matmul(l2g_y_mat,l2g_mat)
l2g_mat = np.matmul(l2g_t_mat,l2g_mat)
g2l_mat = np.linalg.inv(l2g_mat)

g_points = np.transpose(points)

# rotate goniometer in pitch
pitch = 5
rot_p_mat = lsu.get_pitch_mat(pitch)
g_points = np.matmul(rot_p_mat,g_points)
# transform coordinates to lidar
l_points = np.matmul(g2l_mat,g_points)

l_points = np.transpose(l_points)


indices = approximate_lidar_points(l_points,hor_angles,vert_angles)

new_scene = np.zeros((h,w,1), np.uint8)

for idx in indices:
    new_scene[idx[1],idx[0]] = 255

cv2.imshow("Out",new_scene)

