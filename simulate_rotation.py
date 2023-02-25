import numpy as np
import math

def get_yaw_mat(yaw):
    yaw_mat = np.array([[math.cos(np.deg2rad(yaw)),-math.sin(np.deg2rad(yaw)), 0, 0],
                        [math.sin(np.deg2rad(yaw)),math.cos(np.deg2rad(yaw)), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    return yaw_mat

def get_pitch_mat(pitch):
    pitch_mat = np.array([[math.cos(np.deg2rad(pitch)), 0, math.sin(np.deg2rad(pitch)), 0],
                        [0, 1, 0, 0],
                        [-math.sin(np.deg2rad(pitch)), 0, math.cos(np.deg2rad(pitch)), 0],
                        [0, 0, 0, 1]])
    return pitch_mat

def get_roll_mat(roll):
    roll_mat = np.array([[1, 0, 0, 0],
                        [0, math.cos(np.deg2rad(roll)), -math.sin(np.deg2rad(roll)), 0],
                        [0, math.sin(np.deg2rad(roll)), math.cos(np.deg2rad(roll)), 0],
                        [0, 0, 0, 1]])
    return roll_mat

def get_trans_mat(x,y,z):
    trans_mat = np.array([[1, 0, 0, x],
                        [0, 1, 0, y],
                        [0, 0, 1, z],
                        [0, 0, 0, 1]])
    return trans_mat

# transformation parameters from lidar
# to goniometer coordinate system
l2g_yaw = 1.2
l2g_pitch = -1.8
l2g_roll = 0.7
l2g_x = 3.2
l2g_y = -2.7
l2g_z = 1.4

l_points = np.array([[480, 480],
                   [150, -150],
                   [30, 30],
                   [1, 1]])

# transformation matrices
l2g_y_mat = get_yaw_mat(l2g_yaw)
l2g_p_mat = get_pitch_mat(l2g_pitch)
l2g_r_mat = get_roll_mat(l2g_roll)
l2g_t_mat = get_trans_mat(l2g_x,l2g_y,l2g_z)
l2g_mat = np.matmul(l2g_p_mat,l2g_r_mat)
l2g_mat = np.matmul(l2g_y_mat,l2g_mat)
l2g_mat = np.matmul(l2g_t_mat,l2g_mat)
g2l_mat = np.linalg.inv(l2g_mat)
# z difference for non rotated points
print("z coords before rotation in lidar: "+str(l_points[2,0])+" "+str(l_points[2,1])+" diff: "+str(l_points[2,0]-l_points[2,1]))
roll_before = np.rad2deg(math.atan(abs(l_points[2,0]-l_points[2,1])/abs(abs(l_points[1,0]-l_points[1,1]))))
print("roll before: "+str(roll_before))
# pitch for goniometer rotation
pitch = 5
rot_p_mat = get_pitch_mat(pitch)
g_points = np.matmul(l2g_mat,l_points)
# rotate goniometer
g_points_r = np.matmul(rot_p_mat,g_points)
# transform coordinates back to lidar
l_points_r = np.matmul(g2l_mat,g_points_r)
print(l_points_r)
print("z coords after rotation in lidar: "+str(l_points_r[2,0])+" "+str(l_points_r[2,1])+" diff: "+str(l_points_r[2,0]-l_points_r[2,1]))
roll_after = np.rad2deg(math.atan(abs(l_points_r[2,0]-l_points_r[2,1])/abs(abs(l_points_r[1,0]-l_points_r[1,1]))))
print("roll after: "+str(roll_after))




    

