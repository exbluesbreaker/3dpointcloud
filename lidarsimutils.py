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
