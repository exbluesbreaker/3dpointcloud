import numpy as np
import math
import cv2

# get coordinates for point hit by lidar
# at hangle/vangle for plane
# which is perpendicular to X axis of lidar
def get_plane_xyz(hangle,vangle, dist_000):
    r = dist_000/(math.cos(hangle)*math.cos(vangle))
    x = dist_000
    y = r*math.cos(vangle)*math.sin(hangle)
    z = r*math.sin(vangle)
    return (x,y,z)

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
indices = approximate_lidar_points(points,hor_angles,vert_angles)

new_scene = np.zeros((h,w,1), np.uint8)

for idx in indices:
    new_scene[idx[1],idx[0]] = 255

cv2.imshow("Out",new_scene)

