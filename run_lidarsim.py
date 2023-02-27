import lidarsim as ls
import numpy as np
import cv2

num_shots = 1008
num_layers = 360
ah_first = np.deg2rad(63)
ah_last = np.deg2rad(-63)
av_first = np.deg2rad(9)
av_last = np.deg2rad(-9)

lscene = ls.LidarScene(num_shots,num_layers,ah_first,ah_last,av_first, av_last)
# target 1
dist_to_target = 750
r1_p1 = np.asarray([dist_to_target, 450, 75])
r1_p2 = np.asarray([dist_to_target, 250, 75])
r1_p3 = np.asarray([dist_to_target, 250, -75])
r1_p4 = np.asarray([dist_to_target, 450, -75])
lscene.add_rect(r1_p1,r1_p2,r1_p3,r1_p4)

#target 2
dist_to_target = 500
r2_p1 = np.asarray([dist_to_target, -50,  25])
r2_p2 = np.asarray([dist_to_target, -150, 25])
r2_p3 = np.asarray([dist_to_target, -150, -25])
r2_p4 = np.asarray([dist_to_target, -50, -25])
lscene.add_rect(r2_p1,r2_p2,r2_p3,r2_p4)

scan = lscene.get_scan()
cv2.imshow("Lidar scan",scan)
