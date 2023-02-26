import numpy as np
import math
import statistics as st

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

def get_segments(scene):
    is_safe_dfs = lambda s,v,r,c: r>=0 and r<s.shape[0] and c>=0 and c<s.shape[1] and v[r][c] == False and s[r,c,0] != 0
    shape = scene.shape
    count = 0
    visited = np.zeros((shape[0],shape[1]), np.bool)
    segments = []
    for row in range(shape[0]):
        for col in range(shape[1]):
            if visited[row][col] == False and scene[row,col,0] != 0:
                nodes_to_visit = [(row,col)];
                count += 1
                segment = []
                # DFS algorithm to detect segments
                while nodes_to_visit:
                    dfs_row,dfs_col = nodes_to_visit.pop(0)
                    if visited[dfs_row][dfs_col] == True:
                        continue
                    segment.append((dfs_row,dfs_col))
                    visited[dfs_row][dfs_col] = True
                    # use 4 connectivity for segments
                    if is_safe_dfs(scene,visited,dfs_row-1,dfs_col):
                        nodes_to_visit.append((dfs_row-1,dfs_col))
                    if is_safe_dfs(scene,visited,dfs_row+1,dfs_col):
                        nodes_to_visit.append((dfs_row+1,dfs_col))
                    if is_safe_dfs(scene,visited,dfs_row,dfs_col-1):
                        nodes_to_visit.append((dfs_row,dfs_col-1))
                    if is_safe_dfs(scene,visited,dfs_row,dfs_col+1):
                        nodes_to_visit.append((dfs_row,dfs_col+1))
                segments.append(segment)
    return segments

def get_tshape_parts(segments):
    seg_feat = []
    for idx in range(len(segments)):
        seg = segments[idx]
        v_idx = [pt[0] for pt in seg]
        h_idx = [pt[1] for pt in seg]
        seg_feat.append((idx,min(v_idx),max(v_idx),st.mean(v_idx),min(h_idx),max(h_idx),st.mean(h_idx),len(seg)))
    # detect top parts
    # sort by horizontal length
    seg_feat.sort(key=lambda a: a[5]-a[4],reverse=True)
    tb1_top = segments[seg_feat[0][0]]
    tb2_top = segments[seg_feat[1][0]]
    tb1_bot_id = None
    tb2_bot_id = None
    # find bottom part for tb1
    for i in range(2,len(seg_feat)):
        # check segments in horizontal boundaries of tb1_top
        if seg_feat[i][6] >= seg_feat[0][4] and seg_feat[i][6] <= seg_feat[0][5]:
            # change currently assigned botom segment if we found a bigger one
            if tb1_bot_id == None or seg_feat[i][7]>seg_feat[tb1_bot_id][7]:
                tb1_bot_id = i
        # check segments in horizontal boundaries of tb2_top
        if seg_feat[i][6] >= seg_feat[1][4] and seg_feat[i][6] <= seg_feat[1][5]:
            # change currently assigned botom segment if we found a bigger one
            if tb2_bot_id == None or seg_feat[i][7]>seg_feat[tb2_bot_id][7]:
                tb2_bot_id = i
    if tb1_bot_id == None or tb2_bot_id == None:
        print("Cannot find bottom segment")
    tb1_bot = segments[seg_feat[tb1_bot_id][0]]
    tb2_bot = segments[seg_feat[tb2_bot_id][0]]
    return (tb1_top, tb2_top, tb1_bot, tb2_bot)

# get distance for given point from plane
# which is perpendicular to X axis of lidar

def get_rdist(hangle,vangle, dist_000):
    r = dist_000/(math.cos(hangle)*math.cos(vangle))
    return r

# get xyz for point
def get_xyz(hangle,vangle, rdist):
    x = rdist*math.cos(vangle)*math.cos(hangle)
    y = rdist*math.cos(vangle)*math.sin(hangle)
    z = rdist*math.sin(vangle)
    # return homogeneous coordinates
    return (x,y,z,1)


# get coordinates for point hit by lidar
# at hangle/vangle for plane
# which is perpendicular to X axis of lidar
def get_plane_xyz(hangle,vangle, dist_000):
    r = get_rdist(hangle,vangle, dist_000)
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
    for i in range(points.shape[0]):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        (ha,va) = get_ha_va(x,y,z)
        ha_idx = (np.abs(hangles - ha)).argmin()
        va_idx = (np.abs(vangles - va)).argmin()
        indices.append((ha_idx,va_idx))
    return np.asarray(indices)
