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
        seg_feat.append((idx,min(v_idx),max(v_idx),st.mean(v_idx),min(h_idx),max(h_idx),st.mean(h_idx)))
    # detect top parts
    # sort by horizontal length
    seg_feat.sort(key=lambda a: a[5]-a[4],reverse=True)
    tb1_top = segments[seg_feat[0][0]]
    tb2_top = segments[seg_feat[1][0]]
    return (tb1_top, tb2_top)
