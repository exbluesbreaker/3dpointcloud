import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from pytransform3d import transformations as pt
import math

# get distance for given point from plane
# which is perpendicular to X axis of lidar

def get_rdist(hangle,vangle, dist_000):
    r = dist_000/(math.cos(hangle)*math.cos(vangle))
    return r

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

def ray_rect_intersection(u,p1,p2,p3,p4):
    # returns true if ray specified by vector u
    # intersects rectangle specified by point p1,p2,p3,p4
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p1
    # manual cross product for speed
    n = ([v1[1]*v2[2]-v1[2]*v2[1],
          v1[2]*v2[0]-v1[0]*v2[2],
          v1[0]*v2[1]-v1[1]*v2[0]])
    # find intersection point
    # dot(t*u,N) shall be equal to 0
    t = (p1[0]*n[0]+p1[1]*n[1]+p1[2]*n[2])/(u[0]*n[0]+u[1]*n[1]+u[2]*n[2])
    inter_p = u*t
    # https://math.stackexchange.com/questions/476608/how-to-check-if-point-is-within-a-rectangle-on-a-plane-in-3d-space
    if (np.dot(p1,v1) <= np.dot(inter_p,v1) and  np.dot(inter_p,v1) <= np.dot(p2,v1) and
        np.dot(p1,v3) <= np.dot(inter_p,v3) and np.dot(inter_p,v3) <= np.dot(p4,v3)):
            return True
    else:
            return False

class LidarTarget:
    def __init__(self,points):
        self.points = points
        self.__calculate_vectors()

    def __calculate_vectors(self):
        # calculate vectors and values used for ray intersection
        self.v1 = self.points[1,0:3] - self.points[0,0:3]
        self.v2 = self.points[2,0:3] - self.points[0,0:3]
        self.v3 = self.points[3,0:3] - self.points[0,0:3]
        self.n = np.cross(self.v1,self.v2)
        self.dot_p1_v1 = np.dot(self.points[0,0:3],self.v1)
        self.dot_p2_v1 = np.dot(self.points[1,0:3],self.v1)
        self.dot_p1_v3 = np.dot(self.points[0,0:3],self.v3)
        self.dot_p4_v3 = np.dot(self.points[3,0:3],self.v3)
        self.dot_p1_n = np.dot(self.points[0,0:3],self.n)

    def do_transformation(self, transformation):
        self.points = pt.transform(transformation,self.points)
        self.__calculate_vectors()

    def ray_rect_intersection(self,u):
        # returns true if ray specified by vector u
        # intersects rectangle specified by point p1,p2,p3,p4
        #
        # find intersection point
        # dot(t*u,N) shall be equal to 0
        t = self.dot_p1_n/(u[0]*self.n[0]+u[1]*self.n[1]+u[2]*self.n[2])
        inter_p = u*t
        # https://math.stackexchange.com/questions/476608/how-to-check-if-point-is-within-a-rectangle-on-a-plane-in-3d-space
        if (self.dot_p1_v1 <= np.dot(inter_p,self.v1) and  np.dot(inter_p,self.v1) <= self.dot_p2_v1 and
            self.dot_p1_v3 <= np.dot(inter_p,self.v3) and np.dot(inter_p,self.v3) <= self.dot_p4_v3):
            return True
        else:
            return False

class LidarScene:
    def __init__(self, num_shots, num_layers,min_ha,max_ha,min_va,max_va):
        self.num_shots = num_shots
        self.num_layers = num_layers
        self.hor_angles = np.linspace(min_ha,max_ha,num_shots)
        self.vert_angles = np.linspace(min_va,max_va,num_layers)
        self.targets = []
        # make sin cos cache
        self.sincos_cache = {"sin_ha":[],
                             "cos_ha":[],
                             "sin_va":[],
                             "cos_va":[]}
        for h_idx in range(num_shots):
            self.sincos_cache["sin_ha"].append(math.sin(self.hor_angles[h_idx]))
            self.sincos_cache["cos_ha"].append(math.cos(self.hor_angles[h_idx]))
        for v_idx in range(num_layers):
            self.sincos_cache["sin_va"].append(math.sin(self.vert_angles[v_idx]))
            self.sincos_cache["cos_va"].append(math.cos(self.vert_angles[v_idx]))

    def add_rect(self,target):
        self.targets.append(target)

    def transform_scene(self, transformation):
        for t in self.targets:
            t.do_transformation(transformation)

    def get_unit_vector(self,h_idx,v_idx):
        # return unit vector of lidar shot for given angles
        cosv = self.sincos_cache["cos_va"][v_idx]
        sinv = self.sincos_cache["sin_va"][v_idx]
        sinh_cosv = self.sincos_cache["sin_ha"][h_idx]*cosv
        cosh_cosv = self.sincos_cache["cos_ha"][h_idx]*cosv
        x = cosh_cosv
        y = sinh_cosv
        z = sinv
        return np.asarray([x, y, z])

    def get_scan(self):
        # make empty scan
        scan = np.zeros((self.num_layers,self.num_shots,1), np.uint8)
        # simulate lidar shots
        for h_idx in range(self.num_shots):
            for v_idx in range(self.num_layers):
                u = self.get_unit_vector(h_idx,v_idx)
                for r in self.targets:
                    if r.ray_rect_intersection(u):
                        scan[v_idx,h_idx] = 255
        return scan

class Segmentation:
    def __init__(self,dist_to_target,num_shots, num_layers,min_ha,max_ha,min_va,max_va):
        self.dist_to_target = dist_to_target
        self.num_shots = num_shots
        self.num_layers = num_layers
        self.hor_angles = np.linspace(min_ha,max_ha,num_shots)
        self.vert_angles = np.linspace(min_va,max_va,num_layers)
        pass

    def do_segmentation(self,scan):
        is_safe_dfs = lambda s,v,r,c: r>=0 and r<s.shape[0] and c>=0 and c<s.shape[1] and v[r][c] == False and s[r,c] != 0
        shape = scan.shape
        count = 0
        visited = np.zeros((shape[0],shape[1]), np.bool)
        self.segments = []
        for row in range(shape[0]):
            for col in range(shape[1]):
                if visited[row][col] == False and scan[row,col] != 0:
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
                        if is_safe_dfs(scan,visited,dfs_row-1,dfs_col):
                            nodes_to_visit.append((dfs_row-1,dfs_col))
                        if is_safe_dfs(scan,visited,dfs_row+1,dfs_col):
                            nodes_to_visit.append((dfs_row+1,dfs_col))
                        if is_safe_dfs(scan,visited,dfs_row,dfs_col-1):
                            nodes_to_visit.append((dfs_row,dfs_col-1))
                        if is_safe_dfs(scan,visited,dfs_row,dfs_col+1):
                            nodes_to_visit.append((dfs_row,dfs_col+1))
                    self.segments.append(segment)
        # assign segments to target parts
        self.extract_features()
        self.assign_segments()

    def get_segments(self):
        return self.segments

    def extract_features(self):
        self.seg_feat = []
        for idx in range(len(self.segments)):
            seg = self.segments[idx]
            v_idx = [pt[0] for pt in seg]
            h_idx = [pt[1] for pt in seg]
            self.seg_feat.append({"id":idx,
                                  "v_min":min(v_idx),
                                  "v_max":max(v_idx),
                                  "v_mean":mean(v_idx),
                                  "h_min":min(h_idx),
                                  "h_max":max(h_idx),
                                  "h_mean":mean(h_idx),
                                  "size":len(seg)})

    def assign_segments(self):
        # detect top parts
        # sort by horizontal length
        self.seg_feat.sort(key=lambda a: a["h_max"]-a["h_min"],reverse=True)
        tb1_top = self.segments[self.seg_feat[0]["id"]]
        tb2_top = self.segments[self.seg_feat[1]["id"]]
        tb1_bot_id = None
        tb2_bot_id = None
        # find bottom part for tb1
        for i in range(2,len(self.seg_feat)):
            # check segments in horizontal boundaries of tb1_top
            if self.seg_feat[i]["h_mean"] >= self.seg_feat[0]["h_min"] and self.seg_feat[i]["h_mean"] <= self.seg_feat[0]["h_max"]:
                # change currently assigned botom segment if we found a bigger one
                if tb1_bot_id == None or self.seg_feat[i]["size"]>self.seg_feat[tb1_bot_id]["size"]:
                    tb1_bot_id = i
            # check segments in horizontal boundaries of tb2_top
            if self.seg_feat[i]["h_mean"] >= self.seg_feat[1]["h_min"] and self.seg_feat[i]["h_mean"] <= self.seg_feat[1]["h_max"]:
                # change currently assigned botom segment if we found a bigger one
                if tb2_bot_id == None or self.seg_feat[i]["size"]>self.seg_feat[tb2_bot_id]["size"]:
                    tb2_bot_id = i
        if tb1_bot_id == None or tb2_bot_id == None:
            print("Cannot find bottom segment")
        tb1_bot = self.segments[self.seg_feat[tb1_bot_id]["id"]]
        tb2_bot = self.segments[self.seg_feat[tb2_bot_id]["id"]]
        self.tb1_top = tb1_top
        self.tb2_top = tb2_top
        self.tb1_bot = tb1_bot
        self.tb2_bot = tb2_bot

    def calculate_roll(self):
        points_3d = []
        for p in self.tb1_top:
            point = get_plane_xyz(self.hor_angles[p[1]],self.vert_angles[p[0]],self.dist_to_target)
            points_3d.append(point)
        for p in self.tb2_top:
            point = get_plane_xyz(self.hor_angles[p[1]],self.vert_angles[p[0]],self.dist_to_target)
            points_3d.append(point)
        x = np.asarray([p[1] for p in points_3d])
        y = np.asarray([p[2] for p in points_3d])
        a, b = np.polyfit(x, y, 1)
        return np.rad2deg(math.atan(a))

        
