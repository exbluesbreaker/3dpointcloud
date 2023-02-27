import numpy as np
import math

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

    def add_rect(self,p1,p2,p3,p4):
        self.targets.append({"p1":p1,
                             "p2":p2,
                             "p3":p3,
                             "p4":p4})

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
                    if ray_rect_intersection(u,r["p1"],r["p2"],r["p3"],r["p4"]):
                        scan[v_idx,h_idx] = 255
        return scan
