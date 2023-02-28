from sympy import Symbol, symbols
from sympy import sin, cos, pprint
from sympy.matrices import Matrix
from sympy.solvers import solve
from sympy import init_printing
init_printing() 
p,h0,v0,h1,v1,r = symbols("p h0 v0 h1 v1 r")
a_11, a_12, a_13, a_14, a_21, a_22, a_23, a_24, a_31, a_32, a_33, a_34 = symbols("a_11 a_12 a_13 a_14 a_21 a_22 a_23 a_24 a_31 a_32 a_33 a_34")
mat_l2g = Matrix([[a_11, a_12, a_13, a_14],
             [a_21, a_22, a_23, a_24],
             [a_31, a_32, a_33, a_34],
             [0, 0, 0, 1]])
mat_pitch = Matrix([[cos(p), 0, sin(p), 0],
                           [0, 1, 0, 0],
                           [-sin(p), 0, cos(p), 0],
                           [0, 0, 0, 1]])
coord_pc_0 = Matrix([[r*cos(h0)*cos(v0)],
                     [r*sin(h0)*cos(v0)],
                     [r*sin(v0)],
                     [1]])
coord_pc_1 = Matrix([[r*cos(h1)*cos(v1)],
                     [r*sin(h1)*cos(v1)],
                     [r*sin(v1)],
                     [1]])
eq_0 = mat_pitch*mat_l2g*coord_pc_0
eq_1 = mat_l2g*coord_pc_1
print("Coordinates of some point from 000 in lidar coordinate system:")
pprint(coord_pc_0,wrap_line=False)
print("Coordinates of the same point in lidar coordinate system after pitch rotation in gonio:")
pprint(coord_pc_1,wrap_line=False)
print("Transformation matrix from lidar to gonio:")
pprint(mat_l2g,wrap_line=False)
print("Pitch rotation matrix in gonio coordinate system:")
pprint(mat_pitch,wrap_line=False)
print("Coordinates of rotated point in gonio coordinate system from known pitch rotation:")
pprint(eq_0,wrap_line=False)
print("Coordinates of rotated point in gonio coordinate system from known point in lidar coordinate system:")
pprint(eq_1,wrap_line=False)
print("The last 2 should be equal so we can get 3 equations")
print("Known values:")
print("\th0,v0 - h/v angles of 000 point in lidar coordinate system")
print("\th1,v1 - h/v angles of rotated point in lidar coordinate system")
print("\tp - pitch angle of rotation in gonio coordinate system")
print("Unknown values:")
print("\ta_11, a_12, a_13, a_14, a_21, a_22, a_23, a_24, a_31, a_32, a_33, a_34 - transformation matrix between lidar and gonio")
print("\tr - radial distance for given point in lidar coordinate system")
print("Issues:")
print("\ta_21, a_22, a_23, a_24 cannot be calculated because equation is degenerate")
print("\twe probably do not care about a_14, a_24 and a_34 - translation paramters because it is just a shift")
