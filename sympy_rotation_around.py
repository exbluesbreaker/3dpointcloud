from sympy import Symbol, symbols
from sympy import sin, cos, pprint
from sympy.matrices import Matrix
from sympy.solvers import solve
from sympy import simplify
from sympy import init_printing
init_printing()

def get_yaw_mat(yaw):
    yaw_mat = Matrix([[cos(yaw),-sin(yaw), 0],
                        [sin(yaw),cos(yaw), 0],
                        [0, 0, 1]])
    return yaw_mat

def get_pitch_mat(pitch):
    pitch_mat = Matrix([[cos(pitch), 0, sin(pitch)],
                        [0, 1, 0],
                        [-sin(pitch), 0, cos(pitch)]])
    return pitch_mat

def get_roll_mat(roll):
    roll_mat = Matrix([[1, 0, 0],
                       [0, cos(roll), -sin(roll)],
                       [0, sin(roll), cos(roll)]])
    return roll_mat

# https://math.stackexchange.com/questions/4397763/3d-rotation-matrix-around-a-point-not-origin
yaw,pitch,roll,x_r,y_r,z_r = symbols("yaw pitch roll x_r y_r z_r")
# get rotation matrices around 0 0 0 origin
mat_yaw = get_yaw_mat(yaw)
mat_roll = get_roll_mat(roll)
mat_pitch = get_pitch_mat(pitch)
# new origin of rotation
rot_origin = Matrix([[x_r],
                     [y_r],
                     [z_r]])
# yaw rotation for new origin 
origin_rotated_yaw = mat_yaw*rot_origin
# make yaw rotation matrix about different origin
mat_yaw_full = mat_yaw.row_join(rot_origin-origin_rotated_yaw)
mat_yaw_full = mat_yaw_full.col_join(Matrix([[0,0,0,1]]))
print("\nYaw rotation around different origin:\n")
pprint(mat_yaw_full,wrap_line=False)

# pitch rotation for new origin
origin_rotated_pitch = mat_pitch*rot_origin
# make pitch rotation matrix about different origin
mat_pitch_full = mat_pitch.row_join(rot_origin-origin_rotated_pitch)
mat_pitch_full = mat_pitch_full.col_join(Matrix([[0,0,0,1]]))
print("\nPitch rotation around different origin:\n")
pprint(mat_pitch_full,wrap_line=False)

# roll rotation for new origin
origin_rotated_roll = mat_roll*rot_origin
# make roll rotation matrix about different origin
mat_roll_full = mat_roll.row_join(rot_origin-origin_rotated_roll)
mat_roll_full = mat_roll_full.col_join(Matrix([[0,0,0,1]]))
print("\nRoll rotation around different origin:\n")
pprint(mat_roll_full,wrap_line=False)

# combine all 3 rotation into 1 matrix
mat_ypr_full = mat_yaw_full*mat_pitch_full*mat_roll_full
print("\nYPR rotation around different origin:\n")
pprint(mat_ypr_full,wrap_line=False)
# check that new origin stays the same after rotation
print("\nCheck rotation origin after YPR rotation(should not change):\n")
pprint(simplify(mat_ypr_full*rot_origin.col_join(Matrix([[1]]))),wrap_line=False)

# pitch rotation around different origin for a point
x,y,z = symbols("x y z")
point = Matrix([[x],
                [y],
                [z],
                [1]])
print("\nPitch rotation example:\n")
pprint(simplify(mat_pitch_full*point),wrap_line=False)
