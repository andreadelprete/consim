"""has all the configuration for the python simulator  """

import numpy as np


np.set_printoptions(precision=2, linewidth=200, suppress=True,
                    formatter={'all':lambda x: "%.3f, "%x})
LINE_WIDTH = 60

# N_SIMULATION = 4000             # number of time steps simulated
# dt = 0.002                      # controller time step

mu = 0.3                            # friction coefficient
fMin = 1.0                          # minimum normal force
fMax = 1000.0                       # maximum normal force
contact_frames = ['root_joint']
# direction of the normal to the contact surface
contact_normal = np.array([0., 0., 1.])
K = 1e+5*np.asarray(np.diagflat([1., 1., 1.]))
B = 2.4e+2*np.asarray(np.diagflat([1., 1., 1.]))

w_com = 1.0                     # weight of center of mass task
w_posture = 1e-3                # weight of joint posture task
w_forceRef = 1e-6               # weight of force regularization task

kp_contact = 1.0               # proportional gain of contact constraint
kp_com = 2.0                   # proportional gain of center of mass task
kp_posture = 2.0               # proportional gain of joint posture task

PRINT_T = 0.2                   # print every PRINT_T
DISPLAY_T = 0.005                 # update robot configuration in viwewer every DISPLAY_T

q0 = np.array([0., 0., -1e-10, 0., 0., 0., 1.]) [:,None]
dq0 = np.array([0., 0., -1., 0., 0., 0.]) [:,None]
# anchor for contact points
#p0 = 1e-6*np.array([-190000.000, 150050.000, 53.853, -190000.000, -150050.000, 53.853, 
#                     190000.000, 150050.000, 53.853, 190000.000, -150050.000, 53.853, ])
p0 = None

use_viewer = False  
CAMERA_TRANSFORM = [1.0910934209823608, -1.4611519575119019, 0.9074661731719971,
                    0.5040678381919861, 0.17712827026844025, 0.24428671598434448, 0.8092374205589294]

SPHERE_RADIUS = 0.03
REF_SPHERE_RADIUS = 0.03
COM_SPHERE_COLOR = (1, 0.5, 0, 1)
COM_REF_SPHERE_COLOR = (1, 0, 0, 1)
RF_SPHERE_COLOR = (0, 1, 0, 1)
RF_REF_SPHERE_COLOR = (0, 1, 0.5, 1)
LF_SPHERE_COLOR = (0, 0, 1, 1)
LF_REF_SPHERE_COLOR = (0.5, 0, 1, 1)
