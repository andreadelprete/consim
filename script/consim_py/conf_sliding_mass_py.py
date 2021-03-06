# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np
import os

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
K = 1e5*np.asarray(np.diagflat([1., 1., 1.]))
B = 2e2*np.asarray(np.diagflat([1., 1., 1.]))

q0 = np.array([0., 0., -1e-4, 0., 0., 0., 1.])
dq0 = np.array([0., 0., 0., 0., 0., 0.])

w_com = 1.0                     # weight of center of mass task
w_posture = 1e-3                # weight of joint posture task
w_forceRef = 1e-6               # weight of force regularization task

kp_contact = 1.0               # proportional gain of contact constraint
kp_com = 2.0                   # proportional gain of center of mass task
kp_posture = 2.0               # proportional gain of joint posture task

PRINT_T = 0.2                   # print every PRINT_T
DISPLAY_T = 0.005                 # update robot configuration in viwewer every DISPLAY_T

p0 = np.zeros(3)

urdf_path = os.path.abspath('../models/urdf/free_flyer.urdf')
mesh_path = os.path.abspath('../models')
use_viewer = 1
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
