# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np

np.set_printoptions(precision=2, linewidth=200, suppress=True,
                    formatter={'all':lambda x: "%.3f, "%x})
LINE_WIDTH = 60

fMin = 1
fMax = 1e4
mu = 1.0*1e3                            # friction coefficient
contact_frames = ['HL_FOOT', 'HR_FOOT', 'FL_FOOT', 'FR_FOOT']
# direction of the normal to the contact surface
contact_normal = np.array([0., 0., 1.])
K = 1e5*np.ones(3)
B = 1e2*np.ones(3)

anchor_slipping_method = 1

PRINT_T = 0.2                   # print every PRINT_T
DISPLAY_T = 0.005                 # update robot configuration in viwewer every DISPLAY_T

# TSID
w_com = 1.0                     # weight of center of mass task
w_posture = 1e-3                # weight of joint posture task
w_forceRef = 1e-6               # weight of force regularization task
kp_contact = 1.0               # proportional gain of contact constraint
kp_com = 2.0                   # proportional gain of center of mass task
kp_posture = 2.0               # proportional gain of joint posture task

q0 = 1e-6*np.array([0.605, -0.083, 223001.532, 0.187, 5.332, -0.000, 1000000.000, 
                    0.0, -799754.095, 1599492.290, 0.0, -799755.961, 1599496.022, 
                    0.0, -799781.180, 1599546.460, 0.0, -799783.045, 1599550.191])
#q0[2] -= 1e-5
v0 = np.zeros(18)
p0 = None

use_viewer = 1
CAMERA_TRANSFORM = [1.0910934209823608, -1.4611519575119019, 0.9074661731719971,
                    0.5040678381919861, 0.17712827026844025, 0.24428671598434448, 0.8092374205589294]
