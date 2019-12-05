# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np
import os

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

#N_SIMULATION = 4000             # number of time steps simulated
#dt = 0.002                      # controller time step

mu = 0.3                            # friction coefficient
contact_frames = ['BL_contact', 'BR_contact', 'FL_contact', 'FR_contact']
contact_frames += ['BL_upperleg', 'BL_shank', 'BR_upperleg', 'BR_shank', 'FL_upperleg', 'FL_shank', 'FR_upperleg', 'FR_shank']
#contact_normal = np.matrix([0., 0., 1.]).T   # direction of the normal to the contact surface
K = 1e5*np.asmatrix(np.diagflat([1., 1., 1.]))
B = 3e2*np.asmatrix(np.diagflat([1., 1., 1.]))

PRINT_T = 0.2                   # print every PRINT_T
DISPLAY_T = 0.01                 # update robot configuration in viwewer every DISPLAY_T

#filename = str(os.path.dirname(os.path.abspath(__file__)))
#path = filename + '/../models'
path = '/home/student/devel/src/tsid/models'
urdf = path + '/quadruped/urdf/quadruped.urdf'
q0 = np.matrix([[0., 0., 0.223, 0., 0., 0., 1., 
                 -0.8,  1.6, -0.8, 1.6, 
                 -0.8,  1.6, -0.8, 1.6]]).T

use_viewer = 1
CAMERA_TRANSFORM = [2.0044965744018555, 0.9386290907859802, 0.9415794014930725, 
                    0.3012915551662445, 0.49565795063972473, 0.6749107837677002, 0.45611628890037537]