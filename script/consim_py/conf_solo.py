# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

# N_SIMULATION = 4000             # number of time steps simulated
# dt = 0.002                      # controller time step

mu = 0.3                            # friction coefficient
contact_frames = ['HL_FOOT', 'HR_FOOT', 'FL_FOOT', 'FR_FOOT']
contact_frames += ['FL_UPPER_LEG', 'FL_LOWER_LEG', 'FR_UPPER_LEG', 'FR_LOWER_LEG',
                   'HL_UPPER_LEG', 'HL_LOWER_LEG', 'HR_UPPER_LEG', 'HR_LOWER_LEG']
# contact_normal = np.matrix([0., 0., 1.]).T   # direction of the normal to the contact surface
K = 1e5*np.asmatrix(np.diagflat([1., 1., 1.]))
B = 3e2*np.asmatrix(np.diagflat([1., 1., 1.]))

PRINT_T = 0.2                   # print every PRINT_T
DISPLAY_T = 0.01                 # update robot configuration in viwewer every DISPLAY_T

# filename = str(os.path.dirname(os.path.abspath(__file__)))
# path = filename + '/../models'
# path = '/home/student/devel/src/tsid/models'
# urdf = path + '/quadruped/urdf/quadruped.urdf'
q0 = np.array([0., 0., 0.223, 0., 0., 0., 1.,
               -0.8,  1.6, -0.8, 1.6,
               -0.8,  1.6, -0.8, 1.6]).T

use_viewer = 1
CAMERA_TRANSFORM = [1.0910934209823608, -1.4611519575119019, 0.9074661731719971,
                    0.5040678381919861, 0.17712827026844025, 0.24428671598434448, 0.8092374205589294]
