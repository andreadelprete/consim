# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np
import os

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

mu = .3                            # friction coefficient
contact_frames = ['RAnkleRoll', 'LAnkleRoll']
contact_normal = np.array([0., 0., 1.])
K = 1e5*np.ones(3)
B = 1e2*np.ones(3)
anchor_slipping_method = 1

DATA_FILE_TSID = '../demo/references/romeo_walking_traj_tsid.npz'
modelPath = '/home/student/devel/src/tsid/models/romeo'
urdf = modelPath+'/urdf/romeo.urdf'
srdf = modelPath+'/srdf/romeo_collision.srdf'
dt_ref = 0.002

# robot parameters
# ----------------------------------------------
q0 = 1e-6*np.array([1774.19, -125116.90, 831881.83, 59741.07, -80404.91, -1561.47, 994969.15, -15358.75, 53810.89, -107487.89, 560647.38, -292066.73, -174982.92, -12811.18, 59509.60, -208411.22, 777854.20, -407874.10, -180094.67, -2954.87, 1581854.05, -33362.66, -71.93, -9258.54, -77.83, -771.16, -44.36, -30.71, 2628.59, 6113.36, 12.31, 1584237.88, 28815.47, 68.00, 7992.11, -51.17, 809.79, 482.24])
foot_scaling = 1.
lxp = foot_scaling*0.10                          # foot length in positive x direction
lxn = foot_scaling*0.05                          # foot length in negative x direction
lyp = foot_scaling*0.05                          # foot length in positive y direction
lyn = foot_scaling*0.05                          # foot length in negative y direction
lz = -0.07                            # foot sole height with respect to ankle joint
fMin = 0.0                          # minimum normal force
fMax = 1e6                       # maximum normal force
rf_frame_name = "RAnkleRoll"        # right foot frame name
lf_frame_name = "LAnkleRoll"        # left foot frame name


# configuration for TSID
# ----------------------------------------------
#dt = 0.002                      # controller time step
T_pre  = 0.0                    # simulation time before starting to walk
T_post = 1.5                    # simulation time after walking

w_com = 1.0                     # weight of center of mass task
w_foot = 1e0                    # weight of the foot motion task
w_contact = -1e2                 # weight of the foot in contact
w_posture = 1e-4                # weight of joint posture task
w_forceRef = 1e-5               # weight of force regularization task
w_torque_bounds = 0.0           # weight of the torque bounds
w_joint_bounds = 0.0

tau_max_scaling = 1.45           # scaling factor of torque bounds
v_max_scaling = 0.8

kp_contact = 10.0               # proportional gain of contact constraint
kp_foot = 10.0                  # proportional gain of contact constraint
kp_com = 10.0                   # proportional gain of center of mass task
kp_posture = 10.0               # proportional gain of joint posture task

# configuration for viewer
# ----------------------------------------------
PRINT_T = 1.0                   # print every PRINT_T seconds
use_viewer = 1
DISPLAY_N = 20                  # update robot configuration in viwewer every DISPLAY_N time steps
CAMERA_TRANSFORM = [3.578777551651001, 1.2937744855880737, 0.8885031342506409, 0.4116811454296112, 0.5468055009841919, 0.6109083890914917, 0.3978860676288605]
