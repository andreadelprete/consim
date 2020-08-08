# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np
import os
from example_robot_data.robots_loader import getModelPath

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

mu = 1.                            # friction coefficient
contact_frames = ['leg_right_sole_fix_joint', 'leg_left_sole_fix_joint']
#contact_frames = ['base_link', 'leg_left_1_joint', 'leg_left_1_link', 'leg_left_2_joint', 'leg_left_2_link', 'leg_left_3_joint', 'leg_left_3_link', 'leg_left_4_joint', 'leg_left_4_link', 'leg_left_5_joint', 'leg_left_5_link', 'leg_left_6_joint', 'leg_left_6_link', 'leg_left_sole_fix_joint', 'left_sole_link', 'leg_right_1_joint', 'leg_right_1_link', 'leg_right_2_joint', 'leg_right_2_link', 'leg_right_3_joint', 'leg_right_3_link', 'leg_right_4_joint', 'leg_right_4_link', 'leg_right_5_joint', 'leg_right_5_link', 'leg_right_6_joint', 'leg_right_6_link', 'leg_right_sole_fix_joint', 'right_sole_link', 'torso_1_joint', 'torso_1_link', 'torso_2_joint', 'torso_2_link', 'arm_left_1_joint', 'arm_left_1_link', 'arm_left_2_joint', 'arm_left_2_link', 'arm_left_3_joint', 'arm_left_3_link', 'arm_left_4_joint', 'arm_left_4_link', 'arm_left_5_joint', 'arm_left_5_link', 'arm_left_6_joint', 'arm_left_6_link', 'arm_left_7_joint', 'arm_left_7_link', 'wrist_left_ft_joint', 'wrist_left_ft_link', 'wrist_left_tool_joint', 'wrist_left_ft_tool_link', 'gripper_left_base_link_joint', 'gripper_left_base_link', 'gripper_left_inner_double_joint', 'gripper_left_inner_double_link', 'gripper_left_fingertip_1_joint', 'gripper_left_fingertip_1_link', 'gripper_left_fingertip_2_joint', 'gripper_left_fingertip_2_link', 'gripper_left_inner_single_joint', 'gripper_left_inner_single_link', 'gripper_left_fingertip_3_joint', 'gripper_left_fingertip_3_link', 'gripper_left_joint', 'gripper_left_motor_double_link', 'gripper_left_motor_single_joint', 'gripper_left_motor_single_link', 'arm_right_1_joint', 'arm_right_1_link', 'arm_right_2_joint', 'arm_right_2_link', 'arm_right_3_joint', 'arm_right_3_link', 'arm_right_4_joint', 'arm_right_4_link', 'arm_right_5_joint', 'arm_right_5_link', 'arm_right_6_joint', 'arm_right_6_link', 'arm_right_7_joint', 'arm_right_7_link', 'wrist_right_ft_joint', 'wrist_right_ft_link', 'wrist_right_tool_joint', 'wrist_right_ft_tool_link', 'gripper_right_base_link_joint', 'gripper_right_base_link', 'gripper_right_inner_double_joint', 'gripper_right_inner_double_link', 'gripper_right_fingertip_1_joint', 'gripper_right_fingertip_1_link', 'gripper_right_fingertip_2_joint', 'gripper_right_fingertip_2_link', 'gripper_right_inner_single_joint', 'gripper_right_inner_single_link', 'gripper_right_fingertip_3_joint', 'gripper_right_fingertip_3_link', 'gripper_right_joint', 'gripper_right_motor_double_link', 'gripper_right_motor_single_joint', 'gripper_right_motor_single_link', 'head_1_joint', 'head_1_link', 'head_2_joint', 'head_2_link', 'rgbd_joint', 'rgbd_link', 'rgbd_depth_joint', 'rgbd_depth_frame', 'rgbd_depth_optical_joint', 'rgbd_depth_optical_frame', 'rgbd_optical_joint', 'rgbd_optical_frame', 'rgbd_rgb_joint', 'rgbd_rgb_frame', 'rgbd_rgb_optical_joint', 'rgbd_rgb_optical_frame', 'imu_joint', 'imu_link', 'leg_right_sole_fix_joint_0', 'leg_right_sole_fix_joint_1', 'leg_right_sole_fix_joint_2', 'leg_right_sole_fix_joint_3', 'leg_left_sole_fix_joint_0', 'leg_left_sole_fix_joint_1', 'leg_left_sole_fix_joint_2', 'leg_left_sole_fix_joint_3']
contact_normal = np.array([0., 0., 1.])
K = 1e5*np.ones(3)
B = 3e2*np.ones(3)
anchor_slipping_method = 1
unilateral_contacts = 1

DATA_FILE_TSID = '../demo/references/talos_walking_traj_tsid.npz'
urdf = "/talos_data/robots/talos_reduced.urdf"
modelPath = getModelPath(urdf)
urdf = modelPath + urdf
srdf = modelPath + '/talos_data/srdf/talos.srdf'
path = os.path.join(modelPath, '../..')

dt_ref = 0.002

# robot parameters
# ----------------------------------------------
q0 = 1e-6*np.array([      0.,       0., 1020270.,       0.,       0.,       0., 1000000.,       0.,       0., -411354.,  859395., -448041.,   -1708.,       0.,       0., -411354.,  859395., -448041.,   -1708.,
             0.,    6761.,  258470.,  173046.,    -200., -525366.,       0.,       0.,  100000.,       0., -258470., -173046.,     200., -525366.,       0.,       0.,  100000.,       0.,       0.,
             0.])
foot_scaling = 1.
lxp = foot_scaling*0.10                          # foot length in positive x direction
lxn = foot_scaling*0.10                          # foot length in negative x direction
lyp = foot_scaling*0.065                          # foot length in positive y direction
lyn = foot_scaling*0.065                          # foot length in negative y direction
lz = -0.0                            # foot sole height with respect to contact frame
fMin = 0.0                          # minimum normal force
fMax = 1e6                       # maximum normal force
rf_frame_name = "leg_right_sole_fix_joint"  # right foot frame name
lf_frame_name = "leg_left_sole_fix_joint"   # left foot frame name
waist_frame_name = 'base_link' #'torso_2_link'

# configuration for TSID
# ----------------------------------------------
nv = 38
#dt = 0.002                      # controller time step
T_pre  = 2.0                    # simulation time before starting to walk
T_post = 2.0                    # simulation time after walking

w_com = 1.0                     # weight of center of mass task
w_foot = 1e0                    # weight of the foot motion task
w_contact = -1e2                 # weight of the foot in contact
w_waist = 1e-2
w_posture = 1e-3                # weight of joint posture task
w_forceRef = 1e-5               # weight of force regularization task
w_torque_bounds = 0.0           # weight of the torque bounds
w_joint_bounds = 0.0

tau_max_scaling = 1.45           # scaling factor of torque bounds
v_max_scaling = 0.8

kp_contact = 10.0               # proportional gain of contact constraint
kp_foot = 10.0                  # proportional gain of foot task
kp_com = 10.0                   # proportional gain of center of mass task
kp_waist = 10.0
kp_posture = np.array(  # gain vector for postural task
    [
        10.,
        5.,
        5.,
        1.,
        1.,
        10.,  # lleg  #low gain on axis along y and knee
        10.,
        5.,
        5.,
        1.,
        1.,
        10.,  #rleg
        10.,
        10.,  #chest
        10.,
        10.,
        10.,
        10.,
        10.,
        10.,
        10.,
        10.,  #larm
        10.,
        10.,
        10.,
        10.,
        10.,
        10.,
        10.,
        10.,  #rarm
        10.,
        10.
    ]  #head
)

# configuration for viewer
# ----------------------------------------------
PRINT_T = 1.0                   # print every PRINT_T seconds
use_viewer = 0
DISPLAY_N = 20                  # update robot configuration in viwewer every DISPLAY_N time steps
CAMERA_TRANSFORM = [3.578777551651001, 1.2937744855880737, 0.8885031342506409, 0.4116811454296112, 0.5468055009841919, 0.6109083890914917, 0.3978860676288605]
SPHERE_RADIUS = 0.01
SPHERE_COLOR  = (1, 0., 0, 1)
