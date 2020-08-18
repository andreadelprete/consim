import numpy as np 
import pinocchio as pin 
import consim 
import os, sys
import matplotlib.pyplot as plt 
from consim_py.utils import visualize
from consim_py.utils.visualize  import getModelPath, getVisualPath 
from os.path import dirname, exists, join



    
if  __name__=="__main__":
    URDF_FILENAME = "solo12.urdf"
    SRDF_FILENAME = "solo.srdf"
    SRDF_SUBPATH = "/solo_description/srdf/" + SRDF_FILENAME
    URDF_SUBPATH = "/solo_description/robots/" + URDF_FILENAME
    modelPath = getModelPath(URDF_SUBPATH)
    contact_names = []

    visualizer = visualize.Visualizer()

    visual_options = {'contact_names': ['HL_FOOT', 'HR_FOOT', 'FL_FOOT', 'FR_FOOT'], 
                      'robot_color': [.7,.7,.7,.4],
                      'force_color': [1., 0., 0., .75],
                      'force_radius': .002, 
                      'force_length': .025,
                      'cone_color': [0., 1., 0., .3],
                      'cone_length': .02,
                      'friction_coeff': .5
                      }


    viz_object = visualize.ConsimVisual("solo-Euler", modelPath + URDF_SUBPATH, 
                        [getVisualPath(modelPath)], pin.JointModelFreeFlyer(), 
                        visualizer, visual_options)

    q = np.zeros(viz_object.model.nq)
    q[6] = 1 # quaternion 
    q[2] = 1. 
    
    viz_object.display(q)
    #
    hyq_urdf_file = "hyq_no_sensors.urdf"
    hyq_srdf_file = "hyq.srdf"
    HYQ_URDF_SUBPATH = "/hyq_description/robots/" + hyq_urdf_file
    HYQ_SRDF_SUBPATH = "/hyq_description/srdf/" + hyq_srdf_file
    hyq_modelPath = getModelPath(URDF_SUBPATH)

    # viz_object2 = visualize.ConsimVisual("hyq-Exp", hyq_modelPath + HYQ_URDF_SUBPATH, 
    #                     [getVisualPath(hyq_modelPath)], pin.JointModelFreeFlyer(), 
    #                     visualizer.viewer, visualizer.sceneName)
    # #
    # q2 = np.zeros(viz_object2.model.nq)
    # q2[6] = 1 # quaternion 
    # q2[1] = 1.
    # q2[2] = 1. 
    # viz_object2.loadViewerModel()
    # viz_object2.display(q2)

