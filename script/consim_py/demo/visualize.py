import time
import consim 
import numpy as np
from numpy.linalg import norm as norm

from example_robot_data.robots_loader import loadSolo, loadRomeo, getModelPath

import matplotlib.pyplot as plt 
import consim_py.utils.plot_utils as plut
import pinocchio as pin 
from pinocchio.robot_wrapper import RobotWrapper
import pickle

from consim_py.ral2020.simu_cpp_common import Empty, dt_ref, play_motion, load_solo_ref_traj, \
    plot_multi_x_vs_y_log_scale, compute_integration_errors, run_simulation

# CONTROLLERS
from consim_py.ral2020.linear_feedback_controller import LinearFeedbackController
from consim_py.tsid_quadruped import TsidQuadruped
from consim_py.ral2020.tsid_biped import TsidBiped

def ndprint(a, format_string ='{0:.2f}'):
    print([format_string.format(v,i) for i,v in enumerate(a)])

plut.SAVE_FIGURES = 1
PLOT_COM = 0
PLOT_BASE_POS = 0
PLOT_INTEGRATION_ERRORS = 1
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 1

LOAD_GROUND_TRUTH_FROM_FILE = 0
SAVE_GROUND_TRUTH_TO_FILE = 1
RESET_STATE_ON_GROUND_TRUTH = 0  # reset the state of the system on the ground truth

#TEST_NAME = 'solo-squat'
TEST_NAME = 'solo-trot'
# TEST_NAME = 'solo-jump'
# TEST_NAME = 'romeo-walk'
# TEST_NAME = 'talos-walk'

LINE_WIDTH = 100
print("".center(LINE_WIDTH, '#'))
print(" Test Stability C++ ".center(LINE_WIDTH, '#'))
print(TEST_NAME.center(LINE_WIDTH, '#'))
print("".center(LINE_WIDTH, '#'))

plut.FIGURE_PATH = './'+TEST_NAME+'/'
N = 300
dt = 0.010      # controller and simulator time step




if(TEST_NAME=='solo-squat'):
    robot_name = 'solo'
    motionName = 'squat'
    ctrl_type = 'tsid-quadruped'
    com_offset = np.array([0.0, -0.0, 0.0])
    com_amp    = np.array([0.0, 0.0, 0.05])
    com_freq   = np.array([0.0, .0, 2.0])
if(TEST_NAME=='solo-trot'):
    robot_name = 'solo'
    motionName = 'trot'
    ctrl_type = 'linear'
    dt = 0.002      # controller and simulator time step
    assert(np.floor(dt_ref/dt)==dt_ref/dt)
if(TEST_NAME=='solo-jump'):
    robot_name = 'solo'
    motionName = 'jump'
    ctrl_type = 'linear'
    dt = 0.002 
    assert(np.floor(dt_ref/dt)==dt_ref/dt)
elif(TEST_NAME=='romeo-walk'):
    robot_name = 'romeo'
    motionName = 'walk'
    ctrl_type = 'tsid-biped'
    dt = 0.04
elif(TEST_NAME=='talos-walk'):
    robot_name = 'talos'
    motionName = 'walk'
    ctrl_type = 'tsid-biped'
#    ctrl_type = 'linear-static'
    dt = 0.03
    
i_min = 0
i_max = 5

SIMU_PARAMS = []

# EXPONENTIAL INTEGRATOR WITH STANDARD SETTINGS
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'exp %4d'%(2**i),
        'method_name': 'exp',
        'use_exp_int': 1,
        'ndt': 2**i,
        'forward_dyn_method': 3
    }]

i_max = 8

# EULER INTEGRATOR WITH EXPLICIT INTEGRATION
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'euler %4d'%(2**i),
        'method_name': 'euler',
        'use_exp_int': 0,
        'ndt': 2**i,
        'forward_dyn_method': 3,
        'semi_implicit': 0
    }]

# EULER INTEGRATOR WITH SEMI-IMPLICIT INTEGRATION
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'euler semi%4d'%(2**i),
        'method_name': 'euler semi',
        'use_exp_int': 0,
        'ndt': 2**i,
        'forward_dyn_method': 1,
        'semi_implicit': 1
    }]
    

    if __name__=="__main__":
        if(robot_name=='solo'):
            import consim_py.ral2020.conf_solo_cpp as conf
            robot = loadSolo(False)
        elif(robot_name=='romeo' or robot_name=='talos'):
            if(robot_name=='romeo'):
                import consim_py.ral2020.conf_romeo_cpp as conf
            elif(robot_name=='talos'):
                import consim_py.ral2020.conf_talos_cpp as conf
            robot = RobotWrapper.BuildFromURDF(conf.urdf, [conf.modelPath], pin.JointModelFreeFlyer())
        
            contact_point = np.ones((3,4)) * conf.lz
            contact_point[0, :] = [-conf.lxn, -conf.lxn, conf.lxp, conf.lxp]
            contact_point[1, :] = [-conf.lyn, conf.lyp, -conf.lyn, conf.lyp]
            contact_frames = []
            for cf in conf.contact_frames:
                parentFrameId = robot.model.getFrameId(cf)
                parentJointId = robot.model.frames[parentFrameId].parent
                for i in range(4):
                    frame_name = cf+"_"+str(i)
                    placement = pin.XYZQUATToSE3(list(contact_point[:,i])+[0, 0, 0, 1.])
                    placement = robot.model.frames[parentFrameId].placement * placement
                    fr = pin.Frame(frame_name, parentJointId, parentFrameId, placement, pin.FrameType.OP_FRAME)
                    robot.model.addFrame(fr)
                    contact_frames += [frame_name]
            conf.contact_frames = contact_frames
            robot.data = robot.model.createData()


        PRINT_N = int(conf.PRINT_T/dt)
        ground_truth_file_name = robot_name+"_"+motionName+str(dt)+"_cpp.p"

        nq, nv = robot.nq, robot.nv

        # create feedback controller
        if(ctrl_type=='linear'):
            refX, refU, feedBack = load_solo_ref_traj(robot, dt, motionName)
            controller = LinearFeedbackController(robot, dt, refX, refU, feedBack)
            q0, v0 = controller.q0, controller.v0
            N = controller.refU.shape[0]
        elif(ctrl_type=='tsid-quadruped'):
            controller = TsidQuadruped(conf, dt, robot, com_offset, com_freq, com_amp, conf.q0, viewer=False)
            q0, v0 = conf.q0, conf.v0
        elif(ctrl_type=='tsid-biped'):
            controller = TsidBiped(conf, dt, conf.urdf, conf.modelPath, conf.srdf)
            q0, v0 = controller.q0, controller.v0
            N = controller.N+int((conf.T_pre+conf.T_post)/dt)
        elif(ctrl_type=='linear-static'):
            refX = np.array(N*[np.concatenate([conf.q0, np.zeros(conf.nv)])])
            refU = np.array(N*[np.zeros(conf.nv-6)])
            Z6 = np.zeros((conf.nv-6, 6))
            Kp = np.diagflat(conf.kp_posture)
            Kd = np.diagflat(2*np.sqrt(conf.kp_posture))
            K = -np.hstack((Z6, Kp, Z6, Kd))
            feedBack = N*[K]
            controller = LinearFeedbackController(robot, dt, refX, refU, feedBack)
            q0, v0 = controller.q0, controller.v0
            N = controller.refU.shape[0]


        if conf.use_viewer:
            robot.initViewer(loadModel=True)
            robot.viewer.gui.createSceneWithFloor('world')
            robot.viewer.gui.setLightingMode('world/floor', 'OFF')
            robot.viewer.gui.setCameraTransform(robot.viz.windowID, conf.CAMERA_TRANSFORM)
            
            robot.display(q0)
            for cf in conf.contact_frames:
                robot.viewer.gui.addSphere('world/'+cf, conf.SPHERE_RADIUS, conf.SPHERE_COLOR)
                H_cf = robot.framePlacement(q0, robot.model.getFrameId(cf))
        #        print(cf, H_cf.translation)
                robot.viewer.gui.applyConfiguration('world/'+cf, pin.SE3ToXYZQUATtuple(H_cf))
            robot.viewer.gui.refresh()

        data = {}
        for simu_params in SIMU_PARAMS:
            name = simu_params['name']
            print("\nStart simulation", name)
            if(simu_params['use_exp_int']):
                data[name] = run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params)
            else:
                data[name] = run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params)


        import consim_py.utils.visualize as visualize 
        from example_robot_data.robots_loader import getModelPath, getVisualPath

        URDF_FILENAME = "solo12.urdf"
        SRDF_FILENAME = "solo.srdf"
        SRDF_SUBPATH = "/solo_description/srdf/" + SRDF_FILENAME
        URDF_SUBPATH = "/solo_description/robots/" + URDF_FILENAME
        modelPath = getModelPath(URDF_SUBPATH)
        contact_names = []

        CameraTransform = [-0.015492438338696957,-3.359663963317871,0.5188663601875305,0.7071067690849304,
                            0.0,0.0,0.7071067690849304]

        visualizer = visualize.Visualizer(showFloor=True, cameraTF=CameraTransform)

        ground_truth_options = {'contact_names': ['HL_FOOT', 'HR_FOOT', 'FL_FOOT', 'FR_FOOT'], 
                        'robot_color': [.7,.7,.7,.4],
                        'force_color': [1., 0., 0., .75],
                        'force_radius': .002, 
                        'force_length': .025,
                        'cone_color': [0., 1., 0., .66],
                        'cone_length': .02,
                        'friction_coeff': .5
                        }

        sim_options = {'contact_names': ['HL_FOOT', 'HR_FOOT', 'FL_FOOT', 'FR_FOOT'], 
                        'robot_color': [.6,.4, 1.,.8],
                        'force_color': [0., 0., 1., .75],
                        'force_radius': .002, 
                        'force_length': .025,
                        'cone_color': [1., .65, 0., .33],
                        'cone_length': .02,
                        'friction_coeff': .5
                        }


        ground_truth_viz = visualize.ConsimVisual("truth", modelPath + URDF_SUBPATH, 
                            [getVisualPath(modelPath)], pin.JointModelFreeFlyer(), 
                            visualizer, ground_truth_options)

        ground_truth_viz.loadViewerModel()

        sim_viz = visualize.ConsimVisual("simulation", modelPath + URDF_SUBPATH, 
                            [getVisualPath(modelPath)], pin.JointModelFreeFlyer(), 
                            visualizer, sim_options)

        sim_viz.loadViewerModel()

        import os 
        sim_names = [n['name'] for n in SIMU_PARAMS] 
        data_exp = data[sim_names[0]]
        data_truth = data[sim_names[12]]

        q0 = np.zeros(data_exp.q[:,0].shape)
        q0[2] += .015


        horizon = data_truth.q.shape[1]
        saving_path = os.getcwd() +"/image_captures/"+sim_names[0] + "_vs_"+sim_names[12]
        for t in range(horizon):
            sim_viz.display(q0+data_exp.q[:,t],data_exp.f[:,:,t])
            ground_truth_viz.display(q0+data_truth.q[:,t], data_truth.f[:,:,t])
            time.sleep(1.e-3)
        #     visualizer.captureFrame(saving_path+"_{:03d}".format(t)+".png")



