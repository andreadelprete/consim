''' Test accuracy-speed trade-off of cpp simulators with different robots, controllers and reference motions
'''
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

from simu_cpp_common import Empty, dt_ref, play_motion, load_solo_ref_traj, \
    plot_multi_x_vs_y_log_scale, compute_integration_errors, run_simulation

# CONTROLLERS
from linear_feedback_controller import LinearFeedbackController
from consim_py.tsid_quadruped import TsidQuadruped
from consim_py.ral2020.tsid_biped import TsidBiped

def ndprint(a, format_string ='{0:.2f}'):
    print([format_string.format(v,i) for i,v in enumerate(a)])
    
comp_times_exp    = ['exponential_simulator::step',
                     'exponential_simulator::substep',
                     'exponential_simulator::computeExpLDS',
                     'exponential_simulator::computeIntegralsXt',
                     'exponential_simulator::kinematics',
#                     'exponential_simulator::contactDetection',
#                     'exponential_simulator::contactKinematics',
                     'exponential_simulator::forwardDynamics',
                     'exponential_simulator::checkFrictionCone',
                     'exponential_simulator::resizeVectorsAndMatrices',
                     'exponential_simulator::integrateState',
                     'exponential_simulator::computeContactForces'
                     ]
comp_times_euler = ['euler_simulator::step',
                    'euler_simulator::substep']
comp_times_exp_dict = {}
comp_times_euler_dict = {}
for s in comp_times_exp:
    comp_times_exp_dict[s] = s.split('::')[-1]
for s in comp_times_euler:
    comp_times_euler_dict[s] = s.split('::')[-1]
    
plut.SAVE_FIGURES = 0
PLOT_FORCES = 0
PLOT_CONTACT_POINTS = 0
PLOT_VELOCITY_NORM = 0
PLOT_SLIPPING = 0
PLOT_BASE_POS = 0
PLOT_INTEGRATION_ERRORS = 1
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 0
PLOT_MATRIX_MULTIPLICATIONS = 0
PLOT_MATRIX_NORMS = 0

LOAD_GROUND_TRUTH_FROM_FILE = 1
SAVE_GROUND_TRUTH_TO_FILE = 1
RESET_STATE_ON_GROUND_TRUTH = 0  # do not reset state to avoid extra computation times

#TEST_NAME = 'solo-squat'
TEST_NAME = 'solo-trot'
#TEST_NAME = 'solo-jump'
#TEST_NAME = 'romeo-walk'

LINE_WIDTH = 100
print("".center(LINE_WIDTH, '#'))
print(" Test Computation Times ".center(LINE_WIDTH, '#'))
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
    assert(np.floor(dt_ref/dt)==dt_ref/dt)
elif(TEST_NAME=='romeo-walk'):
    robot_name = 'romeo'
    motionName = 'walk'
    ctrl_type = 'tsid-biped'
    dt = 0.04

# ground truth computed with time step 1/64 ms
ground_truth_dt = 1e-3/64
i_ground_truth = int(np.log2(dt / ground_truth_dt))
i_min = 0
i_max = 0

GROUND_TRUTH_EXP_SIMU_PARAMS = {
    'name': 'ground-truth %d'%(2**i_ground_truth),
    'method_name': 'ground-truth-exp',
    'use_exp_int': 1,
    'ndt': 2**i_ground_truth,
}

SIMU_PARAMS = []

# EXPONENTIAL INTEGRATOR WITH STANDARD SETTINGS
i = 0
m = 0
SIMU_PARAMS += [{
    'name': 'exp %4d mmm%2d'%(2**i,m),
    'method_name': 'exp mmm%2d'%(m),
    'use_exp_int': 1,
    'ndt': 2**i,
    'forward_dyn_method': 3,
    'max_mat_mult': m
}]

m = -1
SIMU_PARAMS += [{
    'name': 'exp %4d mmm%2d no-bal'%(2**i,m),
    'method_name': 'exp mmm%2d no-bal'%(m),
    'use_exp_int': 1,
    'ndt': 2**i,
    'forward_dyn_method': 3,
    'max_mat_mult': m,
    'use_balancing': 0
}]
        
SIMU_PARAMS += [{
            'name': 'euler',
            'method_name': 'euler',
            'use_exp_int': 0,
            'ndt': 2**4,
            'forward_dyn_method': 3
        }]
    
if(robot_name=='solo'):
    import conf_solo_cpp as conf
    robot = loadSolo(False)
elif(robot_name=='romeo'):
    import conf_romeo_cpp as conf
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
            fr = pin.Frame(frame_name, parentJointId, parentFrameId, placement, pin.FrameType.OP_FRAME)
            robot.model.addFrame(fr)
            contact_frames += [frame_name]
    conf.contact_frames = contact_frames
    robot.data = robot.model.createData()

PRINT_N = int(conf.PRINT_T/dt)
ground_truth_file_name = robot_name+"_"+motionName+str(dt)+"_cpp.p"

nq, nv = robot.nq, robot.nv

if conf.use_viewer:
    robot.initViewer(loadModel=True)
    robot.viewer.gui.createSceneWithFloor('world')
    robot.viewer.gui.setLightingMode('world/floor', 'OFF')

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


data = {}
for simu_params in SIMU_PARAMS:
    name = simu_params['name']
    print("\nStart simulation", name)
    if(simu_params['use_exp_int']):
        data[name] = run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params, 
                                    comp_times = comp_times_exp_dict)
    else:
        data[name] = run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params,
                                    comp_times=comp_times_euler_dict)
    
#    consim.stop_watch_report(3)

#    for (name,d) in data.items():
    d = data[name]
    tot = d.computation_times['step'].tot
    for (ct_name, ct) in d.computation_times.items():
#        print("%50s: avg %4.0f, percentage: %.1f %%"%(ct_name, 1e6*ct.avg, 1e2*ct.tot/tot))
        print("%25s \t& %4.0f   &  %3.0f \\%% \\\\"%(ct_name, 1e6*ct.avg, 1e2*ct.tot/tot))
        
#plt.show()