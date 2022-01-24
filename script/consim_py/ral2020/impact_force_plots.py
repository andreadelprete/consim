'''
 Test simulator behavior as contact switches happen, checking impact force generated. 
'''

import numpy as np
from numpy.linalg import norm as norm
from example_robot_data import load  

import matplotlib.pyplot as plt 
import consim_py.utils.plot_utils as plut
import pinocchio as pin 
from pinocchio.robot_wrapper import RobotWrapper
import pickle

from simu_cpp_common import dt_ref,  load_solo_ref_traj, run_simulation

# CONTROLLERS
from linear_feedback_controller import LinearFeedbackController
from consim_py.tsid_quadruped import TsidQuadruped
from consim_py.ral2020.tsid_biped import TsidBiped

def ndprint(a, format_string ='{0:.2f}'):
    print([format_string.format(v,i) for i,v in enumerate(a)])

comp_times_exp    = ['exponential_simulator::step',
                     'exponential_simulator::substep']
comp_times_euler = ['euler_simulator::step',
                    'euler_simulator::substep']
comp_times_implicit_euler = ['imp_euler_simulator::step',
                             'imp_euler_simulator::substep']
comp_times_exp_dict = {}
comp_times_euler_dict = {}
comp_times_implicit_euler_dict = {}
comp_times_rk4_dict = {}
for s in comp_times_exp:
    comp_times_exp_dict[s] = s.split('::')[-1]
for s in comp_times_euler:
    comp_times_euler_dict[s] = s.split('::')[-1]
for s in comp_times_implicit_euler:
    comp_times_implicit_euler_dict[s] = s.split('::')[-1]
                    
plut.SAVE_FIGURES = 1
PLOT_NORMAL_CONTACT_FORCE = 1
PLOT_INTEGRATION_ERRORS = 0
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 0
USE_VIEWER = 0
LOAD_DATA_FROM_FILE = 0
RESET_STATE_ON_GROUND_TRUTH = 0  # reset the state of the system on the ground truth
SAVE_DATA = 0

#TEST_NAME = 'solo-squat'
TEST_NAME = 'solo-trot'
# TEST_NAME = 'solo-jump'
# TEST_NAME = 'romeo-walk'

LINE_WIDTH = 100
print("".center(LINE_WIDTH, '#'))
print(" Test Consim C++ ".center(LINE_WIDTH, '#'))
print(TEST_NAME.center(LINE_WIDTH, '#'))
print("".center(LINE_WIDTH, '#'))

plut.FIGURE_PATH = './'+TEST_NAME+'/'
Ns = 300
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
ndt_ground_truth = int(dt / ground_truth_dt)
ndt_exp = 4
ndt_euler = 16
ndt_imp_euler = 4
maxMatMult = 0
#test = 'stiffness'
# test = 'damping'

# ndt_exp = 1
# ndt_euler = 4
# ndt_imp_euler = 1

# print("Test", test)

# if(test=='stiffness'):
#     stiffnesses    = np.logspace(3, 8, 11)
#     damping_ratios = np.linspace(0.5, 0.5, 1)
# else:
#     stiffnesses    = [1e5]
#     damping_ratios = np.linspace(0.2, 1.0, 5)

data_file_name = 'data_contact_force' #+test
data_gt_file_name = 'data_contact_force'+'_gt'

SIMU_PARAMS = []




# EULER INTEGRATOR WITH EXPLICIT INTEGRATION
SIMU_PARAMS += [{
            'name': 'euler',
            'method_name': 'Eul-exp',
            'simulator': 'euler',
            'ndt': ndt_euler,
            'forward_dyn_method': 3,
            'semi_implicit': 0,
        }]

# EULER INTEGRATOR WITH IMPLICIT INTEGRATION
SIMU_PARAMS += [{
    'name': 'eul-imp',
    'method_name': 'Eul-imp',
    'simulator': 'implicit-euler',
    'ndt': ndt_imp_euler,
    'forward_dyn_method': 3,
}]

# EXPONENTIAL INTEGRATOR WITH STANDARD SETTINGS
SIMU_PARAMS += [{
            'name': 'exp',
            'method_name': 'Expo',
            'simulator': 'exponential',
            'ndt': ndt_exp,
            'forward_dyn_method': 3,
            'max_mat_mult': maxMatMult,
        }]


if(robot_name=='solo'):
    import conf_solo_cpp as conf
    robot = load('solo12') #loadSolo(False)
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

conf.use_viewer = USE_VIEWER
PRINT_N = int(conf.PRINT_T/dt)
data_file_name    += '_'+robot_name+"_"+motionName+"_cpp.p"
data_gt_file_name += '_'+robot_name+"_"+motionName+"_cpp.p"


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


if(LOAD_DATA_FROM_FILE):
    data = pickle.load(open(data_file_name, "rb"))
    data_gt = pickle.load(open(data_gt_file_name, "rb"))
else:
    data, data_gt = {}, {}
    for simu_params in SIMU_PARAMS:
        name = simu_params['name']
        print("\nStart simulation", name)
        ndt = simu_params['ndt']
        simu_params['ndt'] = ndt_ground_truth
        data_gt[name] = run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params)
        simu_params['ndt'] = ndt
        if(simu_params['simulator']=='exponential'):
            comp_times = comp_times_exp_dict
        elif(simu_params['simulator']=='euler'):
            comp_times = comp_times_euler_dict
        elif(simu_params['simulator']=='implicit-euler'):
            comp_times = comp_times_implicit_euler_dict
        data[name] = run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params, data_gt[name], comp_times)

if(SAVE_DATA):
    pickle.dump(data, open( data_file_name, "wb" ) )
    pickle.dump(data_gt, open( data_gt_file_name, "wb" ) )
    

# PLOT STUFF
line_styles = 100*['-o', '--o', '-.o', ':o']
# line_styles = 100*['-', '--', '-.', ':']
tt = np.arange(0.0, (N+1)*dt, dt)[:N+1]
plt_color = ['blue', 'green', 'red']
labels = ['Eul-exp', 'Eul-imp', 'Expo']

if 'romeo' in TEST_NAME:
    t_start = 0 
    t_end = 95  
else: 
    t_start = 200 
    t_end = 875 

plut.FIGURE_PATH="/Users/bilal/Desktop/"
if(PLOT_NORMAL_CONTACT_FORCE):
    # (ff, ax) = plut.create_empty_figure(1)
    # for i, simu_params in enumerate(SIMU_PARAMS):
    #     name = simu_params['name']
    #     f = data_gt[name].f 
    #     if 'romeoKa' in TEST_NAME:
    #         ax.plot(tt, f[2,0,:] ,  line_styles[i], alpha=0.7, markerSize=10, label=name)
    #     else:
    #         ax.plot(tt[t_start:t_end], f[2,0,t_start:t_end],  line_styles[i], alpha=0.7, ms=7, markevery=20, label=name)
    #         # ax.plot(tt[t_start:t_end], f[2,0,t_start:t_end],  line_styles[i], alpha=0.7, label=name)
    #     ax.set_xlabel('Time [s]')
    #     ax.set_ylabel('Contact Force [N]')
    #     leg = ax.legend()
    #     if(leg): leg.get_frame().set_alpha(0.5)

    ax = plut.get_empty_figure(3)
    for i, simu_params in enumerate(SIMU_PARAMS):
        name = simu_params['name']
        f = data_gt[name].f 
        ax[i].plot(tt[t_start:t_end], f[2,0,t_start:t_end],  line_styles[i], color=plt_color[i],  alpha=0.7, ms=15, markevery=10, label=labels[i])
        # ax.plot(tt[t_start:t_end], f[2,0,t_start:t_end],  line_styles[i], alpha=0.7, label=name)
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('Normal Force [N]')
        leg = ax[i].legend()
        if(leg): leg.get_frame().set_alpha(0.5)
    
    # plut.saveFigure(TEST_NAME+"_normal_force_gnd_truth")

plt.show()