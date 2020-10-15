''' Test simulator behavior as contact stiffness and damping change
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
                     'exponential_simulator::substep']
comp_times_euler = ['euler_simulator::step',
                    'euler_simulator::substep']
comp_times_exp_dict = {}
comp_times_euler_dict = {}
for s in comp_times_exp:
    comp_times_exp_dict[s] = s.split('::')[-1]
for s in comp_times_euler:
    comp_times_euler_dict[s] = s.split('::')[-1]
                    
plut.SAVE_FIGURES = 1
PLOT_INTEGRATION_ERRORS = 1
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 0
USE_VIEWER = 0
LOAD_DATA_FROM_FILE = 0
RESET_STATE_ON_GROUND_TRUTH = 1  # reset the state of the system on the ground truth
SAVE_DATA = 1

#TEST_NAME = 'solo-squat'
TEST_NAME = 'solo-trot'
#TEST_NAME = 'solo-jump'
#TEST_NAME = 'romeo-walk'

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
maxMatMult = 0
test = 'stiffness'
#test = 'damping'

if(test=='stiffness'):
    stiffnesses    = np.logspace(3, 8, 11)
    damping_ratios = np.linspace(0.5, 0.5, 1)
else:
    stiffnesses    = [1e5]
    damping_ratios = np.linspace(0.2, 1.0, 5)

data_file_name = 'data_test_'+test
data_gt_file_name = 'data_test_'+test+'_gt'

SIMU_PARAMS = []

# EXPONENTIAL INTEGRATOR WITH STANDARD SETTINGS
for ik in stiffnesses:
    for dr in damping_ratios:
        SIMU_PARAMS += [{
            'name': 'exp %3.1f-%3.1f'%(np.log10(ik),dr),
            'method_name': 'Expo',
            'use_exp_int': 1,
            'ndt': ndt_exp,
            'forward_dyn_method': 3,
            'max_mat_mult': maxMatMult,
            'K': ik,
            'damping_ratio': dr
        }]

# EULER INTEGRATOR WITH EXPLICIT INTEGRATION
for ik in stiffnesses:
    for dr in damping_ratios:
        SIMU_PARAMS += [{
            'name': 'euler %3.1f-%3.1f'%(np.log10(ik),dr),
            'method_name': 'Eul-exp',
            'use_exp_int': 0,
            'ndt': ndt_euler,
            'forward_dyn_method': 3,
            'semi_implicit': 0,
            'K': ik,
            'damping_ratio': dr
        }]
        
# EULER INTEGRATOR WITH SEMI-IMPLICIT INTEGRATION
for ik in stiffnesses:
    for dr in damping_ratios:
        SIMU_PARAMS += [{
            'name': 'eul-semi %3.1f-%3.1f'%(np.log10(ik),dr),
            'method_name': 'Eul-semi',
            'use_exp_int': 0,
            'ndt': ndt_euler,
            'forward_dyn_method': 3,
            'semi_implicit': 1,
            'K': ik,
            'damping_ratio': dr
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
        conf.K = simu_params['K']*np.ones(3)
        dr = simu_params['damping_ratio']
        conf.B = dr*2*np.sqrt(conf.K[0])*np.ones(3)
        
        ndt = simu_params['ndt']
        simu_params['ndt'] = ndt_ground_truth
        data_gt[name] = run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params)
        simu_params['ndt'] = ndt
        if(simu_params['use_exp_int']==1):
            comp_times = comp_times_exp_dict
        else:
            comp_times = comp_times_euler_dict
        data[name] = run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params, data_gt[name], comp_times)

if(SAVE_DATA):
    pickle.dump(data, open( data_file_name, "wb" ) )
    pickle.dump(data_gt, open( data_gt_file_name, "wb" ) )
    
def compute_integration_errors_vs_k_b(data, data_gt, robot, dt):
    print('\n')
    res = Empty()
    res.k, res.damping_ratio, res.comp_time, res.realtime_factor = {}, {}, {}, {}
    res.err_infnorm_avg, res.err_infnorm_max = {}, {}
    res.err_traj_infnorm = {}

    for name in sorted(data.keys()):
        if('ground-truth' in name): continue
        d = data[name]
        data_ground_truth = data_gt[name]
        
        err_vec = np.empty((2*robot.nv, d.q.shape[1]))
        err_per_time_inf = np.empty(d.q.shape[1])
        err_inf = 0.0
        for i in range(d.q.shape[1]):
            err_vec[:robot.nv,i] = pin.difference(robot.model, d.q[:,i], data_ground_truth.q[:,i])
            err_vec[robot.nv:,i] = d.v[:,i] - data_ground_truth.v[:,i]
            err_per_time_inf[i] = norm(err_vec[:,i], np.inf)
            err_inf += err_per_time_inf[i]
        err_inf /= d.q.shape[1]
        err_peak = np.max(err_per_time_inf)
        if(d.method_name not in res.err_infnorm_avg):
            for k in res.__dict__.keys():
                res.__dict__[k][d.method_name] = []

        res.err_infnorm_avg[d.method_name] += [err_inf]
        res.err_infnorm_max[d.method_name] += [err_peak]
        res.err_traj_infnorm[name] = err_per_time_inf
        res.k[d.method_name] += [d.K]
        res.damping_ratio[d.method_name] += [d.damping_ratio]
        comp_time = d.computation_times['substep'].avg * d.ndt
        res.comp_time[d.method_name] += [comp_time]
        res.realtime_factor[d.method_name] += [dt/comp_time]
        try:
            print(name, 'Log error inf-norm: %.2f'%np.log10(err_inf), 'RT factor', int(dt/comp_time))
        except:
            pass
    return res
    
# COMPUTE INTEGRATION ERRORS:
res = compute_integration_errors_vs_k_b(data, data_gt, robot, dt)


# PLOT STUFF
line_styles = 100*['-o', '--o', '-.o', ':o']
tt = np.arange(0.0, (N+1)*dt, dt)[:N+1]

# PLOT INTEGRATION ERRORS
if(PLOT_INTEGRATION_ERRORS):
    if(test=='stiffness'):
        plot_multi_x_vs_y_log_scale(res.err_infnorm_avg, res.k, 'Mean error inf-norm', 'Contact stiffness [N/m]')
        plut.saveFigure("local_err_vs_stiffness_with_fixed_damping_ratio")
    else:
        plot_multi_x_vs_y_log_scale(res.err_infnorm_avg, res.damping_ratio, 'Mean error inf-norm', 
                                    'Contact damping ratio', logx=False)
        plut.saveFigure("local_err_vs_damping_with_fixed_stiffness")
    
if(PLOT_INTEGRATION_ERROR_TRAJECTORIES):    
    for (j,name) in enumerate(sorted(res.err_traj_infnorm.keys())):
        if(len(res.err_traj_infnorm[name])>0):
            (ff, ax) = plut.create_empty_figure(1)
            ax.plot(tt, res.err_traj_infnorm[name], line_styles[j], alpha=0.7, label=name)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Error inf-norm')
            ax.set_yscale('log')
            leg = ax.legend()
            if(leg): leg.get_frame().set_alpha(0.5)
#    plut.saveFigure("local_err_traj_"+descr_str)
                
plt.show()