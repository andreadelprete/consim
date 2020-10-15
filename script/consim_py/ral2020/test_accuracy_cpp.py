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
                     'exponential_simulator::forwardDynamics',
                     'exponential_simulator::resizeVectorsAndMatrices']
comp_times_euler = ['euler_simulator::step',
                    'euler_simulator::substep']
                    
comp_times_exp_dict = {}
comp_times_euler_dict = {}
for s in comp_times_exp:
    comp_times_exp_dict[s] = s.split('::')[-1]
for s in comp_times_euler:
    comp_times_euler_dict[s] = s.split('::')[-1]

plut.SAVE_FIGURES = 1
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
RESET_STATE_ON_GROUND_TRUTH = 1  # reset the state of the system on the ground truth

#TEST_NAME = 'solo-squat'
#TEST_NAME = 'solo-trot'
#TEST_NAME = 'solo-jump'
TEST_NAME = 'romeo-walk'
#TEST_NAME = 'talos-walk'

LINE_WIDTH = 100
print("".center(LINE_WIDTH, '#'))
print(" Test Consim C++ ".center(LINE_WIDTH, '#'))
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
elif(TEST_NAME=='talos-walk'):
    robot_name = 'talos'
    motionName = 'walk'
    ctrl_type = 'tsid-biped'
    dt = 0.03

# ground truth computed with time step 1/64 ms
ground_truth_dt = 1e-3/64
i_ground_truth = int(np.log2(dt / ground_truth_dt))

i_min = 0
i_max = i_ground_truth - 2

GROUND_TRUTH_EXP_SIMU_PARAMS = {
    'name': 'ground-truth %d'%(2**i_ground_truth),
    'method_name': 'ground-truth-exp',
    'use_exp_int': 1,
    'ndt': 2**i_ground_truth,
}

SIMU_PARAMS = []

# EXPONENTIAL INTEGRATOR WITH STANDARD SETTINGS
for i in range(i_min, i_max):
    for m in [0, 1, 2, 3, 4, -1]:
#    for m in [-1]:
        SIMU_PARAMS += [{
            'name': 'Expo %4d mmm%2d'%(2**i,m),
            'method_name': 'Expo mmm%2d'%(m),
            'use_exp_int': 1,
            'ndt': 2**i,
            'forward_dyn_method': 3,
            'max_mat_mult': m
        }]

i_min += 0
i_max += 3
i_ground_truth = i_max+2
GROUND_TRUTH_EULER_SIMU_PARAMS = {
    'name': 'ground-truth %d'%(2**i_ground_truth),
    'method_name': 'ground-truth-euler',
    'use_exp_int': 0,
    'ndt': 2**i_ground_truth,
    'semi_implicit': 0
}

# EULER INTEGRATOR WITH EXPLICIT INTEGRATION
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'Eul-exp %4d'%(2**i),
        'method_name': 'Eul-exp',
        'use_exp_int': 0,
        'ndt': 2**i,
        'forward_dyn_method': 3,
        'semi_implicit': 0
    }]
    
# EULER INTEGRATOR WITH SEMI-IMPLICIT INTEGRATION
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'Eul-semi%4d'%(2**i),
        'method_name': 'Eul-semi',
        'use_exp_int': 0,
        'ndt': 2**i,
        'forward_dyn_method': 3,
        'semi_implicit': 1
    }]

#for i in range(i_min, i_max):
#    SIMU_PARAMS += [{
#        'name': 'euler ABA%4d'%(2**i),
#        'method_name': 'euler ABA',
#        'use_exp_int': 0,
#        'ndt': 2**i,
#        'forward_dyn_method': 2
#    }]
#
#for i in range(i_min, i_max):
#    SIMU_PARAMS += [{
#        'name': 'euler Chol%4d'%(2**i),
#        'method_name': 'euler Chol',
#        'use_exp_int': 0,
#        'ndt': 2**i,
#        'forward_dyn_method': 3
#    }]


    
unilateral_contacts = 1


if(robot_name=='solo'):
    import conf_solo_cpp as conf
    robot = loadSolo(False)
elif(robot_name=='romeo' or robot_name=='talos'):
    if(robot_name=='romeo'):
        import conf_romeo_cpp as conf
    elif(robot_name=='talos'):
        import conf_talos_cpp as conf
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


if(LOAD_GROUND_TRUTH_FROM_FILE):    
    print("\nLoad ground truth from file")
    data = pickle.load( open( ground_truth_file_name, "rb" ) )
    
#    i0, i1 = 1314, 1315
#    refX = refX[i0:i1+1,:]
#    refU = refU[i0:i1,:]
#    feedBack = feedBack[i0:i1,:,:]
##    q0, v0 = refX[0,:nq], refX[0,nq:]
#    N = refU.shape[0]
#    for key in ['ground-truth-exp', 'ground-truth-euler']:
#        data[key].q  = data[key].q[:,i0:i1+1]
#        data[key].v  = data[key].v[:,i0:i1+1]
#        data[key].p0 = data[key].p0[:,:,i0:i1+1]
#        data[key].slipping = data[key].slipping[:,i0:i1+1]
#    q0, v0 = data['ground-truth-exp'].q[:,0], data['ground-truth-exp'].v[:,0]
else:
    data = {}
    print("\nStart simulation ground truth Exponential")
    data['ground-truth-exp'] = run_simulation(conf, dt, N, robot, controller, q0, v0, GROUND_TRUTH_EXP_SIMU_PARAMS)
    
    print("\nStart simulation ground truth Euler")
    data['ground-truth-euler'] = run_simulation(conf, dt, N, robot, controller, q0, v0, GROUND_TRUTH_EULER_SIMU_PARAMS)
    if(SAVE_GROUND_TRUTH_TO_FILE):
        pickle.dump( data, open( ground_truth_file_name, "wb" ) )


for simu_params in SIMU_PARAMS:
    name = simu_params['name']
    print("\nStart simulation", name)
    print("q0", q0[2])
    if(simu_params['use_exp_int']):
        data[name] = run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params, 
                                    data['ground-truth-exp'], comp_times_exp_dict)
    else:
        data[name] = run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params, 
                            data['ground-truth-euler'], comp_times_euler_dict)

# COMPUTE INTEGRATION ERRORS:
res = compute_integration_errors(data, robot, dt)

# PLOT STUFF
line_styles = 100*['-o', '--o', '-.o', ':o']
tt = np.arange(0.0, (N+1)*dt, dt)[:N+1]
descr_str = "k_%.1f_b_%.1f"%(np.log10(conf.K[0]), np.log10(conf.B[0]))

# PLOT INTEGRATION ERRORS
if(PLOT_INTEGRATION_ERRORS):
    plot_multi_x_vs_y_log_scale(res.err_infnorm_avg, res.dt, 'Mean error inf-norm', 'Time step [s]')
    plut.saveFigure("local_err_vs_dt_"+descr_str)

    plot_multi_x_vs_y_log_scale(res.err_infnorm_avg, res.comp_time, 'Mean error inf-norm', 'Computation time per step')
    plut.saveFigure("local_err_vs_comp_time_"+descr_str)
    
    (ff,ax) = plot_multi_x_vs_y_log_scale(res.err_infnorm_avg, res.realtime_factor, 'Mean error inf-norm', 'Real-time factor')
    plut.saveFigure("local_err_vs_realtime_factor_"+descr_str)
    ax.get_legend().remove()
    plut.saveFigure("local_err_vs_realtime_factor_"+descr_str+"_nolegend")

#    plot_multi_x_vs_y_log_scale(err_2norm_avg, ndt, 'Mean error 2-norm')
#    plot_multi_x_vs_y_log_scale(err_infnorm_max, ndt, 'Max error inf-norm')

    
if(PLOT_INTEGRATION_ERROR_TRAJECTORIES):
#    (ff, ax) = plut.create_empty_figure(1)
#    for (j,name) in enumerate(sorted(err_traj_2norm.keys())):
#        ax.plot(tt, err_traj_2norm[name], line_styles[j], alpha=0.7, label=name)
#    ax.set_xlabel('Time [s]')
#    ax.set_ylabel('Error 2-norm')
#    ax.set_yscale('log')
#    leg = ax.legend()
#    if(leg): leg.get_frame().set_alpha(0.5)
    
    (ff, ax) = plut.create_empty_figure(1)
    for (j,name) in enumerate(sorted(res.err_traj_infnorm.keys())):
        if(len(res.err_traj_infnorm[name])>0):
            ax.plot(tt, res.err_traj_infnorm[name], line_styles[j], alpha=0.7, label=name)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error inf-norm')
    ax.set_yscale('log')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    plut.saveFigure("local_err_traj_"+descr_str)

if(PLOT_MATRIX_MULTIPLICATIONS):    
    plot_multi_x_vs_y_log_scale(res.mat_mult, res.ndt, 'Mat mult', logy=False)
    (ff, ax) = plut.create_empty_figure(1)
    j=0
    for (name,d) in data.items():
        if('mat_mult' in d.__dict__.keys()):
            ax.plot(tt, d.mat_mult, line_styles[j], alpha=0.7, label=name)
            j+=1
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Matrix Multiplications')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    plut.saveFigure("matrix_multiplications_"+descr_str)
    
if(PLOT_MATRIX_NORMS):    
    plot_multi_x_vs_y_log_scale(res.mat_norm, res.ndt, 'Mat norm')
    (ff, ax) = plut.create_empty_figure(1)
    j=0
    for (name,d) in data.items():
        if('mat_norm' in d.__dict__.keys()):
            ax.plot(tt, d.mat_norm, line_styles[j], alpha=0.7, label=name)
            j+=1
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Matrix Norm')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    plut.saveFigure("matrix_norms_"+descr_str)
            
# PLOT THE CONTACT FORCES OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_FORCES):        
    nc = len(conf.contact_frames)
    for (name, d) in data.items():
        ax = plut.get_empty_figure(nc)
        for i in range(nc):
#            ax[i].plot(tt, norm(d.f[0:2,i,:], axis=0) / (1e-3+d.f[2,i,:]), alpha=0.7, label=name)
            ax[i].plot(tt, d.f[2,i,:], alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('F_Z [N]')
            leg = ax[i].legend()
            if(leg): leg.get_frame().set_alpha(0.5)

if(PLOT_CONTACT_POINTS):
    nc = len(conf.contact_frames)
    for (name, d) in data.items():
        ax = plut.get_empty_figure(nc)
        for i in range(nc):
            ax[i].plot(tt, d.p[2,i,:], alpha=0.7, label=name+' p')
            ax[i].plot(tt, d.p0[2,i,:], alpha=0.7, label=name+' p0')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Z [m]')
            leg = ax[i].legend()
            if(leg): leg.get_frame().set_alpha(0.5)
            
if(PLOT_VELOCITY_NORM):
    (ff, ax) = plut.create_empty_figure(1)
    for (j,name) in enumerate(sorted(data.keys())):
        if(data[name].use_exp_int):
            ax.plot(tt, norm(data[name].v, axis=0), line_styles[j], alpha=0.7, label=name)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity 2-norm')
    ax.set_yscale('log')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    plut.saveFigure("velocity_norm_"+descr_str)
        
# PLOT THE SLIPPING FLAG OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_SLIPPING):
    nc = len(conf.contact_frames)
#    ax = plut.get_empty_figure(nc)
    for (name, d) in data.items():        
        ax = plut.get_empty_figure(nc)
        for i in range(nc):
            ax[i].plot(tt, d.slipping[i,:tt.shape[0]], alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
        ax[0].set_ylabel('Contact Slipping Flag')
        leg = ax[0].legend()
        if(leg): leg.get_frame().set_alpha(0.5)
    plut.saveFigure("slipping_flag_"+descr_str)

    ax = plut.get_empty_figure(nc)
    for (name, d) in data.items():
        for i in range(nc):
            ax[i].plot(tt, d.active[i,:tt.shape[0]], alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
    ax[0].set_ylabel('Contact Active Flag')
    leg = ax[0].legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    plut.saveFigure("active_contact_flag_"+descr_str)

       
# PLOT THE JOINT ANGLES OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_BASE_POS):
    (ff, ax) = plut.create_empty_figure(3)
    ax = ax.reshape(3)
    j = 0
    for (name, d) in data.items():
        for i in range(3):
            ax[i].plot(tt, d.q[i, :], line_styles[j], alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Base pos [m]')
        j += 1
        leg = ax[0].legend()
        if(leg): leg.get_frame().set_alpha(0.5)
        
plt.show()