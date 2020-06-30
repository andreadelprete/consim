''' Test cpp simulator with different robots, controllers and reference motions
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

from simu_cpp_common import Empty, dt_ref, play_motion, load_solo_ref_traj, plot_integration_error_vs_ndt, compute_integration_errors

# CONTROLLERS
from linear_feedback_controller import LinearFeedbackController
from consim_py.tsid_quadruped import TsidQuadruped
from consim_py.ral2020.tsid_biped import TsidBiped

def ndprint(a, format_string ='{0:.2f}'):
    print([format_string.format(v,i) for i,v in enumerate(a)])
    
LINE_WIDTH = 100
print("".center(LINE_WIDTH, '#'))
print(" Test Consim C++ ".center(LINE_WIDTH, '#'))
print("".center(LINE_WIDTH, '#'))

# parameters of the simulation to be tested
i_min = 5
i_max = 6
i_ground_truth = i_max+2

GROUND_TRUTH_EXP_SIMU_PARAMS = {
    'name': 'ground-truth %d'%(2**i_ground_truth),
    'method_name': 'ground-truth-exp',
    'use_exp_int': 1,
    'ndt': 2**i_ground_truth,
}

GROUND_TRUTH_EULER_SIMU_PARAMS = {
    'name': 'ground-truth %d'%(2**i_ground_truth),
    'method_name': 'ground-truth-euler-semi',
    'use_exp_int': 0,
    'ndt': 2**i_ground_truth,
    'semi_implicit': 0
}

SIMU_PARAMS = []

# EXPONENTIAL INTEGRATOR WITH STANDARD SETTINGS
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'exp %4d'%(2**i),
        'method_name': 'exp',
        'use_exp_int': 1,
        'ndt': 2**i,
        'forward_dyn_method': 1
    }]
 
# EULER INTEGRATOR WITH EXPLICIT INTEGRATION
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'euler %4d'%(2**i),
        'method_name': 'euler',
        'use_exp_int': 0,
        'ndt': 2**i,
        'forward_dyn_method': 1,
        'semi_implicit': 0
    }]
    
# EULER INTEGRATOR WITH SEMI-IMPLICIT INTEGRATION
#for i in range(i_min, i_max):
#    SIMU_PARAMS += [{
#        'name': 'euler semi%4d'%(2**i),
#        'method_name': 'euler semi',
#        'use_exp_int': 0,
#        'ndt': 2**i,
#        'forward_dyn_method': 1,
#        'semi_implicit': 1
#    }]

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

PLOT_FORCES = 0
PLOT_CONTACT_POINTS = 0
PLOT_VELOCITY_NORM = 0
PLOT_SLIPPING = 0
PLOT_BASE_POS = 0
PLOT_INTEGRATION_ERRORS = 0
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 0

LOAD_GROUND_TRUTH_FROM_FILE = 1
SAVE_GROUND_TRUTH_TO_FILE = 1
RESET_STATE_ON_GROUND_TRUTH = 1  # reset the state of the system on the ground truth

TEST_NAME = 'solo-squat'
#TEST_NAME = 'solo-trot'
#TEST_NAME = 'solo-jump'
#TEST_NAME = 'romeo-walk'

N_SIMULATION = 300
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
    
unilateral_contacts = 1
compute_predicted_forces = False

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

exp_max_mul = 100 
int_max_mul = 100 

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
    N_SIMULATION = controller.refU.shape[0]
elif(ctrl_type=='tsid-quadruped'):
    controller = TsidQuadruped(conf, dt, robot, com_offset, com_freq, com_amp, conf.q0, viewer=False)
    q0, v0 = conf.q0, conf.v0
elif(ctrl_type=='tsid-biped'):
    controller = TsidBiped(conf, dt, conf.urdf, conf.modelPath, conf.srdf)
    q0, v0 = controller.q0, controller.v0
    N_SIMULATION = controller.N+int((conf.T_pre+conf.T_post)/dt)
    

def run_simulation(q0, v0, simu_params, ground_truth):        
    ndt = simu_params['ndt']
    try:
        forward_dyn_method = simu_params['forward_dyn_method']
    except:
        # forward_dyn_method Options 
        #  1: pinocchio.Minverse()
        #  2: pinocchio.aba()
        #  3: Cholesky factorization 
        forward_dyn_method = 1
    try:
        semi_implicit = simu_params['semi_implicit']
    except:
        semi_implicit = 0
        
    if(simu_params['use_exp_int']):
        simu = consim.build_exponential_simulator(dt, ndt, robot.model, robot.data,
                                    conf.K, conf.B, conf.mu, conf.anchor_slipping_method,
                                    compute_predicted_forces, forward_dyn_method, semi_implicit,
                                    exp_max_mul, int_max_mul)
    else:
        simu = consim.build_euler_simulator(dt, ndt, robot.model, robot.data,
                                        conf.K, conf.B, conf.mu, forward_dyn_method, semi_implicit)
                                        
    cpts = []
    for cf in conf.contact_frames:
        if not robot.model.existFrame(cf):
            print(("ERROR: Frame", cf, "does not exist"))
        cpts += [simu.add_contact_point(cf, robot.model.getFrameId(cf), unilateral_contacts)]
        
    simu.reset_state(q0, v0, True)
            
    t = 0.0    
    nc = len(conf.contact_frames)
    results = Empty()
    results.q = np.zeros((nq, N_SIMULATION+1))
    results.v = np.zeros((nv, N_SIMULATION+1))
    results.u = np.zeros((nv, N_SIMULATION+1))
    results.f = np.zeros((3, nc, N_SIMULATION+1))
    results.p = np.zeros((3, nc, N_SIMULATION+1))
    results.dp = np.zeros((3, nc, N_SIMULATION+1))
    results.p0 = np.zeros((3, nc, N_SIMULATION+1))
    results.slipping = np.zeros((nc, N_SIMULATION+1))
    results.active = np.zeros((nc, N_SIMULATION+1))
    results.computation_times = {'inner-step': Empty()}
    
    results.q[:,0] = np.copy(q0)
    results.v[:,0] = np.copy(v0)
    for ci, cp in enumerate(cpts):
        results.f[:,ci,0] = cp.f
        results.p[:,ci,0] = cp.x
        results.p0[:,ci,0] = cp.x_anchor
        results.dp[:,ci,0] = cp.v
        results.slipping[ci,0] = cp.slipping
        results.active[ci,0] = cp.active
#    print('K*p', conf.K[2]*results.p[2,:,0].squeeze())
    
    try:
        controller.reset(q0, v0, conf.T_pre)
        consim.stop_watch_reset_all()
        time_start = time.time()
        for i in range(0, N_SIMULATION):
            if(RESET_STATE_ON_GROUND_TRUTH and ground_truth):                
                # first reset to ensure active contact points are correctly marked because otherwise the second
                # time I reset the state the anchor points could be overwritten
                reset_anchor_points = True
                simu.reset_state(ground_truth.q[:,i], ground_truth.v[:,i], reset_anchor_points)
                # then reset anchor points
                for ci, cp in enumerate(cpts):
                    cp.resetAnchorPoint(ground_truth.p0[:,ci,i], bool(ground_truth.slipping[ci,i]))
                # then reset once againt to compute updated contact forces, but without touching anchor points
                reset_anchor_points = False
                simu.reset_state(ground_truth.q[:,i], ground_truth.v[:,i], reset_anchor_points)
                    
            results.u[6:,i] = controller.compute_control(simu.get_q(), simu.get_v())
            simu.step(results.u[:,i])
                
            results.q[:,i+1] = simu.get_q()
            results.v[:,i+1] = simu.get_v()
            
            for ci, cp in enumerate(cpts):
                results.f[:,ci,i+1] = cp.f
                results.p[:,ci,i+1] = cp.x
                results.p0[:,ci,i+1] = cp.x_anchor
                results.dp[:,ci,i+1] = cp.v
                results.slipping[ci,i+1] = cp.slipping
                results.active[ci,i+1] = cp.active
#                if(cp.active != results.active[ci,i]):
#                    print("%.3f"%t, cp.name, 'changed contact state to ', cp.active, cp.x)
            
            if(np.any(np.isnan(results.v[:,i+1])) or norm(results.v[:,i+1]) > 1e6):
                raise Exception("Time %.3f Velocities are too large: %.1f. Stop simulation."%(
                                t, norm(results.v[:,i+1])))
    
            if i % PRINT_N == 0:
                print("Time %.3f" % (t))  
            t += dt
        
        print("Real-time factor:", t/(time.time() - time_start))
        consim.stop_watch_report(3)
        if(simu_params['use_exp_int']):
            results.computation_times['inner-step'].avg = \
                consim.stop_watch_get_average_time("exponential_simulator::substep")
        else:
            results.computation_times['inner-step'].avg = \
                consim.stop_watch_get_average_time("euler_simulator::substep")
        print("Inner step: %.1f us"%(results.computation_times['inner-step'].avg*1e6))
    except Exception as e:
#        raise e
        print("Exception while running simulation", e)

    if conf.use_viewer:
        play_motion(robot, results.q, dt)
                    
    for key in simu_params.keys():
        results.__dict__[key] = simu_params[key]

    return results

if(LOAD_GROUND_TRUTH_FROM_FILE):    
    print("\nLoad ground truth from file")
    data = pickle.load( open( ground_truth_file_name, "rb" ) )
    
#    i0, i1 = 1314, 1315
#    refX = refX[i0:i1+1,:]
#    refU = refU[i0:i1,:]
#    feedBack = feedBack[i0:i1,:,:]
##    q0, v0 = refX[0,:nq], refX[0,nq:]
#    N_SIMULATION = refU.shape[0]
#    for key in ['ground-truth-exp', 'ground-truth-euler']:
#        data[key].q  = data[key].q[:,i0:i1+1]
#        data[key].v  = data[key].v[:,i0:i1+1]
#        data[key].p0 = data[key].p0[:,:,i0:i1+1]
#        data[key].slipping = data[key].slipping[:,i0:i1+1]
#    q0, v0 = data['ground-truth-exp'].q[:,0], data['ground-truth-exp'].v[:,0]
else:
    data = {}
    print("\nStart simulation ground truth")
    data['ground-truth-exp'] = run_simulation(q0, v0, GROUND_TRUTH_EXP_SIMU_PARAMS, None)
    data['ground-truth-euler'] = run_simulation(q0, v0, GROUND_TRUTH_EULER_SIMU_PARAMS, None)
    if(SAVE_GROUND_TRUTH_TO_FILE):
        pickle.dump( data, open( ground_truth_file_name, "wb" ) )


for simu_params in SIMU_PARAMS:
    name = simu_params['name']
    print("\nStart simulation", name)
    if(simu_params['use_exp_int']):
        data[name] = run_simulation(q0, v0, simu_params, data['ground-truth-exp'])
    else:
        data[name] = run_simulation(q0, v0, simu_params, data['ground-truth-euler'])

# COMPUTE INTEGRATION ERRORS:
ndt, err_2norm_avg, err_infnorm_avg, err_infnorm_max, err_traj_2norm, err_traj_infnorm = \
    compute_integration_errors(data, robot)

# PLOT STUFF
line_styles = 10*['-o', '--o', '-.o', ':o']
tt = np.arange(0.0, (N_SIMULATION+1)*dt, dt)[:N_SIMULATION+1]
    
# PLOT INTEGRATION ERRORS
if(PLOT_INTEGRATION_ERRORS):
#    plot_integration_error_vs_ndt(err_2norm_avg, ndt, 'Mean error 2-norm')
    plot_integration_error_vs_ndt(err_infnorm_avg, ndt, 'Mean error inf-norm')
    plot_integration_error_vs_ndt(err_infnorm_max, ndt, 'Max error inf-norm')    
    
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
    for (j,name) in enumerate(sorted(err_traj_infnorm.keys())):
        ax.plot(tt, err_traj_infnorm[name], line_styles[j], alpha=0.7, label=name)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error inf-norm')
    ax.set_yscale('log')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    
            
# PLOT THE CONTACT FORCES OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_FORCES):        
    nc = len(conf.contact_frames)
    for (name, d) in data.items():
        if(nc<5):
            (ff, ax) = plut.create_empty_figure(nc, 1)
        else:
            (ff, ax) = plut.create_empty_figure(int(nc/2), 2)
            ax = ax.reshape(ax.shape[0]*ax.shape[1])
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
        if(nc<5):
            (ff, ax) = plut.create_empty_figure(nc, 1)
        else:
            (ff, ax) = plut.create_empty_figure(int(nc/2), 2)
            ax = ax.reshape(ax.shape[0]*ax.shape[1])
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
        
# PLOT THE SLIPPING FLAG OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_SLIPPING):
    nc = len(conf.contact_frames)
    if(nc<5):
        (ff, ax) = plut.create_empty_figure(nc, 1)
    else:
        (ff, ax) = plut.create_empty_figure(int(nc/2), 2)
        ax = ax.reshape(ax.shape[0]*ax.shape[1])
    for (name, d) in data.items():        
        for i in range(nc):
            ax[i].plot(tt, d.slipping[i,:], alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
    ax[0].set_ylabel('Contact Slipping Flag')
    leg = ax[0].legend()
    if(leg): leg.get_frame().set_alpha(0.5)

    if(nc<5):
        (ff, ax) = plut.create_empty_figure(nc, 1)
    else:
        (ff, ax) = plut.create_empty_figure(int(nc/2), 2)
        ax = ax.reshape(ax.shape[0]*ax.shape[1])
    for (name, d) in data.items():
        for i in range(nc):
            ax[i].plot(tt, d.active[i,:], alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
    ax[0].set_ylabel('Contact Active Flag')
    leg = ax[0].legend()
    if(leg): leg.get_frame().set_alpha(0.5)

       
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