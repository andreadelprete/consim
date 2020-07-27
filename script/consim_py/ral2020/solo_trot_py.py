''' Test cpp simulator with point-mass sliding on a flat floor 
'''
import time
#import consim 
from consim_py.simulator import RobotSimulator
import numpy as np
from numpy.linalg import norm as norm

from example_robot_data.robots_loader import loadSolo

import matplotlib.pyplot as plt 
import consim_py.utils.plot_utils as plut
import conf_solo_trot_py as conf
import pinocchio as pin 
import pickle 

from simu_cpp_common import load_solo_ref_traj, Empty, state_diff, dt_ref, play_motion, \
    plot_multi_x_vs_y_log_scale, compute_integration_errors

print("".center(conf.LINE_WIDTH, '#'))
print(" Test Solo Trot Python ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH, '#'))

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
    'method_name': 'ground-truth-euler',
    'use_exp_int': 0,
    'ndt': 2**i_ground_truth,
}

SIMU_PARAMS = []

# EXPONENTIAL INTEGRATOR WITH STANDARD SETTINGS
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'exp Minv%4d'%(2**i),
        'method_name': 'exp Minv',
        'use_exp_int': 1,
        'ndt': 2**i,
        'forward_dyn_method': 1
    }]

#for i in range(i_min, i_max):
#    SIMU_PARAMS += [{
#        'name': 'exp ABA%4d'%(2**i),
#        'method_name': 'exp ABA',
#        'use_exp_int': 1,
#        'ndt': 2**i,
#        'forward_dyn_method': 2
#    }]
#
#for i in range(i_min, i_max):
#    SIMU_PARAMS += [{
#        'name': 'exp Chol%4d'%(2**i),
#        'method_name': 'exp Chol',
#        'use_exp_int': 1,
#        'ndt': 2**i,
#        'forward_dyn_method': 3
#    }]
    
# EULER INTEGRATOR WITH STANDARD SETTINGS
#for i in range(i_min, i_max):
#    SIMU_PARAMS += [{
#        'name': 'euler Minv%4d'%(2**i),
#        'method_name': 'euler Minv',
#        'use_exp_int': 0,
#        'ndt': 2**i,
#        'forward_dyn_method': 1
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
    
PLOT_FORCES = 1
PLOT_FORCE_PREDICTIONS = 1
PLOT_SLIPPING = 0
PLOT_BASE_POS = 0
PLOT_INTEGRATION_ERRORS = 0
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 1

LOAD_GROUND_TRUTH_FROM_FILE = 0
SAVE_GROUND_TRUTH_TO_FILE = 0
RESET_STATE_ON_GROUND_TRUTH = 1  # reset the state of the system on the ground truth
dt     = 0.002                      # controller and simulator time step
unilateral_contacts = 1
compute_predicted_forces = False
PRINT_N = int(conf.PRINT_T/dt)

exp_max_mul = 100 
int_max_mul = 100 

robot = loadSolo(False)
nq, nv = robot.nq, robot.nv

if conf.use_viewer:
    robot.initViewer(loadModel=True)
    robot.viewer.gui.createSceneWithFloor('world')
    robot.viewer.gui.setLightingMode('world/floor', 'OFF')

assert(np.floor(dt_ref/dt)==dt_ref/dt)

# load reference trajectories 
refX, refU, feedBack = load_solo_ref_traj(robot, dt)
q0, v0 = refX[0,:nq], refX[0,nq:]
N_SIMULATION = refU.shape[0]

# TEMPORARY DEBUG CODE
N_SIMULATION = 40
#q0[2] += 1.0         # make the robot fly


def run_simulation(q0, v0, simu_params, ground_truth):
    ndt = simu_params['ndt']
        
    simu = RobotSimulator(conf, robot, pin.JointModelFreeFlyer())
    for name in conf.contact_frames:
        simu.add_contact(name, conf.contact_normal, conf.K, conf.B, conf.mu)
    simu.init(q0, v0, p0=conf.p0)
    try:
        simu.max_mat_mult = simu_params['max_mat_mult']
    except:
        simu.max_mat_mult = 100
    try:
        simu.use_second_integral = simu_params['use_second_integral']
    except:
        simu.use_second_integral = True
    try:
        simu.update_expm_N = simu_params['update_expm_N']
    except:
        simu.update_expm_N = 1
            
    t = 0.0    
    nc = len(conf.contact_frames)
    results = Empty()
    results.q = np.zeros((nq, N_SIMULATION+1))*np.nan
    results.v = np.zeros((nv, N_SIMULATION+1))*np.nan
    results.u = np.zeros((nv, N_SIMULATION+1))
    results.f = np.zeros((3, nc, N_SIMULATION+1))
    results.p = np.zeros((3, nc, N_SIMULATION+1))
    results.dp = np.zeros((3, nc, N_SIMULATION+1))
    results.p0 = np.zeros((3, nc, N_SIMULATION+1))
    results.slipping = np.zeros((nc, N_SIMULATION+1))
    results.active = np.zeros((nc, N_SIMULATION+1))
    results.f_pred_int = np.zeros((3, nc, N_SIMULATION+1))
    results.f_inner = np.zeros((3, nc, N_SIMULATION*ndt))
    results.f_avg  = np.zeros((3, nc, N_SIMULATION*ndt))
    results.f_avg2 = np.zeros((3, nc, N_SIMULATION*ndt))
    results.f_pred = np.zeros((3, nc, N_SIMULATION*ndt))
    
    results.q[:,0] = np.copy(q0)
    results.v[:,0] = np.copy(v0)
#    for ci, cp in enumerate(cpts):
#        results.f[:,ci,0] = cp.f
#        results.p[:,ci,0] = cp.x
#        results.p0[:,ci,0] = cp.x_anchor
#        results.dp[:,ci,0] = cp.v
#        results.slipping[ci,0] = cp.slipping
#        results.active[ci,0] = cp.active
#    print('K*p', conf.K[2]*results.p[2,:,0].squeeze())
    
    try:
        time_start = time.time()
        for i in range(0, N_SIMULATION):
            if(RESET_STATE_ON_GROUND_TRUTH and ground_truth):
                # first reset to ensure active contact points are correctly marked because otherwise the second
                # time I reset the state the anchor points could be overwritten
                simu.init(ground_truth.q[:,i], ground_truth.v[:,i], reset_anchor_points=True)
#                print("Reset anchor points to:\n", ground_truth.p0[:,:,i])
                simu.init(ground_truth.q[:,i], ground_truth.v[:,i], p0=ground_truth.p0[:,:,i].T.reshape(3*nc))
                
            xact = np.concatenate([simu.q, simu.v])
            diff = state_diff(robot, xact, refX[i])
            results.u[6:,i] = refU[i] + feedBack[i].dot(diff)                 

            simu.simulate(results.u[6:,i], dt, ndt, simu_params['use_exp_int'])
            results.q[:,i+1] = simu.q
            results.v[:,i+1] = simu.v
            
            for ci, cp in enumerate(simu.contacts):
                results.f[:,ci,i+1] = cp.f
                results.f_inner[:,ci,i*ndt:(i+1)*ndt] = cp.f_inner
                results.f_pred[ :,ci,i*ndt:(i+1)*ndt] = cp.f_pred
                results.f_avg[  :,ci,i*ndt:(i+1)*ndt] = cp.f_avg
                results.f_avg2[ :,ci,i*ndt:(i+1)*ndt] = cp.f_avg2
            
            if(np.any(np.isnan(results.v[:,i+1])) or norm(results.v[:,i+1]) > 1e3):
                raise Exception("Time %.3f Velocities are too large: %.1f. Stop simulation."%(
                                t, norm(results.v[:,i+1])))
    
            if i % PRINT_N == 0:
                print("Time %.3f" % (t))                            
            t += dt
        print("Real-time factor:", t/(time.time() - time_start))
    except Exception as e:
        print(e)
#        raise e

    if conf.use_viewer:
        play_motion(robot, results.q, dt)
                    
    for key in simu_params.keys():
        results.__dict__[key] = simu_params[key]

    return results

if(LOAD_GROUND_TRUTH_FROM_FILE):    
    print("\nLoad ground truth from file")
    data = pickle.load( open( "solo_trot_cpp.p", "rb" ) )
    
    i0, i1 = 1314, 1315
    refX = refX[i0:i1+1,:]
    refU = refU[i0:i1,:]
    feedBack = feedBack[i0:i1,:,:]
#    q0, v0 = refX[0,:nq], refX[0,nq:]
    N_SIMULATION = refU.shape[0]
    data['ground-truth-exp'].q  = data['ground-truth-exp'].q[:,i0:i1+1]
    data['ground-truth-exp'].v  = data['ground-truth-exp'].v[:,i0:i1+1]
    data['ground-truth-exp'].f  = data['ground-truth-exp'].f[:,:,i0:i1+1]
    data['ground-truth-exp'].p0 = data['ground-truth-exp'].p0[:,:,i0:i1+1]
    data['ground-truth-exp'].slipping = data['ground-truth-exp'].slipping[:,i0:i1+1]
    data['ground-truth-euler'].q  = data['ground-truth-euler'].q[:,i0:i1+1]
    data['ground-truth-euler'].v  = data['ground-truth-euler'].v[:,i0:i1+1]
    data['ground-truth-euler'].f  = data['ground-truth-euler'].f[:,:,i0:i1+1]
    data['ground-truth-euler'].p0 = data['ground-truth-euler'].p0[:,:,i0:i1+1]
    data['ground-truth-euler'].slipping = data['ground-truth-euler'].slipping[:,i0:i1+1]
    q0, v0 = data['ground-truth-exp'].q[:,0], data['ground-truth-exp'].v[:,0]
else:
    data = {}
    print("\nStart simulation ground truth")
    data['ground-truth-exp'] = run_simulation(q0, v0, GROUND_TRUTH_EXP_SIMU_PARAMS, None)
    data['ground-truth-euler'] = run_simulation(q0, v0, GROUND_TRUTH_EULER_SIMU_PARAMS, None)
    if(SAVE_GROUND_TRUTH_TO_FILE):
        pickle.dump( data, open( "solo_trot_cpp.p", "wb" ) )
        
 
for simu_params in SIMU_PARAMS:
    name = simu_params['name']
    print("\nStart simulation", name)
    if(simu_params['use_exp_int']):
        data[name] = run_simulation(q0, v0, simu_params, data['ground-truth-exp'])
    else:
        data[name] = run_simulation(q0, v0, simu_params, data['ground-truth-euler'])

# COMPUTE INTEGRATION ERRORS:
res = compute_integration_errors(data, robot, dt)

# PLOT STUFF
line_styles = 10*['-o', '--o', '-.o', ':o']
tt = np.arange(0.0, (N_SIMULATION+1)*dt, dt)[:N_SIMULATION+1]

# PLOT INTEGRATION ERRORS
if(PLOT_INTEGRATION_ERRORS):
    plot_multi_x_vs_y_log_scale(res.err_infnorm_avg, res.ndt, 'Mean error inf-norm')
    plot_multi_x_vs_y_log_scale(res.err_infnorm_max, res.ndt, 'Max error inf-norm')    
    
if(PLOT_INTEGRATION_ERROR_TRAJECTORIES):
    (ff, ax) = plut.create_empty_figure(1)
    for (j,name) in enumerate(sorted(res.err_traj_infnorm.keys())):
        ax.plot(tt, res.err_traj_infnorm[name], line_styles[j], alpha=0.7, label=name)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error inf-norm')
    ax.set_yscale('log')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    
        
# FOR EACH INTEGRATION METHOD PLOT THE FORCE PREDICTIONS
if(PLOT_FORCE_PREDICTIONS):
    T = N_SIMULATION*dt
    tt = np.arange(0.0, (N_SIMULATION+1)*dt, dt)[:N_SIMULATION+1]
    
#    for (name,d) in data.items():
#        (ff, ax) = plut.create_empty_figure(2,2)
#        ax = ax.reshape(4)       
#        for i in range(4):
#            ax[i].plot(tt, d.f[2,i,:], ' o', markersize=8, label=name)
##            ax[i].plot(tt, d.f_pred_int[2,i,:], ' s', markersize=8, label=name+' pred int')
#            if('ground' not in name):
#                tt_log = np.arange(d.f_pred.shape[2]) * T / d.f_pred.shape[2]
#                ax[i].plot(tt_log, d.f_pred[2,i,:], 'r v', markersize=6, label=name+' pred ')
#                ax[i].plot(tt_log, d.f_avg[2,i,:], 'g s', markersize=6, label=name+' avg ')
#                ax[i].plot(tt_log, d.f_avg2[2,i,:], 'y s', markersize=6, label=name+' avg2 ')
#                ax[i].plot(tt_log, d.f_inner[2,i,:], 'b x', markersize=6, label=name+' real ')
#            ax[i].set_xlabel('Time [s]')
#            ax[i].set_ylabel('Force Z [N]')
#        leg = ax[-1].legend()
#        if(leg): leg.get_frame().set_alpha(0.5)
#        
#        if('ground' not in name):
#            # force prediction error of Euler, i.e. assuming force remains contact during time step
#            ndt = int(d.f_inner.shape[2] / (d.f.shape[2]-1))
#            
#            # force prediction error of matrix exponential 
#            f_pred_err = d.f_pred - d.f_inner
#            print(name, 'Force pred err max exp:', np.sum(np.abs(f_pred_err))/(f_pred_err.shape[0]*f_pred_err.shape[2]))
       
    (ff, ax) = plut.create_empty_figure(2,2)
    ax = ax.reshape(4)
    for (name,d) in data.items():
        if('ground' in name):
            continue
        tt_log = np.arange(d.f_pred.shape[2]) * T / d.f_pred.shape[2]
        for i in range(4):
            if('ground' in name):
                ax[i].plot(tt, d.f[2,i,:], 'b x', markersize=6, label=name)
                ax[i].plot(tt_log, d.f_inner[2,i,:], 'b x', markersize=6, label=name)
            elif(d.use_exp_int):
                ax[i].plot(tt_log, d.f_avg[2,i,:], 's', markersize=6, label=name+' avg ')
                ax[i].plot(tt_log, d.f_avg2[2,i,:], 's', markersize=6, label=name+' avg2 ')
            else:
                ax[i].plot(tt_log, d.f_inner[2,i,:], 'o', markersize=6, label=name)
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('Force Z [N]')
    leg = ax[-1].legend()
    if(leg): leg.get_frame().set_alpha(0.5) 
       
# PLOT THE CONTACT FORCES OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_FORCES):        
    nc = len(conf.contact_frames)
    for (name, d) in data.items():
        (ff, ax) = plut.create_empty_figure(nc, 1)
        for i in range(nc):
            ax[i].plot(tt, norm(d.f[0:2,i,:], axis=0) / (1e-3+d.f[2,i,:]), alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Force X/Z [N]')
            leg = ax[i].legend()
            if(leg): leg.get_frame().set_alpha(0.5)
            
# PLOT THE SLIPPING FLAG OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_SLIPPING):
    nc = len(conf.contact_frames)
    (ff, ax) = plut.create_empty_figure(nc, 1)
    for (name, d) in data.items():        
        for i in range(nc):
            ax[i].plot(tt, d.slipping[i,:], alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
    ax[0].set_ylabel('Contact Slipping Flag')
    leg = ax[0].legend()
    if(leg): leg.get_frame().set_alpha(0.5)

    (ff, ax) = plut.create_empty_figure(nc, 1)
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