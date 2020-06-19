''' Test cpp simulator with point-mass sliding on a flat floor 
'''
import time
import consim 
import numpy as np
from numpy.linalg import norm as norm

from example_robot_data.robots_loader import loadSolo

import matplotlib.pyplot as plt 
import consim_py.utils.plot_utils as plut
import conf_solo_trot_cpp as conf
import pinocchio as pin 
import pickle

from solo_trot_common import load_ref_traj, Empty, state_diff, dt_ref, play_motion, \
    plot_integration_error_vs_ndt, compute_integration_errors

print("".center(conf.LINE_WIDTH, '#'))
print(" Test Solo Trot C++ ".center(conf.LINE_WIDTH, '#'))
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
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'euler Minv%4d'%(2**i),
        'method_name': 'euler Minv',
        'use_exp_int': 0,
        'ndt': 2**i,
        'forward_dyn_method': 1
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

PLOT_FORCES = 0
PLOT_SLIPPING = 0
PLOT_BASE_POS = 0
PLOT_INTEGRATION_ERRORS = 0
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 1

LOAD_GROUND_TRUTH_FROM_FILE = 1
SAVE_GROUND_TRUTH_TO_FILE = 1
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
refX, refU, feedBack = load_ref_traj(robot, dt)
q0, v0 = refX[0,:nq], refX[0,nq:]
N_SIMULATION = refU.shape[0]

# TEMPORARY DEBUG CODE
#N_SIMULATION = 50
#q0[2] += 1.0         # make the robot fly

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
        
    if(simu_params['use_exp_int']):
        simu = consim.build_exponential_simulator(dt, ndt, robot.model, robot.data,
                                    conf.K, conf.B, conf.mu, conf.anchor_slipping_method,
                                    compute_predicted_forces, forward_dyn_method, exp_max_mul, int_max_mul)
    else:
        simu = consim.build_euler_simulator(dt, ndt, robot.model, robot.data,
                                        conf.K, conf.B, conf.mu, forward_dyn_method)
                                        
    cpts = []
    for cf in conf.contact_frames:
        if not robot.model.existFrame(cf):
            print(("ERROR: Frame", cf, "does not exist"))
        cpts += [simu.add_contact_point(cf, robot.model.getFrameId(cf), unilateral_contacts)]

#    robot.forwardKinematics(q0)
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
                    
            xact = np.concatenate([simu.get_q(), simu.get_v()])
            diff = state_diff(robot, xact, refX[i])
            results.u[6:,i] = refU[i] + feedBack[i].dot(diff)                 
            simu.step(results.u[:,i])
#                for ci, cp in enumerate(cpts):
#                    if(cp.active and not results.active[ci,i]):
#                        print(cp.name, 'impact v', cp.v)
                
            results.q[:,i+1] = simu.get_q()
            results.v[:,i+1] = simu.get_v()
            
            for ci, cp in enumerate(cpts):
                results.f[:,ci,i+1] = cp.f
                results.p[:,ci,i+1] = cp.x
                results.p0[:,ci,i+1] = cp.x_anchor
                results.dp[:,ci,i+1] = cp.v
                results.slipping[ci,i+1] = cp.slipping
                results.active[ci,i+1] = cp.active
#                if(cp.active and not results.active[ci,i]):
#                    print(cp.name, 'impact v', cp.v)
            
            if(np.any(np.isnan(results.v[:,i+1])) or norm(results.v[:,i+1]) > 1e3):
                raise Exception("Time %.3f Velocities are too large: %.1f. Stop simulation."%(
                                t, norm(results.v[:,i+1])))
    
            if i % PRINT_N == 0:
                print("Time %.3f" % (t))                            
            t += dt
        print("Real-time factor:", t/(time.time() - time_start))
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
    data = pickle.load( open( "solo_trot_cpp.p", "rb" ) )
    
    i0, i1 = 1314, 1315
    refX = refX[i0:i1+1,:]
    refU = refU[i0:i1,:]
    feedBack = feedBack[i0:i1,:,:]
#    q0, v0 = refX[0,:nq], refX[0,nq:]
    N_SIMULATION = refU.shape[0]
    data['ground-truth-exp'].q  = data['ground-truth-exp'].q[:,i0:i1+1]
    data['ground-truth-exp'].v  = data['ground-truth-exp'].v[:,i0:i1+1]
    data['ground-truth-exp'].p0 = data['ground-truth-exp'].p0[:,:,i0:i1+1]
    data['ground-truth-exp'].slipping = data['ground-truth-exp'].slipping[:,i0:i1+1]
    data['ground-truth-euler'].q  = data['ground-truth-euler'].q[:,i0:i1+1]
    data['ground-truth-euler'].v  = data['ground-truth-euler'].v[:,i0:i1+1]
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
ndt, err_2norm_avg, err_infnorm_avg, err_infnorm_max, err_traj_2norm, err_traj_infnorm = \
    compute_integration_errors(data, robot)

# PLOT STUFF
line_styles = 10*['-o', '--o', '-.o', ':o']
tt = np.arange(0.0, (N_SIMULATION+1)*dt, dt)[:N_SIMULATION+1]
    
# PLOT INTEGRATION ERRORS
if(PLOT_INTEGRATION_ERRORS):
    plot_integration_error_vs_ndt(err_2norm_avg, ndt, 'Mean error 2-norm')
    plot_integration_error_vs_ndt(err_infnorm_avg, ndt, 'Mean error inf-norm')
    plot_integration_error_vs_ndt(err_infnorm_max, ndt, 'Max error inf-norm')    
    
if(PLOT_INTEGRATION_ERROR_TRAJECTORIES):
    (ff, ax) = plut.create_empty_figure(1)
    for (j,name) in enumerate(sorted(err_traj_2norm.keys())):
        ax.plot(tt, err_traj_2norm[name], line_styles[j], alpha=0.7, label=name)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error 2-norm')
    ax.set_yscale('log')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    
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