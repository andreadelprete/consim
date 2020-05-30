''' Test python simulator with point-mass sliding on a flat floor 
'''
import time
from simulator import RobotSimulator
import numpy as np
from numpy import nan
from numpy.linalg import norm as norm

import pinocchio as pin 
from pinocchio.robot_wrapper import RobotWrapper
import os, sys
import matplotlib.pyplot as plt 
import consim_py.utils.plot_utils as plut
import conf_sliding_mass_py as conf

class Empty:
    pass

print("".center(conf.LINE_WIDTH, '#'))
print(" Test Sliding Mass ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH, '#'))

# parameters of the simulation to be tested
i_min = 1
i_max = i_min+9
i_ground_truth = i_max

GROUND_TRUTH_SIMU_PARAMS = {
    'name': 'ground-truth %d'%(2**i_ground_truth),
    'method_name': 'ground-truth',
    'use_exp_int': 1,
    'ndt': 2**i_ground_truth,
}

SIMU_PARAMS = []

# EXPONENTIAL INTEGRATOR WITH STANDARD SETTINGS
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'exp%4d'%(2**i),
        'method_name': 'exp',
        'use_exp_int': 1,
        'ndt': 2**i,
    }]
    
# EULER INTEGRATOR WITH STANDARD SETTINGS
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'euler%4d'%(2**i),
        'method_name': 'euler',
        'use_exp_int': 0,
        'ndt': 2**i,
    }]
    
PLOT_FORCES = 0
PLOT_BASE_POS = 0
PLOT_FORCE_PREDICTIONS = 0
PLOT_FORCE_INTEGRALS = 0
PLOT_INTEGRATION_ERRORS = 1
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 1
PLOT_CONTACT_POINT_PREDICTION = 0
PLOT_ANCHOR_POINTS = 0

dt = 0.005                      # controller time step
T = 0.5

N_SIMULATION = int(T/dt)        # number of time steps simulated
PRINT_N = int(conf.PRINT_T/dt)

robot = RobotWrapper.BuildFromURDF(conf.urdf_path, [conf.mesh_path], pin.JointModelFreeFlyer()) 
nq, nv = robot.nq, robot.nv
simu = RobotSimulator(conf, robot)
q0, v0 = np.copy(simu.q), np.copy(simu.v)

for name in conf.contact_frames:
    simu.add_contact(name, conf.contact_normal, conf.K, conf.B, conf.mu)

""" apply a force that pushes the point mass down and violates the friction cone 
say fN = - 10 N
at v = 0 
the penetration will be 10 = 1.e5*delta -> delta = 1.e-4 not so bad 
then the friction cone bound at mu = 0.3 is 3 N 
given mass  = 1 kg 
applying 4 Nm will result in a = 1 m / s^2 
we can compute the trajectory of the anchor point 
since there is no rolling, tipping and other contact states 
v(t) = v(0) + t*a 
x(t) = x(0) + t*v(0) + .5 * t^2 * a (assuming a is constant ) 
"""
tau_0 = np.array([2., 0., -0.19, 0., 0., 0.])
tau_A = np.array([2., 0.,  0.,   0., 0., 0.])
tau_f = np.array([2., 0.,  0.,   0., 0., 0.])


def run_simulation(q, v, simu_params):
    simu = RobotSimulator(conf, robot)
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
    try:
        simu.fwd_dyn_method = simu_params['fwd_dyn_method']
    except:
        simu.fwd_dyn_method = 'Cholesky'
        
    t = 0.0
    ndt = simu_params['ndt']
    time_start = time.time()
    q = np.zeros((nq, N_SIMULATION+1))*np.nan
    v = np.zeros((nv, N_SIMULATION+1))*np.nan
    u = np.zeros((nv, N_SIMULATION+1))*np.nan
    f = np.zeros((3*len(conf.contact_frames), N_SIMULATION+1))
    f_pred_int = np.zeros((3*len(conf.contact_frames), N_SIMULATION+1))
    f_inner = np.zeros((3*len(conf.contact_frames), N_SIMULATION*ndt))
    f_pred = np.zeros((3*len(conf.contact_frames), N_SIMULATION*ndt))
    f_avg  = np.zeros((3*len(conf.contact_frames), N_SIMULATION*ndt))
    f_avg2 = np.zeros((3*len(conf.contact_frames), N_SIMULATION*ndt))
    f_avg_pre_projection  = np.zeros((3*len(conf.contact_frames), N_SIMULATION*ndt))
    f_avg2_pre_projection = np.zeros((3*len(conf.contact_frames), N_SIMULATION*ndt))
    x_pred  = np.zeros((6*len(conf.contact_frames), N_SIMULATION+1))
    x_pred2 = np.zeros((6*len(conf.contact_frames), N_SIMULATION+1))
    p0  = np.zeros((simu.nk, N_SIMULATION+1))
    dp0 = np.zeros((simu.nk, N_SIMULATION+1))
    dp0_qp = np.zeros((simu.nk, N_SIMULATION+1))
    dp = np.zeros((simu.nk, N_SIMULATION+1))
    dp_fd = np.zeros((simu.nk, N_SIMULATION+1))
    dJv = np.zeros((simu.nk, N_SIMULATION+1))
    dJv_fd = np.zeros((simu.nk, N_SIMULATION+1))
    mat_mult_expm = np.zeros(N_SIMULATION) # number of matrix multiplications inside expm
    mat_norm_expm = np.zeros(N_SIMULATION) # norm of the matrix used by exmp
    
    q[:,0] = np.copy(simu.q)
    v[:,0] = np.copy(simu.v)
#    f[:,0] = np.copy(simu.f)
    
    try:
        for i in range(0, N_SIMULATION):
            u[:,i] = tau_0 + tau_A * np.sin(2*np.pi*tau_f*t)
            q[:,i+1], v[:,i+1], f_i = simu.simulate(u[:,i], dt, ndt,
                                      simu_params['use_exp_int'])
            f[:, i+1] = f_i
            f_pred_int[:,i+1] = simu.f_pred_int        
            f_inner[:, i*ndt:(i+1)*ndt] = simu.f_inner
            f_pred[:, i*ndt:(i+1)*ndt] = simu.f_pred
            f_avg[:, i*ndt:(i+1)*ndt] = simu.F_avg
            f_avg2[:, i*ndt:(i+1)*ndt] = simu.F_avg2
            f_avg_pre_projection[:, i*ndt:(i+1)*ndt] = simu.F_avg_pre_projection
            f_avg2_pre_projection[:, i*ndt:(i+1)*ndt] = simu.F_avg2_pre_projection
            
            x_pred[:,i+1] = simu.x_pred
            x_pred2[:,i+1] = simu.x_pred2
            
            p0[:,i] = simu.p0
            dp0[:,i] = simu.dp0
            dp0_qp[:,i] = simu.dp0_qp
            dp[:,i] = simu.debug_dp
            dp_fd[:,i] = simu.debug_dp_fd
            dJv[:,i] = simu.debug_dJv
            dJv_fd[:,i] = simu.debug_dJv_fd
            
            mat_mult_expm[i] = simu.expMatHelper.mat_mult
            mat_norm_expm[i] = simu.expMatHelper.mat_norm
    
            if i % PRINT_N == 0:
                print("Time %.3f" % (t))
    
            t += dt
    except Exception as e:
        print(e)
        print("ERROR WHILE RUNNING SIMULATION")
        raise e

    time_spent = time.time() - time_start
    print("Real-time factor:", t/time_spent)
    results = Empty()
    for key in simu_params.keys():
        results.__dict__[key] = simu_params[key]
    results.q = q
    results.v = v
    results.u = u
    results.f = f
    results.f_inner = f_inner
    results.f_pred = f_pred
    results.f_pred_int = f_pred_int
    results.f_avg = f_avg
    results.f_avg2 = f_avg2
    results.f_avg_pre_projection = f_avg_pre_projection
    results.f_avg2_pre_projection = f_avg2_pre_projection
    results.x_pred = x_pred
    results.x_pred2 = x_pred2
    results.p0 = p0
    results.dp0 = dp0
    results.dp0_qp = dp0_qp
    results.dp = dp
    results.dp_fd = dp_fd
    results.dJv = dJv
    results.dJv_fd = dJv_fd
    results.mat_mult_expm = mat_mult_expm
    results.mat_norm_expm = mat_norm_expm
    return results

# import cProfile
# cProfile.run('run_simulation(q0, v0)')    
data = {}
for simu_params in SIMU_PARAMS:
    name = simu_params['name']
    print("\nStart simulation", name)
    data[name] = run_simulation(q0, v0, simu_params)

print("\nStart simulation ground truth")
data_ground_truth = run_simulation(q0, v0, GROUND_TRUTH_SIMU_PARAMS)
data['ground-truth'] = data_ground_truth

# COMPUTE INTEGRATION ERRORS:
print('\n')
ndt = {}
total_err, err_traj = {}, {}
mat_mult_expm, mat_norm_expm = {}, {}
for name in sorted(data.keys()):
    if(name=='ground-truth'): continue
    d = data[name]
    err = (norm(d.q - data_ground_truth.q) + norm(d.v - data_ground_truth.v)) / d.q.shape[0]
    err_per_time = np.array(norm(d.q - data_ground_truth.q, axis=0)) + \
                    np.array(norm(d.v - data_ground_truth.v, axis=0))
    print(name, 'Total error: %.2f'%np.log10(err))
    if(d.method_name not in total_err):
        mat_norm_expm[d.method_name] = []
        mat_mult_expm[d.method_name] = []
        total_err[d.method_name] = []
        ndt[d.method_name] = []
    mat_norm_expm[d.method_name] += [np.mean(d.mat_norm_expm)]
    mat_mult_expm[d.method_name] += [np.mean(d.mat_mult_expm)]
    total_err[d.method_name] += [err]
    err_traj[name] = err_per_time
    ndt[d.method_name] += [d.ndt]

# PLOT STUFF
line_styles = 10*['-o', '--o', '-.o', ':o']
tt = np.arange(0.0, (N_SIMULATION+1)*dt, dt)[:N_SIMULATION+1]


# PLOT INTEGRATION ERRORS
if(PLOT_INTEGRATION_ERRORS):
    (ff, ax) = plut.create_empty_figure(1)
    j = 0
    for name in sorted(total_err.keys()):
        err = total_err[name]
        ax.plot(ndt[name], total_err[name], line_styles[j], alpha=0.7, label=name)
        j += 1
    ax.set_xlabel('Number of time steps')
    ax.set_ylabel('Error norm')
    ax.set_xscale('log')
    ax.set_yscale('log')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    
if(PLOT_INTEGRATION_ERROR_TRAJECTORIES):
    (ff, ax) = plut.create_empty_figure(1)
    j = 0
    for name in sorted(err_traj.keys()):
        err = err_traj[name]
        ax.plot(tt, err_traj[name], line_styles[j], alpha=0.7, label=name)
        j += 1
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error norm')
    ax.set_yscale('log')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    
if(PLOT_CONTACT_POINT_PREDICTION):
    (ff, ax) = plut.create_empty_figure(3, 2)
    ax = ax.reshape(6)
    j = 0
    for (name, d) in data.items():
        if('exp' in name):
            for i in range(6):
                ax[i].plot(tt, d.x_pred[i, :], line_styles[j], alpha=0.7, label=name+' pred')
                ax[i].plot(tt, d.x_pred2[i, :], line_styles[j], alpha=0.7, label=name+' pred2')
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel('X '+str(i))
            j += 1
            print(name+" x pred err: ", np.linalg.norm(d.x_pred-d.x_pred2))
            leg = ax[0].legend()
            if(leg): leg.get_frame().set_alpha(0.5)
        
if(PLOT_ANCHOR_POINTS):
    (ff, ax) = plut.create_empty_figure(2, 1)
    j = 0
    for (name, d) in data.items():
        for i in range(2):
            ax[i].plot(tt, d.p0[i, :], line_styles[j], alpha=0.7, label='p0 '+name)
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('p0 '+str(i))
        j += 1
        leg = ax[0].legend()
        if(leg): leg.get_frame().set_alpha(0.5)
        
    (ff, ax) = plut.create_empty_figure(2, 1)
    ax = ax.reshape(2)
    j = 0
    for (name, d) in data.items():
        for i in range(2):
            ax[i].plot(tt, d.dp0[i, :], line_styles[j], alpha=0.7, label='dp0 '+name)
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('dp0 '+str(i))
        j += 1
        leg = ax[0].legend()
        if(leg): leg.get_frame().set_alpha(0.5)
            
# PLOT THE CONTACT FORCES OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_FORCES):
    (ff, ax) = plut.create_empty_figure(3, 1)
    ax = ax.reshape(3)
    j = 0
    for (name, d) in data.items():
        for i in range(3):
            ax[i].plot(tt, d.f[i, :], line_styles[j], alpha=0.7, label=name+' f')
            #ax[i].plot(tt, -d.u[i, :], line_styles[j], alpha=0.7, label=name+' u')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Force '+str(i)+' [N]')
        j += 1
        leg = ax[0].legend()
        if(leg): leg.get_frame().set_alpha(0.5)
        
    (ff, ax) = plut.create_empty_figure(1, 1)
    for (name, d) in data.items():
        ax.plot(tt, d.f[0, :] / (1e-3+d.f[2,:]), alpha=0.7, label=name)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Force X/Z [N]')
        leg = ax.legend()
        if(leg): leg.get_frame().set_alpha(0.5)

# FOR EACH INTEGRATION METHOD PLOT THE FORCE INTEGRALS
if(PLOT_FORCE_INTEGRALS):
    for (name,d) in data.items():
        if('exp' not in name):
            continue
        (ff, ax) = plut.create_empty_figure(3,1)
        ax = ax.reshape(3)
        tt_log = np.arange(d.f_avg.shape[1]) * T / d.f_avg.shape[1]
        for i in range(3):
            ax[i].plot(tt, d.f[i,:], ' o', markersize=8, label=name)
            ax[i].plot(tt_log, d.f_avg[i,:], 'r v', markersize=6, label=name+' avg ')
            ax[i].plot(tt_log, d.f_avg2[i,:], 'b x', markersize=6, label=name+' avg2 ')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Force [N]')
        leg = ax[-1].legend()
        if(leg): leg.get_frame().set_alpha(0.5)
        
        (ff, ax) = plut.create_empty_figure(3,1)
        ax = ax.reshape(3)
        tt_log = np.arange(d.f_avg.shape[1]) * T / d.f_avg.shape[1]
        for i in range(3):
            ax[i].plot(tt, d.f[i,:], ' o', markersize=8, label=name)
            ax[i].plot(tt_log, d.f_avg_pre_projection[i,:], 'r v', markersize=6, label=name+' avg pre')
            ax[i].plot(tt_log, d.f_avg2_pre_projection[i,:], 'b x', markersize=6, label=name+' avg2 pre')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Force [N]')
        leg = ax[-1].legend()
        if(leg): leg.get_frame().set_alpha(0.5)
       
# FOR EACH INTEGRATION METHOD PLOT THE FORCE PREDICTIONS
if(PLOT_FORCE_PREDICTIONS):
    for (name,d) in data.items():
       (ff, ax) = plut.create_empty_figure(3,1)
       ax = ax.reshape(3)
       tt_log = np.arange(d.f_pred.shape[1]) * T / d.f_pred.shape[1]
       for i in range(3):
           ax[i].plot(tt, d.f[i,:], ' o', markersize=8, label=name)
           ax[i].plot(tt, d.f_pred_int[i,:], ' s', markersize=8, label=name+' pred int')
           ax[i].plot(tt_log, d.f_pred[i,:], 'r v', markersize=6, label=name+' pred ')
           ax[i].plot(tt_log, d.f_inner[i,:], 'b x', markersize=6, label=name+' real ')
           ax[i].set_xlabel('Time [s]')
           ax[i].set_ylabel('Force [N]')
       leg = ax[-1].legend()
       if(leg): leg.get_frame().set_alpha(0.5)
       
       # force prediction error of Euler, i.e. assuming force remains contact during time step
       ndt = int(d.f_inner.shape[1] / (d.f.shape[1]-1))
       f_pred_err_euler = np.array([d.f[:,int(np.floor(i/ndt))] - d.f_inner[:,i] for i in range(d.f_inner.shape[1])]).T
       
       # force prediction error of matrix exponential 
       f_pred_err = d.f_pred - d.f_inner
       print(name, 'Force pred err max exp:', np.sum(np.abs(f_pred_err))/(f_pred_err.shape[0]*f_pred_err.shape[1]))
       print(name, 'Force pred err Euler:  ', np.sum(np.abs(f_pred_err_euler))/(f_pred_err.shape[0]*f_pred_err.shape[1]))

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