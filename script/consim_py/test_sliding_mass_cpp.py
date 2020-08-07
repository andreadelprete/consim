''' Test cpp simulator with point-mass sliding on a flat floor 
'''
import time
#from simulator import RobotSimulator
import consim 
import numpy as np
from numpy import nan
from numpy.linalg import norm as norm

import pinocchio as pin 
from pinocchio.robot_wrapper import RobotWrapper
import os, sys
import matplotlib.pyplot as plt 
import consim_py.utils.plot_utils as plut
import conf_sliding_mass_cpp as conf

from consim_py.ral2020.simu_cpp_common import Empty, dt_ref, play_motion, \
    plot_multi_x_vs_y_log_scale, compute_integration_errors, run_simulation

class SinusoidalController:
    ''' u(t) = u_0 + u_A * sin(2*pi*u_f*t) '''

    def __init__(self, u_0, u_A, u_f, dt):
        self.u_0 = u_0
        self.u_A = u_A
        self.u_f = u_f
        self.dt = dt
        self.t = 0.0

    def reset(self, q, v, time_before_start):
        self.t = 0.0

    def compute_control(self, q, v):
        u = self.u_0 + self.u_A * np.sin(2*np.pi*self.u_f*self.t)
        self.t += self.dt
        return u
        
print("".center(conf.LINE_WIDTH, '#'))
print(" Test Sliding Mass ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH, '#'))

# parameters of the simulation to be tested
i_min = 0
i_max = i_min+10
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
        'name': 'exp%4d slip-cont'%(2**i),
        'method_name': 'exp slip-cont',
        'use_exp_int': 1,
        'ndt': 2**i,
        'assumeSlippageContinues': 1
    }]

for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'exp%4d slip-stop'%(2**i),
        'method_name': 'exp slip-stop',
        'use_exp_int': 1,
        'ndt': 2**i,
        'assumeSlippageContinues': 0
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
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 0
PLOT_CONTACT_POINT_PREDICTION = 0
PLOT_CONTACT_POINTS = 0
PLOT_SLIPPING = 0
PLOT_ANCHOR_POINTS = 0

dt = 0.01                      # controller time step
T = 0.5
N = int(T/dt)

N_SIMULATION = int(T/dt)        # number of time steps simulated
PRINT_N = int(conf.PRINT_T/dt)

robot = RobotWrapper.BuildFromURDF(conf.urdf_path, [conf.mesh_path], pin.JointModelFreeFlyer()) 
nq, nv = robot.nq, robot.nv
q0, v0 = conf.q0, conf.v0

''' Apply a force that pushes the point mass and violates the friction cone '''
tau_0 = np.array([2., 0., -0.19, 0., 0., 0.])
tau_A = np.array([2., 0.,  0.,   0., 0., 0.])
tau_f = np.array([2., 0.,  0.,   0., 0., 0.])
#  u[:,i] = tau_0 + tau_A * np.sin(2*np.pi*tau_f*t)
controller = SinusoidalController(tau_0, tau_A, tau_f, dt)
  
data = {}
for simu_params in SIMU_PARAMS:
    name = simu_params['name']
    print("\nStart simulation", name)
    if(simu_params['use_exp_int']):
        data[name] = run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params)
    else:
        data[name] = run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params)
                            
print("\nStart simulation ground truth")
data['ground-truth-exp'] = run_simulation(conf, dt, N, robot, controller, q0, v0, GROUND_TRUTH_EXP_SIMU_PARAMS)
data['ground-truth-euler'] = run_simulation(conf, dt, N, robot, controller, q0, v0, GROUND_TRUTH_EULER_SIMU_PARAMS)
data['ground-truth-euler'] = data['ground-truth-exp']

# COMPUTE INTEGRATION ERRORS:
res = compute_integration_errors(data, robot, dt)

# PLOT STUFF
line_styles = 10*['-o', '--o', '-.o', ':o']
tt = np.arange(0.0, (N_SIMULATION+1)*dt, dt)[:N_SIMULATION+1]


if(PLOT_INTEGRATION_ERRORS):
    plot_multi_x_vs_y_log_scale(res.err_infnorm_avg, res.dt, 'Mean error inf-norm', 'Time step [s]')
#    plot_multi_x_vs_y_log_scale(res.err_infnorm_avg, res.comp_time, 'Mean error inf-norm', 'Computation time per step')
#    plot_multi_x_vs_y_log_scale(res.err_infnorm_avg, res.realtime_factor, 'Mean error inf-norm', 'Real-time factor')
    
if(PLOT_INTEGRATION_ERROR_TRAJECTORIES):    
    (ff, ax) = plut.create_empty_figure(1)
    for (j,name) in enumerate(sorted(res.err_traj_infnorm.keys())):
        if(len(res.err_traj_infnorm[name])>0):
            ax.plot(tt, res.err_traj_infnorm[name], line_styles[j], alpha=0.7, label=name)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error inf-norm')
    ax.set_yscale('log')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    
def get_empty_figure(n):
    if(n<5):
        (ff, ax) = plut.create_empty_figure(n, 1)
        if n==1:
            ax = [ax]
    else:
        (ff, ax) = plut.create_empty_figure(int(n/2), 2)
        ax = ax.reshape(ax.shape[0]*ax.shape[1])
    return ax
        
# PLOT THE CONTACT FORCES OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_FORCES):        
    nc = len(conf.contact_frames)
    ax = get_empty_figure(nc)    
    for (name, d) in data.items():
        for i in range(nc):
#            ax[i].plot(tt, norm(d.f[0:2,i,:], axis=0) / (1e-3+d.f[2,i,:]), alpha=0.7, label=name)
            ax[i].plot(tt, d.f[0,i,:], alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('F_X [N]')
            leg = ax[i].legend()
            if(leg): leg.get_frame().set_alpha(0.5)
            
    for (name, d) in data.items():
        if d.use_exp_int:
            ax = get_empty_figure(nc)    
            for i in range(nc):
    #            ax[i].plot(tt, norm(d.f[0:2,i,:], axis=0) / (1e-3+d.f[2,i,:]), alpha=0.7, label=name)
                ax[i].plot(tt, d.f[0,i,:], alpha=0.7, label='f')
                ax[i].plot(tt, d.f_avg[0,i,:], alpha=0.7, label='f avg')
                ax[i].plot(tt, d.f_avg2[0,i,:], alpha=0.7, label='f avg2')
                ax[i].plot(tt, d.f_prj[0,i,:], alpha=0.7, label='f prj')
                ax[i].plot(tt, d.f_prj2[0,i,:], alpha=0.7, label='f prj2')
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel('F_X [N]')
                ax[i].set_title(name)
                leg = ax[i].legend()
                if(leg): leg.get_frame().set_alpha(0.5)

if(PLOT_CONTACT_POINTS):
    ax = get_empty_figure(nc)
    for (name, d) in data.items():
        for i in range(nc):
            ax[i].plot(tt, d.p[0,i,:], alpha=0.7, label=name+' p')
            ax[i].plot(tt, d.p0[0,i,:], alpha=0.7, label=name+' p0')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('X [m]')
            leg = ax[i].legend()
            if(leg): leg.get_frame().set_alpha(0.5)
            
# PLOT THE SLIPPING FLAG OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_SLIPPING):
    ax = get_empty_figure(nc)
    for (name, d) in data.items():        
        for i in range(nc):
            ax[i].plot(tt, d.slipping[i,:tt.shape[0]], alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
    ax[0].set_ylabel('Contact Slipping Flag')
    leg = ax[0].legend()
    if(leg): leg.get_frame().set_alpha(0.5)

    ax = get_empty_figure(nc)
    for (name, d) in data.items():
        for i in range(nc):
            ax[i].plot(tt, d.active[i,:tt.shape[0]], alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
    ax[0].set_ylabel('Contact Active Flag')
    leg = ax[0].legend()
    if(leg): leg.get_frame().set_alpha(0.5)
#    plut.saveFigure("active_contact_flag_"+descr_str)

       
# PLOT THE JOINT ANGLES OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_BASE_POS):
    ax = get_empty_figure(3)
    j = 0
    for (name, d) in data.items():
        for i in range(3):
            ax[i].plot(tt, d.q[i, :], line_styles[j], alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Base pos [m]')
        j += 1
        leg = ax[0].legend()
        if(leg): leg.get_frame().set_alpha(0.5)
        
#    
#if(PLOT_CONTACT_POINT_PREDICTION):
#    (ff, ax) = plut.create_empty_figure(3, 2)
#    ax = ax.reshape(6)
#    j = 0
#    for (name, d) in data.items():
#        if('exp' in name):
#            for i in range(6):
#                ax[i].plot(tt, d.x_pred[i, :], line_styles[j], alpha=0.7, label=name+' pred')
#                ax[i].plot(tt, d.x_pred2[i, :], line_styles[j], alpha=0.7, label=name+' pred2')
#                ax[i].set_xlabel('Time [s]')
#                ax[i].set_ylabel('X '+str(i))
#            j += 1
#            print(name+" x pred err: ", np.linalg.norm(d.x_pred-d.x_pred2))
#            leg = ax[0].legend()
#            if(leg): leg.get_frame().set_alpha(0.5)
#        
#if(PLOT_ANCHOR_POINTS):
#    (ff, ax) = plut.create_empty_figure(2, 1)
#    j = 0
#    for (name, d) in data.items():
#        for i in range(2):
#            ax[i].plot(tt, d.p0[i, :], line_styles[j], alpha=0.7, label='p0 '+name)
#            ax[i].set_xlabel('Time [s]')
#            ax[i].set_ylabel('p0 '+str(i))
#        j += 1
#        leg = ax[0].legend()
#        if(leg): leg.get_frame().set_alpha(0.5)
#        
#    (ff, ax) = plut.create_empty_figure(2, 1)
#    ax = ax.reshape(2)
#    j = 0
#    for (name, d) in data.items():
#        for i in range(2):
#            ax[i].plot(tt, d.dp0[i, :], line_styles[j], alpha=0.7, label='dp0 '+name)
#            ax[i].set_xlabel('Time [s]')
#            ax[i].set_ylabel('dp0 '+str(i))
#        j += 1
#        leg = ax[0].legend()
#        if(leg): leg.get_frame().set_alpha(0.5)
#            
## PLOT THE CONTACT FORCES OF ALL INTEGRATION METHODS ON THE SAME PLOT
#if(PLOT_FORCES):
#    (ff, ax) = plut.create_empty_figure(3, 1)
#    ax = ax.reshape(3)
#    j = 0
#    for (name, d) in data.items():
#        for i in range(3):
#            ax[i].plot(tt, d.f[i, :], line_styles[j], alpha=0.7, label=name+' f')
#            #ax[i].plot(tt, -d.u[i, :], line_styles[j], alpha=0.7, label=name+' u')
#            ax[i].set_xlabel('Time [s]')
#            ax[i].set_ylabel('Force '+str(i)+' [N]')
#        j += 1
#        leg = ax[0].legend()
#        if(leg): leg.get_frame().set_alpha(0.5)
#        
#    (ff, ax) = plut.create_empty_figure(1, 1)
#    for (name, d) in data.items():
#        ax.plot(tt, d.f[0, :] / (1e-3+d.f[2,:]), alpha=0.7, label=name)
#        ax.set_xlabel('Time [s]')
#        ax.set_ylabel('Force X/Z [N]')
#        leg = ax.legend()
#        if(leg): leg.get_frame().set_alpha(0.5)
#
## FOR EACH INTEGRATION METHOD PLOT THE FORCE INTEGRALS
#if(PLOT_FORCE_INTEGRALS):
#    for (name,d) in data.items():
#        if('exp' not in name):
#            continue
#        (ff, ax) = plut.create_empty_figure(3,1)
#        ax = ax.reshape(3)
#        tt_log = np.arange(d.f_avg.shape[1]) * T / d.f_avg.shape[1]
#        for i in range(3):
#            ax[i].plot(tt, d.f[i,:], ' o', markersize=8, label=name)
#            ax[i].plot(tt_log, d.f_avg[i,:], 'r v', markersize=6, label=name+' avg ')
#            ax[i].plot(tt_log, d.f_avg2[i,:], 'b x', markersize=6, label=name+' avg2 ')
#            ax[i].set_xlabel('Time [s]')
#            ax[i].set_ylabel('Force [N]')
#        leg = ax[-1].legend()
#        if(leg): leg.get_frame().set_alpha(0.5)
#        
#        (ff, ax) = plut.create_empty_figure(3,1)
#        ax = ax.reshape(3)
#        tt_log = np.arange(d.f_avg.shape[1]) * T / d.f_avg.shape[1]
#        for i in range(3):
#            ax[i].plot(tt, d.f[i,:], ' o', markersize=8, label=name)
#            ax[i].plot(tt_log, d.f_avg_pre_projection[i,:], 'r v', markersize=6, label=name+' avg pre')
#            ax[i].plot(tt_log, d.f_avg2_pre_projection[i,:], 'b x', markersize=6, label=name+' avg2 pre')
#            ax[i].set_xlabel('Time [s]')
#            ax[i].set_ylabel('Force [N]')
#        leg = ax[-1].legend()
#        if(leg): leg.get_frame().set_alpha(0.5)
#       
## FOR EACH INTEGRATION METHOD PLOT THE FORCE PREDICTIONS
#if(PLOT_FORCE_PREDICTIONS):
#    for (name,d) in data.items():
#       (ff, ax) = plut.create_empty_figure(3,1)
#       ax = ax.reshape(3)
#       tt_log = np.arange(d.f_pred.shape[1]) * T / d.f_pred.shape[1]
#       for i in range(3):
#           ax[i].plot(tt, d.f[i,:], ' o', markersize=8, label=name)
#           ax[i].plot(tt, d.f_pred_int[i,:], ' s', markersize=8, label=name+' pred int')
#           ax[i].plot(tt_log, d.f_pred[i,:], 'r v', markersize=6, label=name+' pred ')
#           ax[i].plot(tt_log, d.f_inner[i,:], 'b x', markersize=6, label=name+' real ')
#           ax[i].set_xlabel('Time [s]')
#           ax[i].set_ylabel('Force [N]')
#       leg = ax[-1].legend()
#       if(leg): leg.get_frame().set_alpha(0.5)
#       
#       # force prediction error of Euler, i.e. assuming force remains contact during time step
#       ndt = int(d.f_inner.shape[1] / (d.f.shape[1]-1))
#       f_pred_err_euler = np.array([d.f[:,int(np.floor(i/ndt))] - d.f_inner[:,i] for i in range(d.f_inner.shape[1])]).T
#       
#       # force prediction error of matrix exponential 
#       f_pred_err = d.f_pred - d.f_inner
#       print(name, 'Force pred err max exp:', np.sum(np.abs(f_pred_err))/(f_pred_err.shape[0]*f_pred_err.shape[1]))
#       print(name, 'Force pred err Euler:  ', np.sum(np.abs(f_pred_err_euler))/(f_pred_err.shape[0]*f_pred_err.shape[1]))
#
## PLOT THE JOINT ANGLES OF ALL INTEGRATION METHODS ON THE SAME PLOT
#if(PLOT_BASE_POS):
#    (ff, ax) = plut.create_empty_figure(3)
#    ax = ax.reshape(3)
#    j = 0
#    for (name, d) in data.items():
#        for i in range(3):
#            ax[i].plot(tt, d.q[i, :], line_styles[j], alpha=0.7, label=name)
#            ax[i].set_xlabel('Time [s]')
#            ax[i].set_ylabel('Base pos [m]')
#        j += 1
#        leg = ax[0].legend()
#        if(leg): leg.get_frame().set_alpha(0.5)
#        
plt.show()