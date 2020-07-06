# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:02:12 2020

@author: student
"""
import numpy as np
from numpy.linalg import norm
import pinocchio as pin
import time
import consim_py.utils.plot_utils as plut

class Empty:
    pass

dt_ref = 0.01   # time step of reference trajectories

def interpolate_state(robot, x1, x2, d):
    """ interpolate state for feedback at higher rate that plan """
    x = np.zeros(robot.model.nq+robot.model.nv)
    x[:robot.model.nq] =  pin.interpolate(robot.model, x1[:robot.model.nq], x2[:robot.model.nq], d)
    x[robot.model.nq:] = x1[robot.model.nq:] + d*(x2[robot.model.nq:] - x1[robot.model.nq:])
    return x


def state_diff(robot, x1, x2):
    """ returns x2 - x1 """
    xdiff = np.zeros(2*robot.model.nv)
    xdiff[:robot.model.nv] = pin.difference(robot.model, x1[:robot.model.nq], x2[:robot.model.nq]) 
    xdiff[robot.model.nv:] = x2[robot.model.nq:] - x1[robot.model.nq:]
    return xdiff
    
    
def load_solo_ref_traj(robot, dt, motionName='trot'):
    ''' Load reference trajectories '''
    refX_ = np.load('../demo/references/'+motionName+'_reference_states.npy').squeeze()
    refU_ = np.load('../demo/references/'+motionName+'_reference_controls.npy').squeeze() 
    feedBack_ = np.load('../demo/references/'+motionName+'_feedback.npy').squeeze() 
    refX_[:,2] -= 15.37e-3   # ensure contact points are inside the ground at t=0
    N = refU_.shape[0]     
    
    # interpolate reference traj
    ndt_ref = int(dt_ref/dt)
    refX     = np.empty((N*ndt_ref+1, refX_.shape[1]))
    refU     = np.empty((N*ndt_ref, refU_.shape[1]))
    feedBack = np.empty((N*ndt_ref, feedBack_.shape[1], feedBack_.shape[2]))
    for i in range(N):
        for j in range(ndt_ref):
            k = i*ndt_ref+j
            refX[k,:] = interpolate_state(robot, refX_[i], refX_[i+1], j/ndt_ref)
            refU[k,:] = refU_[i]
            feedBack[k,:,:] = feedBack_[i]
    return refX, refU, feedBack


def play_motion(robot, q, dt):
    N_SIMULATION = q.shape[1]
    t, i = 0.0, 0
    time_start_viewer = time.time()
    robot.display(q[:,0])
    while True:
        time_passed = time.time()-time_start_viewer - t
        if(time_passed<dt):
            time.sleep(dt-time_passed)
            ni = 1
        else:
            ni = int(time_passed/dt)
        i += ni
        t += ni*dt
        if(i>=N_SIMULATION or np.any(np.isnan(q[:,i]))):
            break
        robot.display(q[:,i])
        
        
def compute_integration_errors(data, robot):
    print('\n')
    res = Empty()
    res.ndt, res.comp_time = {}, {}
    res.err_2norm_avg, res.err_infnorm_avg, res.err_infnorm_max, res.err_traj_2norm, res.err_traj_infnorm = {}, {}, {}, {}, {}
    res.mat_mult, res.mat_norm = {}, {}
    for name in sorted(data.keys()):
        if('ground-truth' in name): continue
        d = data[name]
        if(d.use_exp_int==0): data_ground_truth = data['ground-truth-euler']
        else:                 data_ground_truth = data['ground-truth-exp']
        
        err_vec = np.empty((2*robot.nv, d.q.shape[1]))
        err_per_time_2 = np.empty(d.q.shape[1])
        err_per_time_inf = np.empty(d.q.shape[1])
        err_2, err_inf = 0.0, 0.0
        for i in range(d.q.shape[1]):
            err_vec[:robot.nv,i] = pin.difference(robot.model, d.q[:,i], data_ground_truth.q[:,i])
            err_vec[robot.nv:,i] = d.v[:,i] - data_ground_truth.v[:,i]
            err_per_time_2[i]   = norm(err_vec[:,i])
            err_per_time_inf[i] = norm(err_vec[:,i], np.inf)
            err_2   += err_per_time_2[i]
            err_inf += err_per_time_inf[i]
        err_2 /= d.q.shape[1]
        err_inf /= d.q.shape[1]
        err_peak = np.max(err_per_time_inf)
        print(name, 'Log error 2-norm: %.2f'%np.log10(err_2))
        if(d.method_name not in res.err_2norm_avg):
            res.err_2norm_avg[d.method_name] = []
            res.err_infnorm_max[d.method_name] = []
            res.err_infnorm_avg[d.method_name] = []
            res.ndt[d.method_name] = []
            res.comp_time[d.method_name] = []
            res.mat_mult[d.method_name] = []
            res.mat_norm[d.method_name] = []
        res.err_2norm_avg[d.method_name] += [err_2]
        res.err_infnorm_avg[d.method_name] += [err_inf]
        res.err_infnorm_max[d.method_name] += [err_peak]
        res.err_traj_2norm[name] = err_per_time_2
        res.err_traj_infnorm[name] = err_per_time_inf
        res.ndt[d.method_name] += [d.ndt]
        res.comp_time[d.method_name] += [d.computation_times['inner-step'].avg * d.ndt]
        res.mat_mult[d.method_name] += [np.mean(d.mat_mult)]
        res.mat_norm[d.method_name] += [np.mean(d.mat_norm)]
    return res
        
        
def plot_multi_x_vs_y_log_scale(y, x, ylabel, xlabel='Number of time steps', logy=True):
    line_styles = 10*['-o', '--o', '-.o', ':o']
    (ff, ax) = plut.create_empty_figure(1)
    j = 0
    for name in sorted(y.keys()):
        ax.plot(x[name], y[name], line_styles[j], alpha=0.7, label=name)
        j += 1
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    if(logy):
        ax.set_yscale('log')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)