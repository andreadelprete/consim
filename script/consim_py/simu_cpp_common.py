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
    if motionName[-4:] == 'jump':
        refX_[:,2] -= 1.e-6
    else:
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
    ''' play motion in viewer '''
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
        
        
def compute_integration_errors(data, robot, dt):
    print('\n')
    res = Empty()
    res.ndt, res.dt, res.comp_time, res.realtime_factor = {}, {}, {}, {}
    res.err_2norm_avg, res.err_infnorm_avg, res.err_infnorm_max = {}, {}, {}
    res.err_traj_2norm, res.err_traj_infnorm = {}, {}
    res.mat_mult, res.mat_norm = {}, {}
    for name in sorted(data.keys()):
        if('ground-truth' in name): continue
        d = data[name]
        data_ground_truth = data['ground-truth-'+d.simulator]
#        if(d.simulator=='exponential'): data_ground_truth = data['ground-truth-exp']
#        elif(d.simulator=='euler'): data_ground_truth = data['ground-truth-euler']
#        elif(d.simulator=='implicit-euler'): data_ground_truth = data['ground-truth-implicit-euler']
#        elif(d.simulator=='rk4'): data_ground_truth = data['ground-truth-rk4']
#        else: raise BaseException('Ground truth method not recognized ')
        
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
        print(name, 'Log error inf-norm: %.2f'%np.log10(err_inf))
        if(d.method_name not in res.err_2norm_avg):
            for k in res.__dict__.keys():
                res.__dict__[k][d.method_name] = []

        res.err_2norm_avg[d.method_name] += [err_2]
        res.err_infnorm_avg[d.method_name] += [err_inf]
        res.err_infnorm_max[d.method_name] += [err_peak]
        res.err_traj_2norm[name] = err_per_time_2
        res.err_traj_infnorm[name] = err_per_time_inf
        res.ndt[d.method_name] += [d.ndt]
        res.dt[d.method_name] += [dt/d.ndt]
        try:
            comp_time = d.computation_times['substep'].avg * d.ndt
        except:
            comp_time = np.nan
        res.comp_time[d.method_name] += [comp_time]
        res.realtime_factor[d.method_name] += [dt/comp_time]
        if(d.simulator=='exponential'):
            try:
                res.mat_mult[d.method_name] += [np.mean(d.mat_mult)]
                res.mat_norm[d.method_name] += [np.mean(d.mat_norm)]
            except:
                pass
    return res
        
        
def plot_multi_x_vs_y_log_scale(y, x, ylabel, xlabel='Number of time steps', logy=True, logx=True):
    line_styles = 10*['-o', '--o', '-.o', ':o']
    (ff, ax) = plut.create_empty_figure(1)
    j = 0
    for name in sorted(y.keys()):
        if(len(y[name])>0):
            ax.plot(x[name], y[name], line_styles[j], alpha=0.7, markerSize=10, label=name)
            j += 1
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if(logx):
        ax.set_xscale('log')
    if(logy):
        ax.set_yscale('log')
    leg = ax.legend(loc='best')
    if(leg): leg.get_frame().set_alpha(0.5)
    return (ff, ax)
    
def plot_multi_x_vs_y_rate_of_change(y, x, ylabel, xlabel='Number of time steps', logy=True, logx=True):
    line_styles = 10*['-o', '--o', '-.o', ':o']
    (ff, ax) = plut.create_empty_figure(1)
    j = 0
    for name in sorted(y.keys()):
        if(len(y[name])>0):
            ax.plot(x[name][:-1], np.diff(y[name])/np.diff(x[name]), line_styles[j], alpha=0.7, markerSize=10, label=name)
            j += 1
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if(logx):
        ax.set_xscale('log')
    if(logy):
        ax.set_yscale('log')
    leg = ax.legend(loc='best')
    if(leg): leg.get_frame().set_alpha(0.5)
    return (ff, ax)
    
    
def run_simulation(conf, dt, N, robot, controller, q0, v0, simu_params, ground_truth=None, comp_times=None):  
    import consim
    compute_predicted_forces = False     
    nq, nv = robot.nq, robot.nv
    ndt = simu_params['ndt']
#    use_exp_int = simu_params['use_exp_int']
    name = simu_params['name']
    simu_type = simu_params['simulator']
    try:
        forward_dyn_method = simu_params['forward_dyn_method']
    except:
        # forward_dyn_method Options 
        #  1: pinocchio.Minverse()
        #  2: pinocchio.aba()
        #  3: Cholesky factorization 
        forward_dyn_method = 3
    try:
        #0: explicit, 1: semi_implicit, 2: classic-explicit
        integration_type = simu_params['integration_type']
    except:
        integration_type = 0
    try:
        max_mat_mult = simu_params['max_mat_mult']
    except:
        max_mat_mult = 100
    try:
        use_balancing = simu_params['use_balancing']
    except:
        use_balancing = True
    try:
        slippageContinues = simu_params['assumeSlippageContinues']
    except:
        slippageContinues = False
    try:
        contact_stabilization_gains = simu_params['contact_stabilization_gains']
    except:
        contact_stabilization_gains = [0, 0]
        
    if('exponential'==simu_type):
        simu = consim.build_exponential_simulator(dt, ndt, robot.model, robot.data,
                                    conf.K, conf.B, conf.mu, conf.anchor_slipping_method,
                                    compute_predicted_forces, forward_dyn_method, integration_type,
                                    max_mat_mult, max_mat_mult, use_balancing)
        simu.assumeSlippageContinues(slippageContinues)
        try:
            simu.setUseDiagonalMatrixExp(simu_params['diagonal_matrix_exp'])
        except:
            pass
        try:
            simu.setUpdateAFrequency(simu_params['update_A_frequency'])
        except:
            pass
    elif('euler' == simu_type):
        simu = consim.build_euler_simulator(dt, ndt, robot.model, robot.data,
                                        conf.K, conf.B, conf.mu, forward_dyn_method, integration_type)
    elif('implicit-euler' == simu_type):
        try:
            use_fin_diff_dyn = simu_params['use_finite_differences_dynamics']
        except:
            use_fin_diff_dyn = False
        try:
            use_fin_diff_nle = simu_params['use_finite_differences_nle']
        except:
            use_fin_diff_nle = False
        try:
            use_current_state_as_initial_guess = simu_params['use_current_state_as_initial_guess']
        except:
            use_current_state_as_initial_guess = False
        try:
            convergence_threshold = simu_params['convergence_threshold']
        except:
            convergence_threshold = 1e-8
        simu = consim.build_implicit_euler_simulator(dt, ndt, robot.model, robot.data,
                                        conf.K, conf.B, conf.mu)
        simu.set_use_finite_differences_dynamics(use_fin_diff_dyn)
        simu.set_use_finite_differences_nle(use_fin_diff_nle)
        simu.set_use_current_state_as_initial_guess(use_current_state_as_initial_guess)
        simu.set_convergence_threshold(convergence_threshold)
    elif('rk4' == simu_type):
        simu = consim.build_rk4_simulator(dt, ndt, robot.model, robot.data,
                                        conf.K, conf.B, conf.mu, forward_dyn_method)
    elif('rigid-euler' == simu_type):
        simu = consim.build_rigid_euler_simulator(dt, ndt, robot.model, robot.data,
                                                  conf.K, conf.B, conf.mu)
        simu.set_contact_stabilization_gains(contact_stabilization_gains[0], contact_stabilization_gains[1])
        try:
            simu.set_integration_scheme(simu_params['integration_scheme'])
        except:
            pass
    else:
        raise Exception("Unknown simulation type: "+simu_type)
                                        
    cpts = []
    for cf in conf.contact_frames:
        if not robot.model.existFrame(cf):
            print(("ERROR: Frame", cf, "does not exist"))
        frame_id = robot.model.getFrameId(cf)
        cpts += [simu.add_contact_point(cf, frame_id, conf.unilateral_contacts)]
        
    simu.reset_state(q0, v0, True)
            
    t = 0.0    
    nc = len(conf.contact_frames)
    results = Empty()
    results.q = np.zeros((nq, N+1))*np.nan
    results.v = np.zeros((nv, N+1))
    results.com_pos = np.zeros((3, N+1))*np.nan
    results.com_vel = np.zeros((3, N+1))
    results.u = np.zeros((nv, N+1))
    results.f = np.zeros((3, nc, N+1))
    results.f_avg = np.zeros((3, nc, N+1))
    results.f_avg2 = np.zeros((3, nc, N+1))
    results.f_prj = np.zeros((3, nc, N+1))
    results.f_prj2 = np.zeros((3, nc, N+1))
    results.p = np.zeros((3, nc, N+1))
    results.dp = np.zeros((3, nc, N+1))
    results.p0 = np.zeros((3, nc, N+1))
    results.slipping = np.zeros((nc, N+1))
    results.active = np.zeros((nc, N+1))
    if('exponential'==simu_type):
        results.mat_mult = np.zeros(N+1)
        results.mat_norm = np.zeros(N+1)
    if('implicit-euler' == simu_type):
        results.avg_iter_num = np.zeros(N+1)
    results.computation_times = {}
    
    results.q[:,0] = np.copy(q0)
    results.v[:,0] = np.copy(v0)
    results.com_pos[:,0] = robot.com(results.q[:,0])
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
        for i in range(0, N):
#            robot.display(results.q[:,i])
            if(ground_truth):                
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
                    
            results.u[:,i] = controller.compute_control(simu.get_q(), simu.get_v())
            simu.step(results.u[:,i])
#            time.sleep(5)
                
            results.q[:,i+1] = simu.get_q()
            results.v[:,i+1] = simu.get_v()
            results.com_pos[:,i+1] = robot.com(results.q[:,i+1])
#            results.com_vel[:,i+1] = robot.com_vel()
            
            if('exponential'==simu_type):
                results.mat_mult[i] = simu.getMatrixMultiplications()
                results.mat_norm[i] = simu.getMatrixExpL1Norm()
            elif('implicit-euler' == simu_type):
                results.avg_iter_num[i] = simu.get_avg_iteration_number()
            
            for ci, cp in enumerate(cpts):
                results.f[:,ci,i+1] = cp.f
                results.f_avg[:,ci,i+1] = cp.f_avg
                results.f_avg2[:,ci,i+1] = cp.f_avg2
                results.f_prj[:,ci,i+1] = cp.f_prj
                results.f_prj2[:,ci,i+1] = cp.f_prj2
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
    
#            if i % PRINT_N == 0:
#                print("Time %.3f" % (t))  
            t += dt
        
#        print("Real-time factor:", t/(time.time() - time_start))
#        consim.stop_watch_report(3)
        if(comp_times):
            for s_key, s_value in comp_times.items():
#                print(s)
                results.computation_times[s_value] = Empty()
                try:
                    results.computation_times[s_value].avg = consim.stop_watch_get_average_time(s_key)                
                    results.computation_times[s_value].tot = consim.stop_watch_get_total_time(s_key)
                except:
                    results.computation_times[s_value].avg = np.nan
                    results.computation_times[s_value].tot = np.nan
#        for key in results.computation_times.keys():
#            print("%20s: %.1f us"%(key, results.computation_times[key].avg*1e6))
            
    except Exception as e:
#        raise e
        print("Exception while running simulation", e)        
        if(comp_times):
            for s_key, s in comp_times.items():
                results.computation_times[s] = Empty()
                results.computation_times[s].avg = np.nan
                results.computation_times[s].tot = np.nan

    if conf.use_viewer:
        play_motion(robot, results.q, dt)
                    
    for key in simu_params.keys():
        results.__dict__[key] = simu_params[key]

    return results