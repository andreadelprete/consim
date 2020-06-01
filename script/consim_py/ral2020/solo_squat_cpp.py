''' Test cpp simulator with point-mass sliding on a flat floor 
'''
import time
import consim 
import numpy as np
from numpy.linalg import norm as norm

import pinocchio as pin 
from example_robot_data.robots_loader import loadSolo
from consim_py.tsid_quadruped import TsidQuadruped

import matplotlib.pyplot as plt 
import consim_py.utils.plot_utils as plut
import conf_solo_squat_cpp as conf

class Empty:
    pass

print("".center(conf.LINE_WIDTH, '#'))
print(" Test Sliding Mass ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH, '#'))

# parameters of the simulation to be tested
i_min = 2
i_max = i_min+4
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

for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'exp ABA%4d'%(2**i),
        'method_name': 'exp ABA',
        'use_exp_int': 1,
        'ndt': 2**i,
        'forward_dyn_method': 2
    }]

for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'exp Chol%4d'%(2**i),
        'method_name': 'exp Chol',
        'use_exp_int': 1,
        'ndt': 2**i,
        'forward_dyn_method': 3
    }]
    
# EULER INTEGRATOR WITH STANDARD SETTINGS
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'euler Minv%4d'%(2**i),
        'method_name': 'euler Minv',
        'use_exp_int': 0,
        'ndt': 2**i,
        'forward_dyn_method': 1
    }]

for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'euler ABA%4d'%(2**i),
        'method_name': 'euler ABA',
        'use_exp_int': 0,
        'ndt': 2**i,
        'forward_dyn_method': 2
    }]

for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'euler Chol%4d'%(2**i),
        'method_name': 'euler Chol',
        'use_exp_int': 0,
        'ndt': 2**i,
        'forward_dyn_method': 3
    }]
    
PLOT_FORCES = 0
PLOT_BASE_POS = 0
PLOT_FORCE_PREDICTIONS = 0
PLOT_FORCE_INTEGRALS = 0
PLOT_INTEGRATION_ERRORS = 1
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 0
PLOT_CONTACT_POINT_PREDICTION = 0
PLOT_ANCHOR_POINTS = 0

dt = 0.01                      # controller time step
T = 0.01
unilateral_contacts = 1
compute_predicted_forces = False
exm_max_mul = 100 
int_max_mul = 100 

conf.q0[2] += 1.0

offset = np.array([0.0, -0.0, 0.0])
amp = np.array([0.0, 0.0, 0.05])
two_pi_f = 2*np.pi*np.array([0.0, .0, 2.0])

N_SIMULATION = int(T/dt)        # number of time steps simulated
PRINT_N = int(conf.PRINT_T/dt)

robot = loadSolo()
nq, nv = robot.nq, robot.nv

if conf.use_viewer:
    robot.initViewer(loadModel=True)
    robot.viewer.gui.createSceneWithFloor('world')
    robot.viewer.gui.setLightingMode('world/floor', 'OFF')
    
    
invdyn = TsidQuadruped(conf, robot, conf.q0, viewer=False)
offset += invdyn.robot.com(invdyn.formulation.data())
two_pi_f_amp = two_pi_f * amp
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp
sampleCom = invdyn.trajCom.computeNext()


def run_simulation(q0, v0, simu_params):
    ndt = simu_params['ndt']
    try:
        forward_dyn_method = simu_params['forward_dyn_method']
    except:
        forward_dyn_method = 1
    if(simu_params['use_exp_int']):
        simu = consim.build_exponential_simulator(dt, ndt, robot.model, robot.data,
                                    conf.K, conf.B, conf.mu, conf.anchor_slipping_method,
                                    compute_predicted_forces, forward_dyn_method, exm_max_mul, int_max_mul)
    else:
        simu = consim.build_euler_simulator(dt, ndt, robot.model, robot.data,
                                        conf.K, conf.B, conf.mu, forward_dyn_method)
                                        
    cpts = []
    for cf in conf.contact_frames:
        if not robot.model.existFrame(cf):
            print(("ERROR: Frame", cf, "does not exist"))
        cpts += [simu.add_contact_point(cf, robot.model.getFrameId(cf), unilateral_contacts)]

    robot.forwardKinematics(q0)
    simu.reset_state(q0, v0, True)
            
    t = 0.0    
    time_start = time.time()
    q = np.zeros((nq, N_SIMULATION+1))*np.nan
    v = np.zeros((nv, N_SIMULATION+1))*np.nan
    u = np.zeros((nv, N_SIMULATION+1))
    f = np.zeros((3, len(conf.contact_frames), N_SIMULATION+1))
    p = np.zeros((3, len(conf.contact_frames), N_SIMULATION+1))
    dp = np.zeros((3, len(conf.contact_frames), N_SIMULATION+1))
    
    q[:,0] = np.copy(q0)
    v[:,0] = np.copy(v0)
#    f[:,0] = np.copy(simu.f)
    
    try:
        for i in range(0, N_SIMULATION):
#            sampleCom.pos(offset + amp * np.sin(two_pi_f*t))
#            sampleCom.vel(two_pi_f_amp * np.cos(two_pi_f*t))
#            sampleCom.acc(-two_pi_f_squared_amp * np.sin(two_pi_f*t))
#            invdyn.comTask.setReference(sampleCom)
#            HQPData = invdyn.formulation.computeProblemData(t, q[:,i], v[:,i])
#            sol = invdyn.solver.solve(HQPData)
#            if(sol.status != 0):
#                print("[%d] QP problem could not be solved! Error code:" % (i), sol.status)
#                break
#            u[6:,i] = invdyn.formulation.getActuatorForces(sol)
#            
            
            simu.step(u[:,i])
            q[:,i+1] = simu.get_q()
            v[:,i+1] = simu.get_v()
            
            for ci, cp in enumerate(cpts):
                f[:,ci,i+1] = cp.f
                p[:,ci,i+1] = cp.x
                dp[:,ci,i+1] = cp.v
                if(cp.active):
                    print("Contact active", cp.name)
            
            if(np.any(np.isnan(v[:,i+1])) or norm(v[:,i+1]) > 1e3):
                raise Exception("Time %.3f Velocities are too large: %.1f. Stop simulation."%(
                                t, norm(v[:,i+1])))
    
            if i % PRINT_N == 0:
                print("Time %.3f" % (t))
                
            if conf.use_viewer:
                robot.display(q[:,i+1])
                time.sleep(dt)
    
            t += dt
    except Exception as e:
        print(e)

    time_spent = time.time() - time_start
    print("Real-time factor:", t/time_spent)
    results = Empty()
    for key in simu_params.keys():
        results.__dict__[key] = simu_params[key]
    results.q = q
    results.v = v
    results.u = u
    results.f = f
    results.p = p
    results.dp = dp
    return results

 
data = {}
for simu_params in SIMU_PARAMS:
    name = simu_params['name']
    print("\nStart simulation", name)
    data[name] = run_simulation(conf.q0, conf.v0, simu_params)

print("\nStart simulation ground truth")
data_ground_truth_exp = run_simulation(conf.q0, conf.v0, GROUND_TRUTH_EXP_SIMU_PARAMS)
data_ground_truth_euler = run_simulation(conf.q0, conf.v0, GROUND_TRUTH_EULER_SIMU_PARAMS)
data['ground-truth-exp'] = data_ground_truth_exp
data['ground-truth-euler'] = data_ground_truth_euler

# COMPUTE INTEGRATION ERRORS:
print('\n')
ndt = {}
total_err, err_traj = {}, {}
mat_mult_expm, mat_norm_expm = {}, {}
for name in sorted(data.keys()):
    if('ground-truth' in name): continue
    d = data[name]
    if(d.use_exp_int==0): data_ground_truth = data_ground_truth_euler
    else:                 data_ground_truth = data_ground_truth_exp
    
    err = (norm(d.q - data_ground_truth.q) + norm(d.v - data_ground_truth.v)) / d.q.shape[0]
    err_per_time = np.array(norm(d.q - data_ground_truth.q, axis=0)) + \
                    np.array(norm(d.v - data_ground_truth.v, axis=0))
    print(name, 'Total error: %.2f'%np.log10(err))
    if(d.method_name not in total_err):
        mat_norm_expm[d.method_name] = []
        mat_mult_expm[d.method_name] = []
        total_err[d.method_name] = []
        ndt[d.method_name] = []
#    mat_norm_expm[d.method_name] += [np.mean(d.mat_norm_expm)]
#    mat_mult_expm[d.method_name] += [np.mean(d.mat_mult_expm)]
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
    
            
# PLOT THE CONTACT FORCES OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_FORCES):
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
        
    nc = len(conf.contact_frames)
    for (name, d) in data.items():
        (ff, ax) = plut.create_empty_figure(nc, 1)
        for i in range(nc):
            ax[i].plot(tt, norm(d.f[0:2,i,:], axis=0) / (1e-3+d.f[2,i,:]), alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Force X/Z [N]')
            leg = ax[i].legend()
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