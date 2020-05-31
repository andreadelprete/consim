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

class Empty:
    pass

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

print("".center(conf.LINE_WIDTH, '#'))
print(" Test Solo Trot C++ ".center(conf.LINE_WIDTH, '#'))
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
PLOT_INTEGRATION_ERRORS = 1
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 1

dt     = 0.002                      # controller and simulator time step
dt_ref = 0.010                      # time step of reference motion
unilateral_contacts = 1
compute_predicted_forces = False
PRINT_N = int(conf.PRINT_T/dt)

robot = loadSolo(False)
nq, nv = robot.nq, robot.nv

if conf.use_viewer:
    robot.initViewer(loadModel=True)
    robot.viewer.gui.createSceneWithFloor('world')
    robot.viewer.gui.setLightingMode('world/floor', 'OFF')

assert(np.floor(dt_ref/dt)==dt_ref/dt)

# load reference trajectories 
whichMotion = 'trot'
refX = np.load('../demo/references/'+whichMotion+'_reference_states.npy').squeeze()
refU = np.load('../demo/references/'+whichMotion+'_reference_controls.npy').squeeze() 
feedBack = np.load('../demo/references/'+whichMotion+'_feedback.npy').squeeze() 
q0, v0 = refX[0,:nq], refX[0,nq:]
N_SIMULATION = refU.shape[0]     

# TEMPORARY DEBUG CODE
N_SIMULATION = 20
q0[2] += 1.0         # make the robot fly
#q0[2] -= 15.37e-3   # ensure contact points are inside the ground at t=0

## Options 
#  1: pinocchio.Minverse()
#  2: pinocchio.aba()
#  3: Cholesky factorization 

whichFD = 2




def run_simulation(q0, v0, simu_params):
    ndt = simu_params['ndt']
    if(simu_params['use_exp_int']):
        simu = consim.build_exponential_simulator(dt, ndt, robot.model, robot.data,
                                    conf.K, conf.B, conf.mu, conf.anchor_slipping_method,
                                    compute_predicted_forces, whichFD)
    else:
        simu = consim.build_euler_simulator(dt, ndt, robot.model, robot.data,
                                        conf.K, conf.B, conf.mu, whichFD)
                                        
    cpts = []
    for cf in conf.contact_frames:
        if not robot.model.existFrame(cf):
            print(("ERROR: Frame", cf, "does not exist"))
        cpts += [simu.add_contact_point(cf, robot.model.getFrameId(cf), unilateral_contacts)]

    robot.forwardKinematics(q0)
    simu.reset_state(q0, v0, True)
            
    t = 0.0    
    q = np.zeros((nq, N_SIMULATION+1))*np.nan
    v = np.zeros((nv, N_SIMULATION+1))*np.nan
    u = np.zeros((nv, N_SIMULATION+1))
    f = np.zeros((3, len(conf.contact_frames), N_SIMULATION+1))
    p = np.zeros((3, len(conf.contact_frames), N_SIMULATION+1))
    dp = np.zeros((3, len(conf.contact_frames), N_SIMULATION+1))
    
    q[:,0] = np.copy(q0)
    v[:,0] = np.copy(v0)
    for ci, cp in enumerate(cpts):
        f[:,ci,0] = cp.f
        p[:,ci,0] = cp.x
        dp[:,ci,0] = cp.v
    print('p\n', 1e5*p[2,:,0].squeeze())
    
    try:
        time_start = time.time()
        for i in range(0, N_SIMULATION):
            for d in range(int(dt_ref/dt)):
                xref = interpolate_state(robot, refX[i], refX[i+1], dt*d/dt_ref)
                xact = np.concatenate([simu.get_q(), simu.get_v()])
                diff = state_diff(robot, xact, xref)
                u[6:,i] = refU[i] + feedBack[i].dot(diff)                 
                u[:,i] *= 0.0
                simu.step(u[:,i])
                
            q[:,i+1] = simu.get_q()
            v[:,i+1] = simu.get_v()
            
            for ci, cp in enumerate(cpts):
                f[:,ci,i+1] = cp.f
                p[:,ci,i+1] = cp.x
                dp[:,ci,i+1] = cp.v
            
            if(np.any(np.isnan(v[:,i+1])) or norm(v[:,i+1]) > 1e3):
                raise Exception("Time %.3f Velocities are too large: %.1f. Stop simulation."%(
                                t, norm(v[:,i+1])))
    
            if i % PRINT_N == 0:
                print("Time %.3f" % (t))                            
            t += dt
        print("Real-time factor:", t/(time.time() - time_start))
    except Exception as e:
        print(e)

    if conf.use_viewer:
        for i in range(0, N_SIMULATION):
            if(np.any(np.isnan(q[:,i]))):
                break
            time_start_viewer = time.time()
            robot.display(q[:,i])
            time_passed = time.time()-time_start_viewer
            if(time_passed<dt):
                time.sleep(dt-time_passed)
                
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
    data[name] = run_simulation(q0, v0, simu_params)

print("\nStart simulation ground truth")
data_ground_truth_exp = run_simulation(q0, v0, GROUND_TRUTH_EXP_SIMU_PARAMS)
data_ground_truth_euler = run_simulation(q0, v0, GROUND_TRUTH_EULER_SIMU_PARAMS)
data['ground-truth-exp'] = data_ground_truth_exp
data['ground-truth-euler'] = data_ground_truth_euler

# COMPUTE INTEGRATION ERRORS:
print('\n')
ndt = {}
total_err, err_max, err_traj = {}, {}, {}
for name in sorted(data.keys()):
    if('ground-truth' in name): continue
    d = data[name]
    if(d.use_exp_int==0): data_ground_truth = data_ground_truth_euler
    else:                 data_ground_truth = data_ground_truth_exp
    
    err = (norm(d.q - data_ground_truth.q) + norm(d.v - data_ground_truth.v)) / d.q.shape[0]
    err_per_time = np.array(norm(d.q - data_ground_truth.q, axis=0)) + \
                    np.array(norm(d.v - data_ground_truth.v, axis=0))
    err_peak = np.max(err_per_time)
    print(name, 'Total error: %.2f'%np.log10(err))
    if(d.method_name not in total_err):
        total_err[d.method_name] = []
        err_max[d.method_name] = []
        ndt[d.method_name] = []
    total_err[d.method_name] += [err]
    err_max[d.method_name] += [err_peak]
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
    ax.set_ylabel('Mean error norm')
    ax.set_xscale('log')
    ax.set_yscale('log')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    
    (ff, ax) = plut.create_empty_figure(1)
    j = 0
    for name in sorted(total_err.keys()):
        err = total_err[name]
        ax.plot(ndt[name], err_max[name], line_styles[j], alpha=0.7, label=name)
        j += 1
    ax.set_xlabel('Number of time steps')
    ax.set_ylabel('Max error norm')
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
    nc = len(conf.contact_frames)
    for (name, d) in data.items():
        (ff, ax) = plut.create_empty_figure(nc, 1)
        for i in range(nc):
            ax[i].plot(tt, norm(d.f[0:2,i,:], axis=0) / (1e-3+d.f[2,i,:]), alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Force X/Z [N]')
            leg = ax[i].legend()
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