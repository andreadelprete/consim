''' Test cpp simulator with point-mass sliding on a flat floor 
'''
import time, os 
import consim 
from consim_py.simulator import RobotSimulator
import numpy as np
from numpy.linalg import norm as norm

import matplotlib.pyplot as plt 
import consim_py.utils.plot_utils as plut
import sliding_point_conf as conf
import pinocchio as pin 
from pinocchio.robot_wrapper import RobotWrapper 


class Empty:
    pass


print("".center(conf.LINE_WIDTH, '#'))
print(" Test Sliding Point Mass C++ VS Python".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH, '#'))

# parameters of the simulation to be tested
i_min = 1
i_max = i_min+1
i_ground_truth = i_max+2

GROUND_TRUTH_EXP_SIMU_PARAMS = {
    'name': 'ground-truth %d'%(2**i_ground_truth),
    'method_name': 'ground-truth-exp',
    'use_exp_int': 1,
    'ndt': 2**i_ground_truth
}

SIMU_PARAMS = []

# EXPONENTIAL INTEGRATOR WITH STANDARD SETTINGS
for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'exp%4d'%(2**i),
        'method_name': 'exp',
        'use_exp_int': 1,
        'ndt': 2**i,
        'forward_dyn_method': 1
    }]

for i in range(i_min, i_max):
    SIMU_PARAMS += [{
        'name': 'euler%4d'%20,
        'method_name': 'euler',
        'use_exp_int': 0,
        'ndt': 20,
        'forward_dyn_method': 1
    }]

    
PLOT_FORCES = 1
PLOT_SLIPPING = 0
PLOT_BASE_POS = 1
PLOT_INTEGRATION_ERRORS = 0
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 0

RESET_STATE_ON_GROUND_TRUTH = 0  # reset the state of the system on the ground truth
dt     = 0.002                      # controller and simulator time step
dt_ref = 0.010                      # time step of reference motion
unilateral_contacts = 1
compute_predicted_forces = False
PRINT_N = int(conf.PRINT_T/dt)

exp_max_mul = 100 
int_max_mul = 100 

urdf_path = os.path.abspath('../../models/urdf/free_flyer.urdf')
mesh_path = os.path.abspath('../../models')
robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_path], pin.JointModelFreeFlyer()) 
print(" RobotWrapper Object Created Successfully ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH, '#'))
nq, nv = robot.nq, robot.nv

if conf.use_viewer:
    robot.initViewer(loadModel=True)
    robot.viewer.gui.createSceneWithFloor('world')
    robot.viewer.gui.setLightingMode('world/floor', 'OFF')

assert(np.floor(dt_ref/dt)==dt_ref/dt)

### set a constant pushing force 
tau = np.zeros(robot.nv)
tau[0] = 4. 
tau[2] = -0.19 # total fz to 10 N
q0, v0 = conf.q0, conf.v0

# # TEMPORARY DEBUG CODE
N_SIMULATION = 100


def run_simulation_cpp(q0, v0, simu_params, ground_truth):
    ndt = simu_params['ndt']
    
    try:
        forward_dyn_method = simu_params['forward_dyn_method']
    except:
        #  1: pinocchio.Minverse()
        #  2: pinocchio.aba()
        #  3: Cholesky factorization 
        forward_dyn_method = 1
        
    if(simu_params['use_exp_int']):
        simu_cpp = consim.build_exponential_simulator(dt, ndt, robot.model, robot.data,
                                    conf.K, conf.B, conf.mu, conf.anchor_slipping_method,
                                    compute_predicted_forces, forward_dyn_method, exp_max_mul, int_max_mul)
    else:
        simu_cpp = consim.build_euler_simulator(dt, ndt, robot.model, robot.data,
                                        conf.K, conf.B, conf.mu, forward_dyn_method)
                                        
    cpts = []
    for cf in conf.contact_frames:
        if not robot.model.existFrame(cf):
            print(("ERROR: Frame", cf, "does not exist"))
        cpts += [simu_cpp.add_contact_point(cf, robot.model.getFrameId(cf), unilateral_contacts)]

    simu_cpp.reset_state(q0, v0, True)
            
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
    
    results.q[:,0] = np.copy(q0)
    results.v[:,0] = np.copy(v0)
    for ci, cp in enumerate(cpts):
        results.f[:,ci,0] = cp.f
        results.p[:,ci,0] = cp.x
        results.p0[:,ci,0] = cp.x_anchor
        results.dp[:,ci,0] = cp.v
        results.slipping[ci,0] = cp.slipping
        results.active[ci,0] = cp.active
    
    try:
        time_start = time.time()
        for i in range(0, N_SIMULATION):                    
            for d in range(int(dt_ref/dt)):
                results.u[:,i] = tau.copy()                
                simu_cpp.step(results.u[:,i])
                
            results.q[:,i+1] = simu_cpp.get_q()
            results.v[:,i+1] = simu_cpp.get_v()
            
            for ci, cp in enumerate(cpts):
                results.f[:,ci,i+1] = cp.f
                results.p[:,ci,i+1] = cp.x
                results.p0[:,ci,i+1] = cp.x_anchor
                results.dp[:,ci,i+1] = cp.v
                results.slipping[ci,i+1] = cp.slipping
                results.active[ci,i+1] = cp.active
            
            if(np.any(np.isnan(results.v[:,i+1])) or norm(results.v[:,i+1]) > 1e3):
                raise Exception("Time %.3f Velocities are too large: %.1f. Stop simulation."%(
                                t, norm(results.v[:,i+1])))
    
            if i % PRINT_N == 0:
                print("Time %.3f" % (t))                            
            t += dt_ref
        print("Real-time factor:", t/(time.time() - time_start))
    except Exception as e:
        print(e)

    if conf.use_viewer:
        for i in range(0, N_SIMULATION):
            if(np.any(np.isnan(results.q[:,i]))):
                break
            time_start_viewer = time.time()
            robot.display(results.q[:,i])
            time_passed = time.time()-time_start_viewer
            if(time_passed<dt):
                time.sleep(dt-time_passed)
                    
    for key in simu_params.keys():
        results.__dict__[key] = simu_params[key]

    return results
    
def run_simulation_py(q0, v0, simu_params, ground_truth):
    ndt = simu_params['ndt']
        
    simu_py = RobotSimulator(conf, robot)
    for name in conf.contact_frames:
        simu_py.add_contact(name, conf.contact_normal, conf.K, conf.B, conf.mu)
    simu_py.init(q0, v0, p0=conf.p0)
            
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
    
    results.q[:,0] = np.copy(q0)
    results.v[:,0] = np.copy(v0)
    for ci, cn in enumerate(conf.contact_frames):
        results.f[:,ci,0] = simu_py.contacts[ci].f.copy()
    
    try:
        time_start = time.time()
        for i in range(0, N_SIMULATION):
            for d in range(int(dt_ref/dt)):
                results.u[:,i] = tau.copy()
                simu_py.simulate(results.u[:,i], dt, ndt, simu_params['use_exp_int'])
                
            results.q[:,i+1] = simu_py.q
            results.v[:,i+1] = simu_py.v
            # collect the contact forces 
            for ci, cn in enumerate(conf.contact_frames):
                results.f[:,ci,i+1] = simu_py.contacts[ci].f.copy()

            
            if(np.any(np.isnan(results.v[:,i+1])) or norm(results.v[:,i+1]) > 1e3):
                raise Exception("Time %.3f Velocities are too large: %.1f. Stop simulation."%(
                                t, norm(results.v[:,i+1])))
    
            if i % PRINT_N == 0:
                print("Time %.3f" % (t))                            
            t += dt_ref
        print("Real-time factor:", t/(time.time() - time_start))
    except Exception as e:
        print(e)
#        raise e

    if conf.use_viewer:
        for i in range(0, N_SIMULATION):
            if(np.any(np.isnan(results.q[:,i]))):
                break
            time_start_viewer = time.time()
            robot.display(results.q[:,i])
            time_passed = time.time()-time_start_viewer
            if(time_passed<dt):
                time.sleep(dt-time_passed)
                    
    for key in simu_params.keys():
        results.__dict__[key] = simu_params[key]

    return results

data = {}
print("\nStart simulation ground truth")
data_ground_truth_exp = run_simulation_py(q0, v0, GROUND_TRUTH_EXP_SIMU_PARAMS, None)
data['ground-truth-exp'] = data_ground_truth_exp
 
for simu_params in SIMU_PARAMS:
    name = simu_params['name']
    print("\nStart simulation python", name)
    data[name+' py'] = run_simulation_py(q0, v0, simu_params, data_ground_truth_exp)
    print("\nStart simulation c++", name)
    data[name+' cpp'] = run_simulation_cpp(q0, v0, simu_params, data_ground_truth_exp)


# COMPUTE INTEGRATION ERRORS:
print('\n')
ndt = {}
total_err, err_max, err_traj = {}, {}, {}
for name in sorted(data.keys()):
    if('ground-truth' in name): continue
    d = data[name]
    data_ground_truth = data_ground_truth_exp
    
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
tt = np.arange(0.0, (N_SIMULATION+1)*dt_ref, dt_ref)[:N_SIMULATION+1]


### plot normal contact force 
if(PLOT_FORCES):        
    nc = len(conf.contact_frames)
    plt.figure("Normal Contact Forces")
    j = 0
    for (name, d) in data.items():
        for i in range(nc):
            plt.plot(tt, d.f[2,i,:],  line_styles[j], alpha=0.7, label=name)
        j+= 1 
    plt.xlabel('Time [s]')
    plt.ylabel('Force Z [N]')
    leg = plt.legend()
    if(leg): leg.get_frame().set_alpha(0.5)

    nc = len(conf.contact_frames)
    plt.figure("Tangent Contact Forces")
    j = 0
    for (name, d) in data.items():
        for i in range(nc):
            plt.plot(tt, np.sqrt(d.f[0,i,:]**2 + d.f[1,i,:]**2 ),  line_styles[j], alpha=0.7, label=name)
        j+= 1 
    plt.xlabel('Time [s]')
    plt.ylabel('Force Tangent [N]')
    leg = plt.legend()
    if(leg): leg.get_frame().set_alpha(0.5)

### plot contact position 
directions = ['X', 'Y', 'Z']
if(PLOT_BASE_POS):
    for di, dn in enumerate(directions):
        plt.figure("Ball %s position "%dn)
        j = 0
        for (name, d) in data.items():
            plt.plot(tt, d.q[di,:],  line_styles[j], alpha=0.7, label=name)
            j+= 1 
        plt.xlabel('Time [s]')
        plt.ylabel('Position %s [m]'%dn)
        leg = plt.legend()
        if(leg): leg.get_frame().set_alpha(0.5)





# # PLOT INTEGRATION ERRORS
# if(PLOT_INTEGRATION_ERRORS):
#     (ff, ax) = plut.create_empty_figure(1)
#     j = 0
#     for name in sorted(total_err.keys()):
#         err = total_err[name]
#         ax.plot(ndt[name], total_err[name], line_styles[j], alpha=0.7, label=name)
#         j += 1
#     ax.set_xlabel('Number of time steps')
#     ax.set_ylabel('Mean error norm')
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     leg = ax.legend()
#     if(leg): leg.get_frame().set_alpha(0.5)
    
#     (ff, ax) = plut.create_empty_figure(1)
#     j = 0
#     for name in sorted(total_err.keys()):
#         err = total_err[name]
#         ax.plot(ndt[name], err_max[name], line_styles[j], alpha=0.7, label=name)
#         j += 1
#     ax.set_xlabel('Number of time steps')
#     ax.set_ylabel('Max error norm')
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     leg = ax.legend()
#     if(leg): leg.get_frame().set_alpha(0.5)
    
# if(PLOT_INTEGRATION_ERROR_TRAJECTORIES):
#     (ff, ax) = plut.create_empty_figure(1)
#     j = 0
#     for name in sorted(err_traj.keys()):
#         err = err_traj[name]
#         ax.plot(tt, err_traj[name], line_styles[j], alpha=0.7, label=name)
#         j += 1
#     ax.set_xlabel('Time [s]')
#     ax.set_ylabel('Error norm')
#     ax.set_yscale('log')
#     leg = ax.legend()
#     if(leg): leg.get_frame().set_alpha(0.5)
    
            
# # PLOT THE CONTACT FORCES OF ALL INTEGRATION METHODS ON THE SAME PLOT
# if(PLOT_FORCES):        
#     nc = len(conf.contact_frames)
#     for (name, d) in data.items():
#         (ff, ax) = plut.create_empty_figure(nc, 1)
#         for i in range(nc):
#             ax[i].plot(tt, norm(d.f[0:2,i,:], axis=0) / (1e-3+d.f[2,i,:]), alpha=0.7, label=name)
#             ax[i].set_xlabel('Time [s]')
#             ax[i].set_ylabel('Force X/Z [N]')
#             leg = ax[i].legend()
#             if(leg): leg.get_frame().set_alpha(0.5)
            
# # PLOT THE SLIPPING FLAG OF ALL INTEGRATION METHODS ON THE SAME PLOT
# if(PLOT_SLIPPING):
#     nc = len(conf.contact_frames)
#     (ff, ax) = plut.create_empty_figure(nc, 1)
#     for (name, d) in data.items():        
#         for i in range(nc):
#             ax[i].plot(tt, d.slipping[i,:], alpha=0.7, label=name)
#             ax[i].set_xlabel('Time [s]')
#     ax[0].set_ylabel('Contact Slipping Flag')
#     leg = ax[0].legend()
#     if(leg): leg.get_frame().set_alpha(0.5)

#     (ff, ax) = plut.create_empty_figure(nc, 1)
#     for (name, d) in data.items():
#         for i in range(nc):
#             ax[i].plot(tt, d.active[i,:], alpha=0.7, label=name)
#             ax[i].set_xlabel('Time [s]')
#     ax[0].set_ylabel('Contact Active Flag')
#     leg = ax[0].legend()
#     if(leg): leg.get_frame().set_alpha(0.5)

       
# # PLOT THE JOINT ANGLES OF ALL INTEGRATION METHODS ON THE SAME PLOT
# if(PLOT_BASE_POS):
#     (ff, ax) = plut.create_empty_figure(3)
#     ax = ax.reshape(3)
#     j = 0
#     for (name, d) in data.items():
#         for i in range(3):
#             ax[i].plot(tt, d.q[i, :], line_styles[j], alpha=0.7, label=name)
#             ax[i].set_xlabel('Time [s]')
#             ax[i].set_ylabel('Base pos [m]')
#         j += 1
#         leg = ax[0].legend()
#         if(leg): leg.get_frame().set_alpha(0.5)
        
plt.show()