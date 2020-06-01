''' Test python simulator with Solo robot 
'''
import time
import matplotlib.pyplot as plt

from simulator import RobotSimulator
from tsid_quadruped import TsidQuadruped
import conf_solo_py as conf
from example_robot_data.robots_loader import loadSolo
import utils.plot_utils as plut

import numpy as np
from numpy import nan
from numpy.linalg import norm as norm

import pinocchio as se3

class Empty:
    pass


print("".center(conf.LINE_WIDTH, '#'))
print(" Test Solo ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH, '#'))

# parameters of the simulation to be tested
i_min = 0
i_max = 7
i_ground_truth = i_max+2

GROUND_TRUTH_SIMU_PARAMS = {
    'name': 'ground-truth %d'%(2**i_ground_truth),
    'method_name': 'ground-truth',
    'use_exp_int': 1,
    'ndt': 2**i_ground_truth,
}

SIMU_PARAMS = []

# EXPONENTIAL INTEGRATOR WITH STANDARD SETTINGS
#for i in range(i_min, i_max):
#    SIMU_PARAMS += [{
#        'name': 'exp%4d'%(2**i),
#        'method_name': 'exp',
#        'use_exp_int': 1,
#        'ndt': 2**i,
#    }]
    
# EXPONENTIAL INTEGRATOR, UPDATE MATRIX EXPONENTIAL EVERY FEW ITERATIONS
#for i in range(i_min, i_max):
#    for j in range(1,min(i,5)):
#        SIMU_PARAMS += [{
#            'name': 'exp%4d update%3d'%(2**i, 2**j),
#            'method_name': 'exp update%3d'%(2**j),
#            'use_exp_int': 1,
#            'ndt': 2**i,
#            'update_expm_N': 2**j
#        }]
    
# EXPONENTIAL INTEGRATOR, USE ONLY FIRST INTEGRAL
#for i in range(i_max):
#    SIMU_PARAMS += [{
#        'name': 'exp%4d n2i'%(2**i),
#        'method_name': 'exp-no-2nd-int',
#        'use_exp_int': 1,
#        'ndt': 2**i,
#        'use_second_integral': False
#    }]

# EXPONENTIAL INTEGRATOR, REDUCE NUMBER OF MATRIX MULTIPLICATIONS
#for i in range(2,3):
#    for j in range(0,7):
#        SIMU_PARAMS += [{
#            'name': 'exp%4d mmm-%d'%(2**i,j),
#            'method_name': 'exp mmm-%d'%j,
#            'use_exp_int': 1,
#            'ndt': 2**i,
#            'max_mat_mult': j,
#        }]

# EULER SIMULATOR with Cholesky
for i in range(5, i_max):
    SIMU_PARAMS += [{
        'name': 'euler%4d Chol'%(2**i),
        'method_name': 'euler Chol',
        'use_exp_int': 0,
        'ndt': 2**i,
    }]
    
# EULER SIMULATOR with ABA
for i in range(5, i_max):
    SIMU_PARAMS += [{
        'name': 'euler%4d ABA'%(2**i),
        'method_name': 'euler ABA',
        'use_exp_int': 0,
        'ndt': 2**i,
        'fwd_dyn_method': 'aba'
    }]
    
## EULER SIMULATOR with pinocchio::computeMinverse
for i in range(5, i_max):
    SIMU_PARAMS += [{
        'name': 'euler%4d pinMinv'%(2**i),
        'method_name': 'euler pinMinv',
        'use_exp_int': 0,
        'ndt': 2**i,
        'fwd_dyn_method': 'pinMinv'
    }]

PLOT_UPSILON = 0
PLOT_FORCES = 0
PLOT_BASE_POS = 0
PLOT_FORCE_PREDICTIONS = 0
PLOT_INTEGRATION_ERRORS = 1
PLOT_INTEGRATION_ERROR_TRAJECTORIES = 0
PLOT_MAT_MULT_EXPM = 0
PLOT_MAT_NORM_EXPM = 0

ASSUME_A_INVERTIBLE = 0
USE_CONTROLLER = 1
dt = 0.01                      # controller time step
T = 0.01

offset = np.array([0.0, -0.0, 0.0])
amp = np.array([0.0, 0.0, 0.05])
two_pi_f = 2*np.pi*np.array([0.0, .0, 2.0])

N_SIMULATION = int(T/dt)        # number of time steps simulated
PRINT_N = int(conf.PRINT_T/dt)

solo = loadSolo()
nq, nv = solo.nq, solo.nv
simu = RobotSimulator(conf, solo, se3.JointModelFreeFlyer())
# simu = RobotSimulator(conf, solo, se3.JointModelFreeFlyer(), 'logStuff')  # With logger enabled
simu.assume_A_invertible = ASSUME_A_INVERTIBLE
q0, v0 = np.copy(simu.q), np.copy(simu.v)

# DEBUG
#q0[2] += 1.0

for name in conf.contact_frames:
    simu.add_contact(name, conf.contact_normal, conf.K, conf.B, conf.mu)

invdyn = TsidQuadruped(conf, solo, q0, viewer=False)
robot = invdyn.robot

com_pos = np.empty((3, N_SIMULATION))*nan
com_vel = np.empty((3, N_SIMULATION))*nan
com_acc = np.empty((3, N_SIMULATION))*nan
com_pos_ref = np.empty((3, N_SIMULATION))*nan
com_vel_ref = np.empty((3, N_SIMULATION))*nan
com_acc_ref = np.empty((3, N_SIMULATION))*nan
# acc_des = acc_ref - Kp*pos_err - Kd*vel_err
com_acc_des = np.empty((3, N_SIMULATION))*nan
offset += invdyn.robot.com(invdyn.formulation.data())
two_pi_f_amp = two_pi_f * amp
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp
sampleCom = invdyn.trajCom.computeNext()

def run_simulation(q, v, simu_params):
    simu = RobotSimulator(conf, solo, se3.JointModelFreeFlyer())
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
    f = np.zeros((simu.nk, N_SIMULATION+1))
    f_pred_int = np.zeros((simu.nk, N_SIMULATION+1))
    f_inner = np.zeros((simu.nk, N_SIMULATION*ndt))
    f_pred = np.zeros((simu.nk, N_SIMULATION*ndt))
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
    
            if(USE_CONTROLLER):
                sampleCom.pos(offset + amp * np.sin(two_pi_f*t))
                sampleCom.vel(two_pi_f_amp * np.cos(two_pi_f*t))
                sampleCom.acc(-two_pi_f_squared_amp * np.sin(two_pi_f*t))
                invdyn.comTask.setReference(sampleCom)
    
                HQPData = invdyn.formulation.computeProblemData(t, q[:,i], v[:,i])
                sol = invdyn.solver.solve(HQPData)
                if(sol.status != 0):
                    print("[%d] QP problem could not be solved! Error code:" % (i), sol.status)
                    break
    
                u = invdyn.formulation.getActuatorForces(sol)
            else:
                invdyn.formulation.computeProblemData(t, q[:,i], v[:,i])
                #        robot.computeAllTerms(invdyn.data(), q, v)
                u = -0.03*conf.kp_posture*v[6:, i]
    
            q[:,i+1], v[:,i+1], f_i = simu.simulate(u, dt, ndt,
                                      simu_params['use_exp_int'])
            f[:, i+1] = f_i
            f_pred_int[:,i+1] = simu.f_pred_int        
            f_inner[:, i*ndt:(i+1)*ndt] = simu.f_inner
            f_pred[:, i*ndt:(i+1)*ndt] = simu.f_pred        
            dv = simu.dv
    
            com_pos[:, i] = invdyn.robot.com(invdyn.formulation.data())
            com_vel[:, i] = invdyn.robot.com_vel(invdyn.formulation.data())
            com_acc[:, i] = invdyn.comTask.getAcceleration(dv)
            com_pos_ref[:, i] = sampleCom.pos()
            com_vel_ref[:, i] = sampleCom.vel()
            com_acc_ref[:, i] = sampleCom.acc()
            com_acc_des[:, i] = invdyn.comTask.getDesiredAcceleration
            
            dp[:,i] = simu.debug_dp
            dp_fd[:,i] = simu.debug_dp_fd
            dJv[:,i] = simu.debug_dJv
            dJv_fd[:,i] = simu.debug_dJv_fd
            
            mat_mult_expm[i] = simu.expMatHelper.mat_mult
            mat_norm_expm[i] = simu.expMatHelper.mat_norm
    
            if i % PRINT_N == 0:
                print("Time %.3f" % (t))
    #            print("\tNormal forces:         ", f_i[2::3].T)
    #            if(USE_CONTROLLER):
    #                print("\tDesired normal forces: ")
    #                for contact in invdyn.contacts:
    #                    if invdyn.formulation.checkContact(contact.name, sol):
    #                        f_des = invdyn.formulation.getContactForce(contact.name, sol)
    #                        print("%4.1f" % (contact.getNormalForce(f_des)))
    #
    #                print("\n\ttracking err %s: %.3f" % (invdyn.comTask.name.ljust(20, '.'),
    #                                                     norm(invdyn.comTask.position_error, 2)))
    #                print("\t||v||: %.3f\t ||dv||: %.3f" % (norm(v, 2), norm(dv)))
    
            t += dt
    except Exception as e:
        print(e)
        print("ERROR WHILE RUNNING SIMULATION")
#        raise e

    time_spent = time.time() - time_start
    print("Real-time factor:", t/time_spent)
    results = Empty()
    for key in simu_params.keys():
        results.__dict__[key] = simu_params[key]
    results.q = q
    results.v = v
    results.f = f
    results.f_inner = f_inner
    results.f_pred = f_pred
    results.f_pred_int = f_pred_int
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
    
if(PLOT_MAT_MULT_EXPM):
    (ff, ax) = plut.create_empty_figure(1)
    j = 0
    for (name, err) in total_err.items():
        ax.plot(ndt[name], mat_mult_expm[name], line_styles[j], alpha=0.7, label=name)
        j += 1
    ax.set_xlabel('Number of time steps')
    ax.set_ylabel('Mean # mat mult in expm')
    ax.set_xscale('log')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)
    
if(PLOT_MAT_NORM_EXPM):
    (ff, ax) = plut.create_empty_figure(1)
    j = 0
    for (name, err) in total_err.items():
        ax.plot(ndt[name], mat_norm_expm[name], line_styles[j], alpha=0.7, label=name)
        j += 1
    ax.set_xlabel('Number of time steps')
    ax.set_ylabel('Mean mat norm in expm')
    ax.set_xscale('log')
    ax.set_yscale('log')
    leg = ax.legend()
    if(leg): leg.get_frame().set_alpha(0.5)

# PLOT THE CONTACT FORCES OF ALL INTEGRATION METHODS ON THE SAME PLOT
if(PLOT_FORCES):
    (ff, ax) = plut.create_empty_figure(2, 2)
    ax = ax.reshape(4)
    j = 0
    for (name, d) in data.items():
        for i in range(4):
            ax[i].plot(tt, d.f[2+3*i, :], line_styles[j], alpha=0.7, label=name)
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Force Z [N]')
        j += 1
        leg = ax[0].legend()
        if(leg): leg.get_frame().set_alpha(0.5)

# FOR EACH INTEGRATION METHOD PLOT THE FORCE PREDICTIONS
if(PLOT_FORCE_PREDICTIONS):
    for (name,d) in data.items():
       (ff, ax) = plut.create_empty_figure(2,2)
       ax = ax.reshape(4)
       tt_log = np.arange(d.f_pred.shape[1]) * T / d.f_pred.shape[1]
       for i in range(4):
           ax[i].plot(tt, d.f[2+3*i,:], ' o', markersize=8, label=name)
           ax[i].plot(tt, d.f_pred_int[2+3*i,:], ' s', markersize=8, label=name+' pred int')
           ax[i].plot(tt_log, d.f_pred[2+3*i,:], 'r v', markersize=6, label=name+' pred ')
           ax[i].plot(tt_log, d.f_inner[2+3*i,:], 'b x', markersize=6, label=name+' real ')
           ax[i].set_xlabel('Time [s]')
           ax[i].set_ylabel('Force Z [N]')
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
        
#print('')
#for (name, d) in data.items():
#    (ff, ax) = plut.create_empty_figure(2,2)
#    ax = ax.reshape(4)
#    for i in range(4):
#        ax[i].plot(tt, d.dp[2+3*i, :], line_styles[0], alpha=0.7, label=name+' dp '+str(i))
#        ax[i].plot(tt, d.dp_fd[2+3*i, :], line_styles[1], alpha=0.7, label=name+' dp fd '+str(i))
#        ax[i].set_xlabel('Time [s]')
#        ax[i].set_ylabel('Vel [m/s]')
#    leg = ax[0].legend()
#    leg.get_frame().set_alpha(0.5)
#    print(name, "max dp error", np.max(np.abs(d.dp-d.dp_fd)))
    
#print('')
#for (name, d) in data.items():
#    (ff, ax) = plut.create_empty_figure(2,2)
#    ax = ax.reshape(4)
#    for i in range(4):
#        ax[i].plot(tt, d.dJv[2+3*i, :], line_styles[0], alpha=0.7, label=name+'dJv '+str(i))
#        ax[i].plot(tt, d.dJv_fd[2+3*i, :], line_styles[1], alpha=0.7, label=name+'dJv fd '+str(i))
#        ax[i].set_xlabel('Time [s]')
#        ax[i].set_ylabel('Acc [m/s]')
#    leg = ax[0].legend()
#    leg.get_frame().set_alpha(0.5)
#    print(name, "max dJv error", np.max(np.abs(d.dJv-d.dJv_fd), axis=1))

if(PLOT_UPSILON):
    np.set_printoptions(precision=2, linewidth=200, suppress=True)
    plt.matshow(simu.Upsilon)
    plt.colorbar()

    U = simu.Upsilon.copy()
    for i in range(U.shape[0]):
        U[i, :] /= U[i, i]
    print(U)

plt.show()
