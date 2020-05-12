import time
import matplotlib.pyplot as plt

from robot_simulator_exponential_integrator import RobotSimulator
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
print("Test Exponential Integrator with Quadruped Robot ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH, '#'))

SIMU_PARAMS = []
#SIMU_PARAMS += [{
#    'name': 'exp2',
#    'use_exp_int': 1,
#    'ndt': 2,
#    'sparse': 0
#}]
SIMU_PARAMS += [{
    'name': 'exp20',
    'use_exp_int': 1,
    'ndt': 20,
    'sparse': 0
}]
#SIMU_PARAMS += [{
#    'name': 'exp20',
#    'use_exp_int': 1,
#    'ndt': 20,
#    'sparse': 0
#}]
#SIMU_PARAMS += [{
#    'name': 'exp50',
#    'use_exp_int': 1,
#    'ndt': 50,
#    'sparse': 0
#}]
#SIMU_PARAMS += [{
#    'name': 'exp200',
#    'use_exp_int': 1,
#    'ndt': 200,
#    'sparse': 0
#}]
#SIMU_PARAMS += [{
#    'name': 'euler50',
#    'use_exp_int': 0,
#    'ndt': 50,
#    'sparse': 0
#}]
#SIMU_PARAMS += [{
#    'name': 'euler200',
#    'use_exp_int': 0,
#    'ndt': 200,
#    'sparse': 0
#}]
#SIMU_PARAMS += [{
#    'name': 'euler1000',
#    'use_exp_int': 0,
#    'ndt': 1000,
#    'sparse': 0
#}]

#SIMU_PARAMS += [{
#    'name': 'exp sparse',
#    'use_exp_int': 1,
#    'ndt': 1,
#    'sparse': 1
#}]

ASSUME_A_INVERTIBLE = 0
USE_CONTROLLER = 1
#ndt_force = 50
dt = 0.002                      # controller time step
T = 0.4

offset = np.array([0.0, -0.0, 0.0])
amp = np.array([0.0, 0.0, 0.05])
two_pi_f = 2*np.pi*np.array([0.0, .0, 2.0])

N_SIMULATION = int(T/dt)        # number of time steps simulated
PRINT_N = int(conf.PRINT_T/dt)
DISPLAY_N = int(conf.DISPLAY_T/dt)

solo = loadSolo()
nq, nv = solo.nq, solo.nv
simu = RobotSimulator(conf, solo, se3.JointModelFreeFlyer())
# simu = RobotSimulator(conf, solo, se3.JointModelFreeFlyer(), 'logStuff')  # With logger enabled
simu.assume_A_invertible = ASSUME_A_INVERTIBLE
#simu.ndt_force = ndt_force
q0, v0 = np.copy(simu.q), np.copy(simu.v)

for name in conf.contact_frames:
    simu.add_contact(name, conf.contact_normal, conf.K, conf.B)

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
    simu.init(q0, v0)
    t = 0.0
    ndt = simu_params['ndt']
    time_start = time.time()
    q = np.zeros((nq, N_SIMULATION+1))*np.nan
    v = np.zeros((nv, N_SIMULATION+1))*np.nan
    f = np.zeros((3*len(conf.contact_frames), N_SIMULATION+1))
    f_pred_int = np.zeros((3*len(conf.contact_frames), N_SIMULATION+1))
    f_inner = np.zeros((3*len(conf.contact_frames), N_SIMULATION*ndt))
    f_pred = np.zeros((3*len(conf.contact_frames), N_SIMULATION*ndt))
    dp = np.zeros((simu.nk, N_SIMULATION+1))
    dp_fd = np.zeros((simu.nk, N_SIMULATION+1))
    dJv = np.zeros((simu.nk, N_SIMULATION+1))
    dJv_fd = np.zeros((simu.nk, N_SIMULATION+1))
    
    q[:,0] = np.copy(simu.q)
    v[:,0] = np.copy(simu.v)
#    f[:,0] = np.copy(simu.f)
    
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

            # TMP: update control every 2 time steps
            if(i%2==0):
                u = invdyn.formulation.getActuatorForces(sol)
            #            dv_des = invdyn.formulation.getAccelerations(sol)
        else:
            invdyn.formulation.computeProblemData(t, q[:,i], v[:,i])
            #        robot.computeAllTerms(invdyn.data(), q, v)
            u = -0.03*conf.kp_posture*v[6:, i]

        q[:,i+1], v[:,i+1], f_i = simu.simulate(u, dt, ndt,
                                  simu_params['use_exp_int'],
                                  simu_params['sparse'])
        if(i+1 < N_SIMULATION):
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

    time_spent = time.time() - time_start
    print("Real-time factor:", t/time_spent, '\n')
    results = Empty()
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
    return results


# import cProfile
# cProfile.run('run_simulation(q0, v0)')
#Q, V = {}, {}
#forces = {}
#forces_log = {}
#ddp, ddp_fd = {}, {}
data = {}
for simu_params in SIMU_PARAMS:
    name = simu_params['name']
    print("\nStart simulation", name)
    data[name] = run_simulation(q0, v0, simu_params)
#    Q[name], V[name], forces[name], forces_log[name], ddp[name], ddp_fd[name] = run_simulation(q0, v0, simu_params)

# PLOT STUFF
line_styles = 3*['-', '--', '-.', ':']
tt = np.arange(0.0, (N_SIMULATION+1)*dt, dt)[:N_SIMULATION+1]

PLOT_UPSILON = 0
if(PLOT_UPSILON):
    np.set_printoptions(precision=2, linewidth=200, suppress=True)
    plt.matshow(simu.Upsilon)
    plt.colorbar()

    U = simu.Upsilon.copy()
    for i in range(U.shape[0]):
        U[i, :] /= U[i, i]
    print(U)
    plt.show()

#(ff, ax) = plut.create_empty_figure(2, 2)
#ax = ax.reshape(4)
#j = 0
#for (name, f) in forces.items():
#    for i in range(4):
#        ax[i].plot(tt, f[2+3*i, :], line_styles[j], alpha=0.7, label=name+' '+str(i))
#        ax[i].set_xlabel('Time [s]')
#        ax[i].set_ylabel('Force Z [N]')
#    j += 1
#    leg = ax[0].legend()
#    leg.get_frame().set_alpha(0.5)

for (name,d) in data.items():
   (ff, ax) = plut.create_empty_figure(2,2)
   ax = ax.reshape(4)
   tt_log = np.arange(d.f_pred.shape[1]) * T / d.f_pred.shape[1]
   for i in range(4):
       ax[i].plot(tt, d.f[2+3*i,:], ' o', markersize=8, label=name+' '+str(i))
       ax[i].plot(tt, d.f_pred_int[2+3*i,:], ' s', markersize=8, label=name+' pred int'+str(i))
       ax[i].plot(tt_log, d.f_pred[2+3*i,:], 'r v', markersize=6, label=name+' pred '+str(i))
       ax[i].plot(tt_log, d.f_inner[2+3*i,:], 'b x', markersize=6, label=name+' real '+str(i))
       ax[i].set_xlabel('Time [s]')
       ax[i].set_ylabel('Force Z [N]')
   leg = ax[0].legend()
   leg.get_frame().set_alpha(0.5)
   
#   f_pred_err = d.f_pred[:, (ndt-1)::ndt] - d.f[:,1:]
   f_pred_err = d.f_pred - d.f_inner
   print(name, 'Force pred err:', np.sum(np.abs(f_pred_err))/(f_pred_err.shape[0]*f_pred_err.shape[1]))


print('')
for (name, d) in data.items():
#    (ff, ax) = plut.create_empty_figure(2,2)
#    ax = ax.reshape(4)
#    for i in range(4):
#        ax[i].plot(tt, d.dp[2+3*i, :], line_styles[0], alpha=0.7, label=name+' dp '+str(i))
#        ax[i].plot(tt, d.dp_fd[2+3*i, :], line_styles[1], alpha=0.7, label=name+' dp fd '+str(i))
#        ax[i].set_xlabel('Time [s]')
#        ax[i].set_ylabel('Vel [m/s]')
#    leg = ax[0].legend()
#    leg.get_frame().set_alpha(0.5)
    print(name, "max dp error", np.max(np.abs(d.dp-d.dp_fd)))
    
print('')
for (name, d) in data.items():
#    (ff, ax) = plut.create_empty_figure(2,2)
#    ax = ax.reshape(4)
#    for i in range(4):
#        ax[i].plot(tt, d.dJv[2+3*i, :], line_styles[0], alpha=0.7, label=name+'dJv '+str(i))
#        ax[i].plot(tt, d.dJv_fd[2+3*i, :], line_styles[1], alpha=0.7, label=name+'dJv fd '+str(i))
#        ax[i].set_xlabel('Time [s]')
#        ax[i].set_ylabel('Acc [m/s]')
#    leg = ax[0].legend()
#    leg.get_frame().set_alpha(0.5)
    print(name, "max dJv error", np.max(np.abs(d.dJv-d.dJv_fd), axis=1))

#nplots = 1
#plot_offset = 0
#(ff, ax) = plut.create_empty_figure(nplots, 1)
#if(nplots==1):
#    ax = [ax]
#else:
#    ax = ax.reshape(nplots)
#j = 0
#for (name, q) in Q.items():
#    for i in range(nplots):
#        ax[i].plot(tt, q[plot_offset+i, :], line_styles[j], alpha=0.7, label=name+' '+str(i))
#        ax[i].set_xlabel('Time [s]')
#        ax[i].set_ylabel('q [rad]')
#    j += 1
#    leg = ax[0].legend()
#    leg.get_frame().set_alpha(0.5)

# (ff, ax) = plut.create_empty_figure(3,1)
# for i in range(3):
#    ax[i].plot(tt, com_pos[i,:], label='CoM '+str(i))
#    ax[i].plot(tt, com_pos_ref[i,:], 'r:', label='CoM Ref '+str(i))
#    ax[i].set_xlabel('Time [s]')
#    ax[i].set_ylabel('CoM [m]')
#    leg = ax[i].legend()
#    leg.get_frame().set_alpha(0.5)
#
# (f, ax) = plut.create_empty_figure(3,1)
# for i in range(3):
#    ax[i].plot(tt, com_vel[i,:], label='CoM Vel '+str(i))
#    ax[i].plot(tt, com_vel_ref[i,:], 'r:', label='CoM Vel Ref '+str(i))
#    ax[i].set_xlabel('Time [s]')
#    ax[i].set_ylabel('CoM Vel [m/s]')
#    leg = ax[i].legend()
#    leg.get_frame().set_alpha(0.5)
#
# (f, ax) = plut.create_empty_figure(3,1)
# for i in range(3):
#    ax[i].plot(tt, com_acc[i,:], label='CoM Acc '+str(i))
#    ax[i].plot(tt, com_acc_ref[i,:], 'r:', label='CoM Acc Ref '+str(i))
#    ax[i].plot(tt, com_acc_des[i,:], 'g--', label='CoM Acc Des '+str(i))
#    ax[i].set_xlabel('Time [s]')
#    ax[i].set_ylabel('CoM Acc [m/s^2]')
#    leg = ax[i].legend()
#    leg.get_frame().set_alpha(0.5)


plt.show()
