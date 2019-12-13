import time
import matplotlib.pyplot as plt

from robot_simulator_exponential_integrator import RobotSimulator
from tsid_quadruped import TsidQuadruped
import conf_solo_py as conf
from example_robot_data.robots_loader import loadSolo
import utils.plot_utils as plut

import numpy as np
import numpy.matlib as matlib
from numpy import nan
from numpy.linalg import norm as norm

import pinocchio as se3
se3.setNumpyType(np.matrix)


print("".center(conf.LINE_WIDTH, '#'))
print("Test Exponential Integrator with Quadruped Robot ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH, '#'), '\n')

simu_params_standard = {
    'name': 'euler',
    'use_exp_int': 0,
    'ndt': 20,
    'sparse': 0
}
simu_params_exp_int = {
    'name': 'exp',
    'use_exp_int': 1,
    'ndt': 1,
    'sparse': 0
}
simu_params_exp_int_sparse = {
    'name': 'exp sparse',
    'use_exp_int': 1,
    'ndt': 1,
    'sparse': 1
}
SIMU_PARAMS = [simu_params_standard, simu_params_exp_int]
ASSUME_A_INVERTIBLE = 0
USE_CONTROLLER = 1
ndt_force = simu_params_standard['ndt']
dt = 0.001                      # controller time step
T = 1

offset = np.matrix([0.0, -0.0, 0.0]).T
amp = np.matrix([0.0, 0.0, 0.05]).T
two_pi_f = 2*np.pi*np.matrix([0.0, .0, 2.0]).T

N_SIMULATION = int(T/dt)        # number of time steps simulated
PRINT_N = int(conf.PRINT_T/dt)
DISPLAY_N = int(conf.DISPLAY_T/dt)

solo = loadSolo()

simu = RobotSimulator(conf, solo, se3.JointModelFreeFlyer())
simu.assume_A_invertible = ASSUME_A_INVERTIBLE
simu.ndt_force = ndt_force
q0, v0 = np.copy(simu.q), np.copy(simu.v)

for name in conf.contact_frames:
    simu.add_contact(name, conf.contact_normal, conf.K, conf.B)

invdyn = TsidQuadruped(conf, solo, q0, viewer=False)
robot = invdyn.robot

com_pos = matlib.empty((3, N_SIMULATION))*nan
com_vel = matlib.empty((3, N_SIMULATION))*nan
com_acc = matlib.empty((3, N_SIMULATION))*nan
com_pos_ref = matlib.empty((3, N_SIMULATION))*nan
com_vel_ref = matlib.empty((3, N_SIMULATION))*nan
com_acc_ref = matlib.empty((3, N_SIMULATION))*nan
# acc_des = acc_ref - Kp*pos_err - Kd*vel_err
com_acc_des = matlib.empty((3, N_SIMULATION))*nan
offset += invdyn.robot.com(invdyn.formulation.data())
two_pi_f_amp = np.multiply(two_pi_f, amp)
two_pi_f_squared_amp = np.multiply(two_pi_f, two_pi_f_amp)
sampleCom = invdyn.trajCom.computeNext()


def run_simulation(q, v, simu_params):
    simu.init(q0, v0)
    t = 0.0
    time_start = time.time()
    f = np.zeros((3*len(conf.contact_frames), N_SIMULATION))
    f_log = np.zeros((3*len(conf.contact_frames), N_SIMULATION*ndt_force))
    for i in range(0, N_SIMULATION):

        if(USE_CONTROLLER):
            sampleCom.pos(offset + np.multiply(amp, matlib.sin(two_pi_f*t)))
            sampleCom.vel(np.multiply(two_pi_f_amp, matlib.cos(two_pi_f*t)))
            sampleCom.acc(np.multiply(two_pi_f_squared_amp, -matlib.sin(two_pi_f*t)))
            invdyn.comTask.setReference(sampleCom)

            HQPData = invdyn.formulation.computeProblemData(t, q, v)
            sol = invdyn.solver.solve(HQPData)
            if(sol.status != 0):
                print("[%d] QP problem could not be solved! Error code:" % (i), sol.status)
                break

            u = invdyn.formulation.getActuatorForces(sol)
#            dv_des = invdyn.formulation.getAccelerations(sol)
        else:
            invdyn.formulation.computeProblemData(t, q, v)
    #        robot.computeAllTerms(invdyn.data(), q, v)
            u = -0.01*conf.kp_posture*v[6:, 0]

        q, v, f_i = simu.simulate(u, dt, simu_params['ndt'],
                                  simu_params['use_exp_int'],
                                  simu_params['sparse'])
        if(i+1 < N_SIMULATION):
            f[:, i+1] = f_i.A1
        f_log[:, i*ndt_force:(i+1)*ndt_force] = simu.f_log
        dv = simu.dv

        com_pos[:, i] = invdyn.robot.com(invdyn.formulation.data())
        com_vel[:, i] = invdyn.robot.com_vel(invdyn.formulation.data())
        com_acc[:, i] = invdyn.comTask.getAcceleration(dv)
        com_pos_ref[:, i] = sampleCom.pos()
        com_vel_ref[:, i] = sampleCom.vel()
        com_acc_ref[:, i] = sampleCom.acc()
        com_acc_des[:, i] = invdyn.comTask.getDesiredAcceleration

        if i % PRINT_N == 0:
            print("Time %.3f" % (t))
            print("\tNormal forces:         ", f_i[2::3].T)
            if(USE_CONTROLLER):
                print("\tDesired normal forces: ")
                for contact in invdyn.contacts:
                    if invdyn.formulation.checkContact(contact.name, sol):
                        f_des = invdyn.formulation.getContactForce(contact.name, sol)
                        print("%4.1f" % (contact.getNormalForce(f_des)))

                print("\n\ttracking err %s: %.3f" % (invdyn.comTask.name.ljust(20, '.'),
                                                     norm(invdyn.comTask.position_error, 2)))
                print("\t||v||: %.3f\t ||dv||: %.3f" % (norm(v, 2), norm(dv)))

        t += dt

    time_spent = time.time() - time_start
    print("Real-time factor:", t/time_spent)
    return f, f_log


# import cProfile
# cProfile.run('run_simulation(q0, v0)')
forces = {}
forces_log = {}
for simu_params in SIMU_PARAMS:
    name = simu_params['name']
    forces[name], forces_log[name] = run_simulation(q0, v0, simu_params)

# PLOT STUFF
tt = np.arange(0.0, N_SIMULATION*dt, dt)
tt_log = np.arange(0.0, N_SIMULATION*dt, dt/ndt_force)

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

(ff, ax) = plut.create_empty_figure(2, 2)
ax = ax.reshape(4)
for (name, f) in forces.items():
    for i in range(4):
        ax[i].plot(tt, f[2+3*i, :].squeeze(), '--',   label=name+' '+str(i))
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('Force Z [N]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)

# for ((name,f),(nname,f_log)) in zip(forces.iteritems(),
#       forces_log.iteritems()):
#    (ff, ax) = plut.create_empty_figure(2,2)
#    ax = ax.reshape(4)
#    for i in range(4):
#        ax[i].plot(tt, f[2+3*i,:].squeeze(), '--',   label=name+' '+str(i))
#        ax[i].plot(tt_log, f_log[2+3*i,:].squeeze(), '--',
#       label=name+' '+str(i))
#        ax[i].set_xlabel('Time [s]')
#        ax[i].set_ylabel('Force Z [N]')
#    leg = ax[0].legend()
#    leg.get_frame().set_alpha(0.5)

# (ff, ax) = plut.create_empty_figure(3,1)
# for i in range(3):
#    ax[i].plot(tt, com_pos[i,:].A1, label='CoM '+str(i))
#    ax[i].plot(tt, com_pos_ref[i,:].A1, 'r:', label='CoM Ref '+str(i))
#    ax[i].set_xlabel('Time [s]')
#    ax[i].set_ylabel('CoM [m]')
#    leg = ax[i].legend()
#    leg.get_frame().set_alpha(0.5)
#
# (f, ax) = plut.create_empty_figure(3,1)
# for i in range(3):
#    ax[i].plot(tt, com_vel[i,:].A1, label='CoM Vel '+str(i))
#    ax[i].plot(tt, com_vel_ref[i,:].A1, 'r:', label='CoM Vel Ref '+str(i))
#    ax[i].set_xlabel('Time [s]')
#    ax[i].set_ylabel('CoM Vel [m/s]')
#    leg = ax[i].legend()
#    leg.get_frame().set_alpha(0.5)
#
# (f, ax) = plut.create_empty_figure(3,1)
# for i in range(3):
#    ax[i].plot(tt, com_acc[i,:].A1, label='CoM Acc '+str(i))
#    ax[i].plot(tt, com_acc_ref[i,:].A1, 'r:', label='CoM Acc Ref '+str(i))
#    ax[i].plot(tt, com_acc_des[i,:].A1, 'g--', label='CoM Acc Des '+str(i))
#    ax[i].set_xlabel('Time [s]')
#    ax[i].set_ylabel('CoM Acc [m/s^2]')
#    leg = ax[i].legend()
#    leg.get_frame().set_alpha(0.5)

if(1):
    plt.show()
