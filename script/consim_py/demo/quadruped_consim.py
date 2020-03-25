# constructs and simulates the drop of a point mass using both Euler and Exponential Simulators 
import numpy as np 
import pinocchio as pin 
import consim 
from pinocchio.robot_wrapper import RobotWrapper
import os, sys
from os.path import dirname, join
import matplotlib.pyplot as plt 
import utils.plot_utils as plut
import time
import matplotlib.pyplot as plt

from tsid_quadruped import TsidQuadruped
import conf_solo_py as conf
from example_robot_data.robots_loader import loadSolo

import numpy.matlib as matlib
from numpy import nan
from numpy.linalg import norm as norm

pin.setNumpyType(np.matrix)




USE_CONTROLLER = True 
# parameters used for CoM Sinusoid 
offset = np.matrix([0.0, -0.0, 0.0]).T
amp = np.matrix([0.0, 0.0, 0.05]).T
two_pi_f = 2*np.pi*np.matrix([0.0, .0, 1.0]).T
controller_dt = 5.e-3 

if __name__=="__main__":
    # simulation parameters 
    simu_params = []
    simu_params += [{'name': 'euler 100',
                    'type': 'euler', 
                    'ndt': 100}]
    simu_params += [{'name': 'euler 10',
                    'type': 'euler', 
                    'ndt': 10}]
    simu_params += [{'name': 'exponential 100',
                    'type': 'exponential', 
                    'ndt': 100}]
    simu_params += [{'name': 'exponential 10',
                    'type': 'exponential', 
                    'ndt': 10}]
    simu_params += [{'name': 'exponential 1',
                    'type': 'exponential', 
                    'ndt': 1}]
    

    line_styles = ['-', '--', '-.', ':']
    i_ls = 0
    
    mu = 0.3        # friction coefficient
    isSparse = False 
    isInvertible = False
    unilateral_contacts = True 
    K = 1e5
    B = 3e2
    T = 1 #  2 second simution  
    dt = 1.e-3 

    N_SIMULATION = int(T/dt)        # number of time steps simulated
    PRINT_N = int(conf.PRINT_T/dt)
    DISPLAY_N = int(conf.DISPLAY_T/dt)

    # PLOT STUFF
    tt = np.arange(0.0, N_SIMULATION*dt + dt, dt)

    # load robot 
    robot = loadSolo()
    print(" Solo Loaded Successfully ".center(conf.LINE_WIDTH, '#'))

    q0 = conf.q0
    # lower base height such that feet start below contact surface
    # this is needed for bilateral contacts 
    # q0[2] -=  .01195 # (.012 + 1.e-4)   
    v0 = np.zeros(robot.nv) [:,None]
 
    # loop over simulations     
    for simu_param in simu_params:
        ndt = simu_param['ndt']
        name = simu_param['name']
        simu_type = simu_param['type']
        # build the simulator 
        if(simu_type=='exponential'):
            sim = consim.build_exponential_simulator(dt, ndt, robot.model, robot.data,
                                        K, B ,K, B, mu, mu, isSparse, isInvertible)
        else:
            sim = consim.build_euler_simulator(dt, ndt, robot.model, robot.data,
                                            K, B ,K, B, mu, mu)
        
        # add the  contact points 
        cpts = [] # list of all contact points
        for cname in conf.contact_frames:
            if not robot.model.existFrame(cname):
                print(("ERROR: Frame", cname, "does not exist"))
            cpts += [sim.add_contact_point(robot.model.getFrameId(cname), unilateral_contacts)]
        print(" %s Contact Points Added ".center(conf.LINE_WIDTH, '-')%(len(cpts)))

        # inverse dynamics controller 
        invdyn = TsidQuadruped(conf, robot, q0, viewer=False)
        print(" TSID Initialized Successfully ".center(conf.LINE_WIDTH, '#'))


        # trajectory log 
        com_pos = np.empty((N_SIMULATION, 3))*nan
        com_vel = np.empty((N_SIMULATION, 3))*nan
        com_acc = np.empty((N_SIMULATION, 3))*nan
        com_pos_ref = np.empty((N_SIMULATION, 3))*nan
        com_vel_ref = np.empty((N_SIMULATION, 3))*nan
        com_acc_ref = np.empty((N_SIMULATION, 3))*nan
        # acc_des = acc_ref - Kp*pos_err - Kd*vel_err
        com_acc_des = np.empty((N_SIMULATION, 3))*nan
        # ConSim log 
        sim_f = np.empty((N_SIMULATION+1,len(conf.contact_frames),3))*nan
        sim_q = np.empty((N_SIMULATION+1,robot.nq))*nan
        contact_x = np.empty((N_SIMULATION+1,len(conf.contact_frames),3))*nan
        contact_v = np.empty((N_SIMULATION+1,len(conf.contact_frames),3))*nan

        q = [q0]
        sim_q[0,:] = np.resize(q[-1], robot.nq)
        v = [v0]

    
        offset += invdyn.robot.com(invdyn.formulation.data())
        two_pi_f_amp = np.multiply(two_pi_f, amp)
        two_pi_f_squared_amp = np.multiply(two_pi_f, two_pi_f_amp)
       

        sampleCom = invdyn.trajCom.computeNext()

        # reset simulator 
        pin.framesForwardKinematics(robot.model, robot.data, q0)
        sim.reset_state(q0, v0, True)
        print('Reset state done '.center(conf.LINE_WIDTH, '-'))

        for ci, cp in enumerate(cpts):
                sim_f[0,ci,:] = np.resize(cp.f,3)
                contact_x[0,ci,:] = np.resize(cp.x,3)
                contact_v[0,ci,:] = np.resize(cp.v,3)

        

        t = 0.0   # used for control frequency  
        time_start = time.time()
        # simulation loop 
        for i in range(N_SIMULATION):

            if(USE_CONTROLLER):
                # sinusoid trajectory 
                sampleCom.pos(offset + np.multiply(amp, matlib.sin(two_pi_f*t)))
                sampleCom.vel(np.multiply(two_pi_f_amp, matlib.cos(two_pi_f*t)))
                sampleCom.acc(np.multiply(two_pi_f_squared_amp, -matlib.sin(two_pi_f*t)))
                invdyn.comTask.setReference(sampleCom)

                HQPData = invdyn.formulation.computeProblemData(t, q[i], v[i])
                sol = invdyn.solver.solve(HQPData)
                if(sol.status != 0):
                    print("[%d] QP problem could not be solved! Error code:" % (i), sol.status)
                    break

                u = invdyn.formulation.getActuatorForces(sol)
                #            dv_des = invdyn.formulation.getAccelerations(sol)
            else:
                invdyn.formulation.computeProblemData(t, q[i], v[i])
                #        robot.computeAllTerms(invdyn.data(), q, v)
                u = -0.03*conf.kp_posture*v[6:, 0]

            # log reference data 
            # com_pos[i, :] = np.resize(invdyn.robot.com(invdyn.formulation.data()),3)
            # com_vel[i, :] = np.resize(invdyn.robot.com_vel(invdyn.formulation.data()),3)
            # com_acc[i, :] = np.resize(invdyn.comTask.getAcceleration(sim.get_dv()),3)
            # com_vel_ref[i, :] = np.resize(sampleCom.vel(),3)
            com_pos_ref[i, :] = np.resize(sampleCom.pos(),3)
            # com_acc_ref[i, :] = np.resize(sampleCom.acc(),3)
            # com_acc_des[i, :] = np.resize(invdyn.comTask.getDesiredAcceleration,3)

            sim.step(u) 
            q += [sim.get_q()]
            sim_q[i+1,:] = np.resize(q[-1], robot.nq)
            v += [sim.get_v()]
            for ci, cp in enumerate(cpts):
                sim_f[i+1,ci,:] = np.resize(cp.f,3)
                contact_x[i+1,ci,:] = np.resize(cp.x,3)
                contact_v[i+1,ci,:] = np.resize(cp.v,3)
    
            t += dt 
        # end simulation loop
        time_spent = time.time() - time_start
        print("Real-time factor:", t/time_spent)    
        # run only first simulation (test everything works)
        plt.figure('base_height')
        plt.plot(tt[:-1], com_pos_ref[:,2], line_styles[i_ls], alpha=0.7, label='com_ref')
        plt.plot(tt, sim_q[:,2], line_styles[i_ls+1], alpha=0.7, label=name)
        plt.legend()
        plt.grid()
        plt.title('Base Height vs time ')

        i_ls += 1 
        break  # do one simulation only 

    consim.stop_watch_report(3)
    plt.show()