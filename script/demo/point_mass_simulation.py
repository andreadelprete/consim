# constructs and simulates the drop of a point mass using both Euler and Exponential Simulators 
import numpy as np 
import pinocchio as pin 
import consim 
from pinocchio.robot_wrapper import RobotWrapper
import os, sys
from os.path import dirname, join
import matplotlib.pyplot as plt 

if __name__=="__main__":
    # build the point mass model 
    urdf_path = os.path.abspath('../models/urdf/free_flyer.urdf')
    mesh_path = os.path.abspath('../models')
    robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_path], pin.JointModelFreeFlyer()) 
    print 'RobotWrapper Object Created Successfully!'
    # create the euler simulator 
    dt = 1.e-3 
    ndt = 1
    mu = 0.3        # friction coefficient

    K = 1e5
    B = 3e2

    N = 800

    euler_sim = consim.build_euler_simulator(dt, ndt, robot.model, robot.data,
                                        K, B ,K, B, mu, mu)
    contact_names = ['root_joint']
    cpts_euler = []
    for cf in contact_names:
        if not robot.model.existFrame(cf):
            print("ERROR: Frame", cf, "does not exist")
        cpts_euler += [euler_sim.add_contact_point(robot.model.getFrameId(cf))]
    print 'EulerSimulator Object Created Successfully!'

    fcnt_euler = np.zeros([N+1, len(cpts_euler), 3])

    

    q0 = np.array([0., 0., 1., 0., 0., 0., 1.]) [:,None]
    dq0 = np.zeros(robot.nv)[:,None]
    tau = np.zeros(robot.nv) [:,None]
    robot.forwardKinematics(q0)
    euler_sim.reset_state(q0, dq0, True)

    for i, cp in enumerate(cpts_euler):
        fcnt_euler[0,i,:] = np.resize(cp.f,3)


    
    q = [q0]
    dq = [dq0]
    for t in range(N):
        euler_sim.step(tau)
        q += [euler_sim.get_q()]
        dq += [euler_sim.get_dq()]
        # store the contact forces 
        for i, cp in enumerate(cpts_euler):
            fcnt_euler[t+1,i,:] = np.resize(cp.f,3)

    print 'Simulation using euler simulator done '

    qz = []
    dqz = []
    for i,qi in enumerate(q):
        qz += [qi[2,0]]
        dqz += [dq[i][2,0]]
    
    plt.figure('Ball Height')
    plt.plot(dt*np.arange(N+1), qz, label='euler')

    isSparse = False 
    isInvertible = False 
    exp_sim = consim.build_exponential_simulator(dt, ndt, robot.model, robot.data,
                                        K, B ,K, B, mu, mu, isSparse, isInvertible)
    print 'ExponentialSimulator Object Created Successfully!'

    cpts_exp = []
    for cf in contact_names:
        if not robot.model.existFrame(cf):
            print("ERROR: Frame", cf, "does not exist")
        cpts_exp += [exp_sim.add_contact_point(robot.model.getFrameId(cf))]
    print 'Contacts added to ExponentialSimulator Successfully!'

    fcnt_exp = np.zeros([N+1, len(cpts_exp), 3])

    robot.forwardKinematics(q0)
    exp_sim.allocate_data()
    print 'allocate data works'
    exp_sim.reset_state(q0, dq0, True)
    print 'ExpoSim reset state done !' 
    q = [q0]
    dq = [dq0]

    for i, cp in enumerate(cpts_exp):
        fcnt_exp[0,i,:] = np.resize(cp.f,3)


    for t in range(N):
        exp_sim.step(tau)
        q += [exp_sim.get_q()]
        dq += [exp_sim.get_dq()]
        for i, cp in enumerate(cpts_exp):
            fcnt_exp[t+1,i,:] = np.resize(cp.f,3)

    print 'Simulation using exponential simulator done '

    qz = []
    dqz = []
    for i,qi in enumerate(q):
        qz += [qi[2,0]]
        dqz += [dq[i][2,0]]

    plt.plot(dt*np.arange(N+1), qz,'--',label='exp')
    plt.legend()
    plt.grid()
    plt.title('Ball Height vs time ')

    plt.figure('normal contact forces')
    for i,cp in enumerate(cpts_euler):
        plt.plot(dt*np.arange(N+1), fcnt_euler[:,i,2], label='euler fz pt %s'%i)

    for i,cp in enumerate(cpts_exp):
        plt.plot(dt*np.arange(N+1), fcnt_exp[:,i,2],'--', label='exp fz pt %s'%i)
    plt.legend()
    plt.grid()
    plt.title('normal contact forces')

    plt.show()

    


        
