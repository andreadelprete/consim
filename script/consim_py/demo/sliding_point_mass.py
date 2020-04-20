""" place a point mass on the floor and slide it, test how different anchor point updates behave """

import numpy as np 
import pinocchio as pin 
import consim 
from pinocchio.robot_wrapper import RobotWrapper
import os, sys
from os.path import dirname, join
import matplotlib.pyplot as plt 
import utils.plot_utils as plut

pin.setNumpyType(np.matrix)

if __name__=="__main__":
    # build the point mass model 
    urdf_path = os.path.abspath('../../models/urdf/free_flyer.urdf')
    mesh_path = os.path.abspath('../../models')
    robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_path], pin.JointModelFreeFlyer()) 
    print('RobotWrapper Object Created Successfully!')

    dt = 1.e-3
    mu = 0.3        # friction coefficient
    isSparse = False 
    isInvertible = False
    unilateral_contacts = True 
    K = 1e5
    B = 3e2
    N = 1000 
    
    q0 = np.array([0., 0., 0., 0., 0., 0., 1.]) [:,None]
    dq0 = np.array([0., 0., 0., 0., 0., 0.]) [:,None]
    """ apply a force that pushes the point mass down and violates the friction cone 
    say fN = - 10 N
    at v = 0 
    the penetration will be 10 = 1.e5*delta -> delta = 1.e-4 not so bad 
    then the friction cone bound at mu = 0.3 is 3 N 
    given mass  = 1 kg 
    applying 4 Nm will result in a = 1 m / s^2 
    we can compute the trajectory of the anchor point 
    since there is no rolling, tipping and other contact states 
    v(t) = v(0) + t*a 
    x(t) = x(0) + t*v(0) + .5 * t^2 * a (assuming a is constant ) 
    """
    tau0 = np.zeros(robot.nv) [:,None]
    tau0[2] = -0.19 # complete it to 10 Nm  
    tau = np.zeros(robot.nv) [:,None]
    tau[0] = 4. 
    tau[2] = -0.19 # complete it to 10 Nm  

    contact_names = ['root_joint']
    
    simu_params = []
    simu_params += [{'name': 'exponential 100',
                    'type': 'exponential', 
                    'ndt': 100}]
    simu_params += [{'name': 'exponential 10',
                    'type': 'exponential', 
                    'ndt': 10}]
    simu_params += [{'name': 'exponential 1',
                    'type': 'exponential', 
                    'ndt': 1}]
    simu_params += [{'name': 'euler 100',
                    'type': 'euler', 
                    'ndt': 100}]
    simu_params += [{'name': 'euler 10',
                    'type': 'euler', 
                    'ndt': 10}]

    line_styles = ['-', '--', '-.', ':']
    i_ls = 0
    for simu_param in simu_params:
        ndt = simu_param['ndt']
        name = simu_param['name']
        simu_type = simu_param['type']
        if(simu_type=='exponential'):
            sim = consim.build_exponential_simulator(dt, ndt, robot.model, robot.data,
                                        K, B ,K, B, mu, mu, isSparse, isInvertible)
        else:
            sim = consim.build_euler_simulator(dt, ndt, robot.model, robot.data,
                                            K, B ,K, B, mu, mu)
        
        cpts = []
        for cf in contact_names:
            if not robot.model.existFrame(cf):
                print(("ERROR: Frame", cf, "does not exist"))
            cpts += [sim.add_contact_point(robot.model.getFrameId(cf), unilateral_contacts)]
        print('Contacts added to simulator Successfully!')
    
        fcnt = np.zeros([N+1, len(cpts), 3])
        xstart = np.zeros([N+1, len(cpts), 3])
    
        robot.forwardKinematics(q0)
        sim.reset_state(q0, dq0, True)
        print('Reset state done !') 
    
        for i, cp in enumerate(cpts):
            fcnt[0,i,:] = np.resize(cp.f,3)
    
        q = [q0]
        dq = [dq0]
        for t in range(N):
            # check if contact stabalized before applying tangential force 
            # if (fcnt[t,0,2]- 10.)**2 < 1.e-4: 
            #     sim.step(tau)
            # else:
            sim.step(tau)

            q += [sim.get_q()]
            dq += [sim.get_v()]
            for i, cp in enumerate(cpts):
                fcnt[t+1,i,:] = np.resize(cp.f,3)
                xstart[t+1,i,:] = np.resize(cp.x_start,3)
        print('Simulation done ')

        qz = []
        dqz = []
        qx = []
        for i,qi in enumerate(q):
            # print qi.shape 
            if(abs(qi[2])>1e2):
                qi[2] = np.nan
            qz += [qi[2]]
            qx += [qi[0]]
            dqz += [dq[i][2]]
        
        plt.figure('Ball Height')
        plt.plot(dt*np.arange(N+1), qz, line_styles[i_ls], alpha=0.7, label=name)
        plt.legend()
        plt.grid()
        plt.title('Ball Height vs time ')

        plt.figure('Ball XPos')
        plt.plot(dt*np.arange(N+1), qx, line_styles[i_ls], alpha=0.7, label=name)
        plt.legend()
        plt.grid()
        plt.title('Ball XPos vs time ')

        plt.figure('normal contact forces')
        for i,cp in enumerate(cpts):
            plt.plot(dt*np.arange(N+1), fcnt[:,i,2], line_styles[i_ls], alpha=0.7, label=name+' pnt %s'%i)
        plt.legend()
        plt.grid()
        plt.title('normal contact forces')

        plt.figure('tangent contact forces')
        for i,cp in enumerate(cpts):
            plt.plot(dt*np.arange(N+1), np.sqrt(fcnt[:,i,0]*fcnt[:,i,0]+fcnt[:,i,1]*fcnt[:,i,1]), line_styles[i_ls], alpha=0.7, label=name+' pnt %s'%i)
        plt.legend()
        plt.grid()
        plt.title('tangent contact forces')

        plt.figure('Anchor Point')
        for i,cp in enumerate(cpts):
            plt.plot(dt*np.arange(N+1), xstart[:,i,0], line_styles[i_ls], alpha=0.7, label=name+' pnt %s'%i)
        plt.legend()
        plt.grid()
        plt.title('Anchor Point')
        
        i_ls += 1
        if(i_ls >= len(line_styles)):
            i_ls = 0

    consim.stop_watch_report(3)

    plt.show()
