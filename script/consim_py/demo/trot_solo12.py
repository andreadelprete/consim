import numpy as np 
import pinocchio as pin 
import consim 
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import fromListToVectorOfString, rotate
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


def load_solo12_pinocchio(model_path):
    """ model path is path to robot_properties_solo """
    urdf_path = '/urdf/solo12.urdf'
    mesh_path = fromListToVectorOfString([dirname(model_path)])
    robot = RobotWrapper.BuildFromURDF(
        model_path+urdf_path, mesh_path, pin.JointModelFreeFlyer())

    q = [0., 0., 0.32, 0., 0., 0., 1.] + 4 * [0., 0., 0.]
    q[8], q[9], q[11], q[12] = np.pi/4, -np.pi/2, np.pi/4,-np.pi/2
    q[14], q[15], q[17], q[18] = -np.pi/4, np.pi/2, -np.pi/4, np.pi/2
    q = np.array(q)

    pin.framesForwardKinematics(robot.model, robot.data,
                                q[:, None])
    q[2] = .32 - \
        robot.data.oMf[robot.model.getFrameId('FL_FOOT')].translation[2]
    pin.framesForwardKinematics(robot.model, robot.data,
                                q[:, None])
    # q[2] += .015
    robot.q0.flat = q
    robot.model.referenceConfigurations["half_sitting"] = robot.q0
    robot.model.referenceConfigurations["reference"] = robot.q0
    robot.model.referenceConfigurations["standing"] = robot.q0

    robot.defaultState = np.concatenate([robot.q0, np.zeros(robot.model.nv)])
    # robot.model.defaultState = np.concatenate([q, np.zeros((robot.model.nv, 1))])
    # compute contact at point feet 
    return robot

def interpolate_state(robot, x1, x2, d):
        """ interpolate state for feedback at higher rate that plan """
        x = np.zeros([robot.model.nq+robot.model.nv,1])
        x[:robot.model.nq] =  pin.interpolate(robot.model, x1[:robot.model.nq], x2[:robot.model.nq], d)
        x[robot.model.nq:] = x1[robot.model.nq:] + d*(x2[robot.model.nq:] - x1[robot.model.nq:])
        return x

def state_diff(robot, x1, x2):
    """ returns x2 - x1 """
    xdiff = np.zeros([2*robot.model.nv, 1])
    xdiff[:robot.model.nv] = pin.difference(robot.model, x1[:robot.model.nv], x2[:robot.model.nv]) 
    xdiff[robot.model.nv:] = x2[robot.model.nv:] - x1[robot.model.nv:]
    return xdiff


whichMotion = 'trot'
USE_CONTROLLER = True 


if __name__ == "__main__":
    # simulation parameters 
    simu_params = []
    
    simu_params += [{'name': 'euler 1000',
                    'type': 'euler', 
                    'ndt': 1000}]
    # simu_params += [{'name': 'euler 200',
    #                 'type': 'euler', 
    #                 'ndt': 200}]
    # simu_params += [{'name': 'euler 100',
    #                 'type': 'euler', 
    #                 'ndt': 100}]

    simu_params += [{'name': 'exponential 500',
                    'type': 'exponential', 
                    'ndt': 500}]

    # simu_params += [{'name': 'exponential 100',
    #                 'type': 'exponential', 
    #                 'ndt': 100}]
        
    # simu_params += [{'name': 'euler 10',
    #                 'type': 'euler', 
    #                 'ndt': 10}]

    # simu_params += [{'name': 'exponential 10',
    #                 'type': 'exponential', 
    #                 'ndt': 10}]

    # simu_params += [{'name': 'exponential 1',
    #                 'type': 'exponential', 
    #                 'ndt': 1}]
    
    # simu_params += [{'name': 'euler 1',
    #                 'type': 'euler', 
    #                 'ndt': 1}]

    line_styles = ['-', '--', '-.', '-..', ':','-o']
    i_ls = 0
    
    mu = 0.3        # friction coefficient
    isSparse = False 
    isInvertible = False
    unilateral_contacts = False                  
    K = 1e5 * np.ones([3,1])
    B = 3e2 * np.ones([3,1])
    T = 1 #  1 second simution  
    dt = 1.e-3 

    N_SIMULATION = int(T/dt)        # number of time steps simulated
    PRINT_N = int(conf.PRINT_T/dt)
    DISPLAY_N = int(conf.DISPLAY_T/dt)

    # PLOT STUFF
    tt = np.arange(0.0, N_SIMULATION*dt + dt, dt) 

    # load robot 
    model_path = "../../models/robot_properties_solo"
    robot = load_solo12_pinocchio(model_path)
    print(" Solo Loaded Successfully ".center(conf.LINE_WIDTH, '#'))
    
    # now load reference trajectories 
    refX = np.load('references/'+whichMotion+'_reference_states.npy')
    refU = np.load('references/'+whichMotion+'_reference_controls.npy') 
    feedBack = np.load('references/'+whichMotion+'_feedback.npy') 

    tau = np.zeros([robot.model.nv,1])

    N_SIMULATION = refU.shape[0] 

    for simu_param in simu_params:
        ndt = simu_param['ndt']
        name = simu_param['name']
        print(" Running %s Simulation ".center(conf.LINE_WIDTH, '#')%name)
        simu_type = simu_param['type']
        # build the simulator 
        if(simu_type=='exponential'):
            sim = consim.build_exponential_simulator(dt, ndt, robot.model, robot.data,
                                        K, B , mu, isSparse, isInvertible)
        else:
            sim = consim.build_euler_simulator(dt, ndt, robot.model, robot.data,
                                            K, B , mu)
                
        # add the  contact points 
        cpts = [] # list of all contact points
        for cname in conf.contact_frames:
            if not robot.model.existFrame(cname):
                print(("ERROR: Frame", cname, "does not exist"))
            cpts += [sim.add_contact_point(cname, robot.model.getFrameId(cname), unilateral_contacts)]
        print(" %s Contact Points Added ".center(conf.LINE_WIDTH, '-')%(len(cpts)))

        # reset simulator 
        sim.reset_state(refX[0,:robot.nq], refX[0,robot.nq:], True)
        print('Reset state done '.center(conf.LINE_WIDTH, '-')) 

        
        sim_f = np.empty((N_SIMULATION+1,len(conf.contact_frames),3))*nan
        sim_q = np.empty((N_SIMULATION+1,robot.nq))*nan
        sim_v = np.empty((N_SIMULATION+1,robot.nq))*nan
        contact_x = np.empty((N_SIMULATION+1,len(conf.contact_frames),3))*nan
        contact_v = np.empty((N_SIMULATION+1,len(conf.contact_frames),3))*nan

        q = [refX[0,:robot.nq].copy()]
        sim_q[0,:] = np.resize(q[-1], robot.nq)
        v = [refX[0,robot.nq:].copy()]

        for ci, cp in enumerate(cpts):
                sim_f[0,ci,:] = np.resize(cp.f,3)
                contact_x[0,ci,:] = np.resize(cp.x,3)
                contact_v[0,ci,:] = np.resize(cp.v,3)
        

        for ci, cframe in enumerate(conf.contact_frames):
            print('initial contact position for contact '+cframe)
            print(contact_x[0,ci,:])

  
        time_start = time.time()
        # simulation loop 
        for i in range(N_SIMULATION):
            for d in range(10):
                if(USE_CONTROLLER):
                    xref = interpolate_state(robot, refX[t], refX[t+1], .1*d)
                    xact = np.concatenate([sim.get_q(), sim_get_v()])
                    diff = state_diff(xact, xref)
                    tau[6:] = refU[i] + solver.K[t].dot(diff) 
                else:
                    tau[6:] = refU[i]
                

            sim.step(tau) 
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


        # plot base trajectory 
        plt.figure('base_height')
        plt.plot(tt, sim_q[:,2], line_styles[i_ls], alpha=0.7, label=name)
        plt.legend()
        plt.title('Base Height vs time ')

        # plot contact forces 
        for ci, ci_name in enumerate(conf.contact_frames):
            plt.figure(ci_name+" normal force")
            plt.plot(tt, sim_f[:, ci, 2], line_styles[i_ls], alpha=0.7, label=name)
            plt.legend()
            plt.title(ci_name+" normal force vs time")
        
        
        i_ls += 1 
        

    consim.stop_watch_report(3)
    plt.show()


