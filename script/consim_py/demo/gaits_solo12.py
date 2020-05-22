import numpy as np 
import pinocchio as pin 
import consim 
import os, sys
from os.path import dirname, join
import matplotlib.pyplot as plt 
import consim_py.utils.plot_utils as plut
import time
import matplotlib.pyplot as plt

import consim_py.conf_solo_py as conf
from example_robot_data.robots_loader import loadSolo

import numpy.matlib as matlib
from numpy import nan
from numpy.linalg import norm as norm

pin.setNumpyType(np.matrix)


def interpolate_state(robot, x1, x2, d):
        """ interpolate state for feedback at higher rate that plan """
        x = np.zeros([robot.model.nq+robot.model.nv,1])
        x[:robot.model.nq] =  pin.interpolate(robot.model, x1[:robot.model.nq], x2[:robot.model.nq], d)
        x[robot.model.nq:] = x1[robot.model.nq:] + d*(x2[robot.model.nq:] - x1[robot.model.nq:])
        return x

def state_diff(robot, x1, x2):
    """ returns x2 - x1 """
    xdiff = np.zeros([2*robot.model.nv, 1])
    xdiff[:robot.model.nv] = pin.difference(robot.model, x1[:robot.model.nq], x2[:robot.model.nq]) 
    xdiff[robot.model.nv:] = x2[robot.model.nq:] - x1[robot.model.nq:]
    return xdiff


whichMotion = 'trot'  # options = ['trot', 'jump"]
USE_CONTROLLER = True 
DISPLAY_SIMULATION = True  


if __name__ == "__main__":
    # simulation parameters 
    simu_params = []
    
#    simu_params += [{'name': 'euler 100',
#                    'type': 'euler', 
#                    'ndt': 100}]
    # simu_params += [{'name': 'euler 200',
    #                 'type': 'euler', 
    #                 'ndt': 200}]
    simu_params += [{'name': 'euler 100',
                     'type': 'euler', 
                     'ndt': 100}]

    # simu_params += [{'name': 'exponential 500',
    #                 'type': 'exponential', 
    #                 'ndt': 500}]

    simu_params += [{'name': 'exponential 100',
                    'type': 'exponential', 
                    'ndt': 100}]
        
#    simu_params += [{'name': 'euler 10',
#                     'type': 'euler', 
#                     'ndt': 10}]

    # simu_params += [{'name': 'exponential 10',
    #                 'type': 'exponential', 
    #                 'ndt': 10}]
#
    # simu_params += [{'name': 'exponential 1',
    #                  'type': 'exponential', 
    #                  'ndt': 1}]
    
#    simu_params += [{'name': 'euler 1',
#                     'type': 'euler', 
#                     'ndt': 1}]

    line_styles = ['-', '--', '-.', '-..', ':','-o']
    i_ls = 0
    
    mu = 1.        # friction coefficient
    anchor_slipping = 1 
    unilateral_contacts = True                   
    K = 1.e+5 * np.ones([3,1])
    B = 2.4e+2 * np.ones([3,1])
    T = 1 #  1 second simution  
    dt = 2.e-3 
    dt_ref = 1e-2 # time step of reference motion

    PRINT_N = int(conf.PRINT_T/dt)
    DISPLAY_N = int(conf.DISPLAY_T/dt)

    # load robot 
    # model_path = "../../models/robot_properties_solo"
    robot = loadSolo(False) ## load_solo12_pinocchio(model_path)
    print(" Solo Loaded Successfully ".center(conf.LINE_WIDTH, '#'))
    
    # now load reference trajectories 
    refX = np.load('references/'+whichMotion+'_reference_states.npy')
    refU = np.load('references/'+whichMotion+'_reference_controls.npy') 
    feedBack = np.load('references/'+whichMotion+'_feedback.npy') 

    tau = np.zeros([robot.model.nv,1])

    N_SIMULATION = refU.shape[0] 
    tt = np.arange(0.0, N_SIMULATION*dt + dt, dt) 

    for simu_param in simu_params:
        ndt = simu_param['ndt']
        name = simu_param['name']
        print(" Running %s Simulation ".center(conf.LINE_WIDTH, '#')%name)
        simu_type = simu_param['type']
        # build the simulator 
        if(simu_type=='exponential'):
            sim = consim.build_exponential_simulator(dt, ndt, robot.model, robot.data,
                                        K, B , mu, anchor_slipping)
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
        
        sim_f = np.empty((N_SIMULATION+1,len(conf.contact_frames),3))*nan
        sim_q = np.empty((N_SIMULATION+1,robot.nq))*nan
        sim_v = np.empty((N_SIMULATION+1,robot.nv))*nan
        contact_x = np.empty((N_SIMULATION+1,len(conf.contact_frames),3))*nan
        contact_v = np.empty((N_SIMULATION+1,len(conf.contact_frames),3))*nan

        q = [refX[0,:robot.nq].copy()]
        sim_q[0,:] = np.resize(q[-1], robot.nq)
        v = [refX[0,robot.nq:].copy()]

        for ci, cp in enumerate(cpts):
            sim_f[0,ci,:] = np.resize(cp.f,3)
            contact_x[0,ci,:] = np.resize(cp.x,3)
            contact_v[0,ci,:] = np.resize(cp.v,3)

        t = 0. 
        time_start = time.time()
        # simulation loop 
        for i in range(N_SIMULATION):
            for d in range(int(dt_ref/dt)):
                if(USE_CONTROLLER):
                    xref = interpolate_state(robot, refX[i], refX[i+1], dt*d/dt_ref)
                    xact = np.concatenate([sim.get_q(), sim.get_v()])
                    diff = state_diff(robot, xact, xref)
                    tau[6:] = refU[i] + feedBack[i].dot(diff) 
                else:
                    tau[6:] = refU[i]

                sim.step(tau) 
            # log only at 10 ms 
            sim_q[i+1,:] = np.resize(sim.get_q(), robot.nq)
            sim_v[i+1,:] = np.resize(sim.get_v(), robot.nv)

            for ci, cp in enumerate(cpts):
                sim_f[i+1,ci,:] = np.resize(cp.f,3)
                contact_x[i+1,ci,:] = np.resize(cp.x,3)
                contact_v[i+1,ci,:] = np.resize(cp.v,3)
    
            t += dt
        # end simulation loop
        time_spent = time.time() - time_start
        print("Real-time factor:", t/time_spent)    

        if DISPLAY_SIMULATION:
            robot.initViewer(loadModel=True)
            robot.viewer.gui.createSceneWithFloor('world')
            robot.viewer.gui.setLightingMode('world/floor', 'OFF')
            cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
#            robot.viewer.gui.setCameraTransform('python-pinocchio', cameraTF)
            # backgroundColor = [1., 1., 1., 1.]
            # floorColor = [0.7, 0.7, 0.7, 1.]
            # #   
            # window_id = robot.viz.viewer.gui.getWindowID("python-pinocchio")
            # robot.viz.viewer.gui.setBackgroundColor1(window_id, backgroundColor)
            # robot.viz.viewer.gui.setBackgroundColor2(window_id, backgroundColor)

            for i in range(N_SIMULATION):
                robot.display(sim_q[i][:,None])
                time.sleep(dt_ref)


        qz = []

        for i in range(N_SIMULATION+1):
            qz += [sim_q[i,2]]

    #     # plot base trajectory 
        plt.figure('base_height')
        plt.plot(tt, qz , line_styles[i_ls], alpha=0.7, label=name)
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
