# constructs and simulates the drop of a point mass using both Euler and Exponential Simulators 
import numpy as np 
import pinocchio as pin 
import consim 
from pinocchio.robot_wrapper import RobotWrapper
import os, sys
from os.path import dirname, join
import matplotlib.pyplot as plt 
import consim_py.utils.plot_utils as plut

pin.setNumpyType(np.matrix)

if __name__=="__main__":
    # build the point mass model 
    urdf_path = os.path.abspath('../../models/urdf/free_flyer.urdf')
    mesh_path = os.path.abspath('../../models')
    robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_path], pin.JointModelFreeFlyer()) 
    print('RobotWrapper Object Created Successfully!')

    dt = 1.e-3
    mu = 0.3        # friction coefficient
    anchor_slipping = 1
    unilateral_contacts = False  
    K = 1e5 * np.ones([3,1])
    B = 3e2 * np.ones([3,1])
    N_SIMULATION = 200
    
    q0 = np.array([0., 0., -1e-10, 0., 0., 0., 1.]) [:,None]
    dq0 = np.array([0., 0., -1., 0., 0., 0.]) [:,None]
    tau = np.zeros(robot.nv) [:,None]
    contact_names = ['root_joint']
    
    simu_params = []
    simu_params += [{'name': 'exponential 100',
                    'type': 'exponential', 
                    'ndt': 100}]
    # simu_params += [{'name': 'exponential 500',
    #                 'type': 'exponential', 
    #                 'ndt': 500}]
    # simu_params += [{'name': 'exponential 1',
    #                 'type': 'exponential', 
    #                 'ndt': 1}]
    # simu_params += [{'name': 'euler 500',
    #                  'type': 'euler',
    #                 'ndt': 500}]
    simu_params += [{'name': 'exponential 10',
                    'type': 'exponential', 
                    'ndt': 10}]  
#    simu_params += [{'name': 'exponential 1',
#                    'type': 'exponential', 
#                    'ndt': 1}]
#    simu_params += [{'name': 'euler 300',
#                    'type': 'euler', 
#                    'ndt': 300}]
#    simu_params += [{'name': 'euler 100',
#                    'type': 'euler', 
#                    'ndt': 100}]
    # simu_params += [{'name': 'euler 10',
    #                 'type': 'euler', 
    #                 'ndt': 10}]
#    simu_params += [{'name': 'euler 1',
#                    'type': 'euler', 
#                    'ndt': 1}]
    
    line_styles = ['-', '--', '-.', ':']
    i_ls = 0
    for simu_param in simu_params:
        ndt = simu_param['ndt']
        name = simu_param['name']
        simu_type = simu_param['type']
        if(simu_type=='exponential'):
            sim = consim.build_exponential_simulator(dt, ndt, robot.model, robot.data,
                                        K, B , mu, anchor_slipping, True)
        else:
            sim = consim.build_euler_simulator(dt, ndt, robot.model, robot.data,
                                            K, B, mu)
        
        cpts = []
        for cf in contact_names:
            if not robot.model.existFrame(cf):
                print(("ERROR: Frame", cf, "does not exist"))
            cpts += [sim.add_contact_point(cf, robot.model.getFrameId(cf), unilateral_contacts)]
        print('Contacts added to simulator Successfully!')
    
        fcnt = np.zeros([N_SIMULATION+1, len(cpts), 3])
        pred_f = np.zeros([N_SIMULATION+1, len(cpts), 3])
        pred_x = np.zeros([N_SIMULATION+1, len(cpts), 3])
        q_log = np.zeros([N_SIMULATION+1, robot.nq])
        v_log = np.zeros([N_SIMULATION+1, robot.nv])
    
        robot.forwardKinematics(q0)
        sim.reset_state(q0, dq0, True)
        print('Reset state done !') 
        q_log[0,:] = np.resize(sim.get_q(),robot.nq)
        v_log[0,:] = np.resize(sim.get_v(),robot.nv)


        for i, cp in enumerate(cpts):
            fcnt[0,i,:] = np.resize(cp.f,3)
            pred_f[0,i,:] = np.resize(cp.predicted_f,3)
            pred_x[0,i,:] = np.resize(cp.predicted_x,3)

        for t in range(N_SIMULATION):
            sim.step(tau) 
            q_log[t+1,:] = np.resize(sim.get_q(),robot.nq)
            v_log[t+1,:] = np.resize(sim.get_v(),robot.nv)
            for i, cp in enumerate(cpts):
                fcnt[t+1,i,:] = np.resize(cp.f,3)
                pred_f[t+1,i,:] = np.resize(cp.predicted_f,3)
                pred_x[t+1,i,:] = np.resize(cp.predicted_x,3)
        print('Simulation done ')


        
        # plt.figure('Ball Height')
        # plt.plot(dt*np.arange(N_SIMULATION+1), q_log[:,2], line_styles[i_ls], alpha=0.7, label=name)
        # if(simu_type=='exponential'):
        #     for i,cp in enumerate(cpts):
        #         plt.plot(dt*np.arange(N_SIMULATION+1), pred_x[:,i,2], line_styles[i_ls+1], alpha=0.7, label=name+'  predicted')
        # plt.legend()
        # plt.grid()
        # plt.title('Ball Height vs time ')

        # plt.figure('normal contact forces')
        # for i,cp in enumerate(cpts):
        #     plt.plot(dt*np.arange(N_SIMULATION+1), fcnt[:,i,2], line_styles[i_ls], alpha=0.7, label=name+' pnt %s'%i)
        #     if(simu_type=='exponential'):
        #         plt.plot(dt*np.arange(N_SIMULATION+1), pred_f[:,i,2], line_styles[i_ls], alpha=0.7, label=name+' pnt %s predicted'%i)
        # plt.legend()
        # plt.grid()
        # plt.title('normal contact forces')
        
        # i_ls += 1
        # if(i_ls >= len(line_styles)):
        #     i_ls = 0
        plt.figure('ball height')
        plt.plot(dt*np.arange(N_SIMULATION+1), q_log[:,2])
        plt.title('ball height')
   

    consim.stop_watch_report(3)

    plt.show()
