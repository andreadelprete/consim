
import numpy as np 
import pinocchio as pin 
import consim 
from pinocchio.robot_wrapper import RobotWrapper
import os, sys
from os.path import dirname, join
import matplotlib.pyplot as plt 
from consim_py import simulator_python2
import conf_pointmass



if __name__=="__main__":
    # build the point mass model 
    urdf_path = os.path.abspath('../../../models/urdf/free_flyer.urdf')
    mesh_path = os.path.abspath('../../../models')
    robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_path], pin.JointModelFreeFlyer()) 
    print('RobotWrapper Object Created Successfully!')

    dt = 1.e-3
    mu = 0.3        # friction coefficient
    anchor_slipping = 1
    unilateral_contacts = True  
    K = 1e5 * np.eye(3)
    B = 3e2 * np.eye(3)
    normal = np.array([0., 0., 1.])
    N_SIMULATION = 200 
    
    q0 = np.array([0., 0., -1e-10, 0., 0., 0., 1.]) 
    v0 = np.array([0., 0., -1., 0., 0., 0.])
    tau = np.zeros(robot.nv) 
    contact_names = ['root_joint']

    
    line_styles = ['-', '--', '-.', ':']
    i_ls = 0

    sim = simulator_python2.RobotSimulator(conf_pointmass, robot)
    cpts = []
    for cf in contact_names:
        if not robot.model.existFrame(cf):
            print(("ERROR: Frame", cf, "does not exist"))
        sim.add_contact(cf, normal, K, B)
        if unilateral_contacts:
            sim.contacts[-1].unilateral = True
        else:
            sim.contacts[-1].unilateral = False 
    print('Contacts added to simulator Successfully!')

    q = np.zeros((robot.nq, N_SIMULATION+1))*np.nan
    v = np.zeros((robot.nv, N_SIMULATION+1))*np.nan
    f = np.zeros((3*len(conf_pointmass.contact_frames), N_SIMULATION+1))

    sim.init(q0, v0, True)
    q[:,0] = np.copy(sim.q)
    v[:,0] = np.copy(sim.v)

    for t in range(N_SIMULATION):
        q[:,t+1], v[:,t+1], _ = sim.simulate(tau, dt=1.e-3, ndt=10)


    print dt 


    plt.figure('ball height')
    plt.plot(dt*np.arange(N_SIMULATION+1), q[2,:])
    plt.title('ball height')
   
   
   
   
    plt.show()