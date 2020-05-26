
import numpy as np 
import pinocchio as pin 
import consim 
from pinocchio.robot_wrapper import RobotWrapper
import os, sys
from os.path import dirname, join
import matplotlib.pyplot as plt 
from consim_py import simulator_python2
import conf_pointmass

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

if __name__=="__main__":
    # build the point mass model 
    urdf_path = os.path.abspath('../../../models/urdf/free_flyer.urdf')
    mesh_path = os.path.abspath('../../../models')
    robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_path], pin.JointModelFreeFlyer()) 
    print('RobotWrapper Object Created Successfully!')

    dt = 1.e-3
    mu = 0.3        # friction coefficient
    # isSparse = False 
    # isInvertible = False
    anchor_slipping = 1 
    unilateral_contacts = True  
    K = 1e5 * np.eye(3)
    B = 2e2 * np.eye(3)
    normal = np.array([0., 0., 1.])
    N_SIMULATION = 200 
    
    q0 = np.array([0., 0., 0., 0., 0., 0., 1.])
    v0 = np.array([0., 0., 0., 0., 0., 0.])

    tau0 = np.zeros(robot.nv)
    tau0[2] = -0.19 # complete it to 10 Nm  
    tau = np.zeros(robot.nv) 
    tau[0] = 4. 
    tau[2] = -0.19 # complete it to 10 Nm  

    contact_names = ['root_joint']

    sim = simulator_python2.RobotSimulator(conf_pointmass, robot)
    if anchor_slipping == 1:
        sim.cone_method = "average"
    elif anchor_slipping == 2:
        sim.cone_method = "qp"
    else:
        raise Exception("cone update method not recognized")

            
    cpts = []
    for cf in contact_names:
        if not robot.model.existFrame(cf):
            print(("ERROR: Frame", cf, "does not exist"))
        sim.add_contact(cf, normal, K, B)
        if unilateral_contacts:
            sim.contacts[-1].unilateral = True
        else:
            sim.contacts[-1].unilateral = False 
        sim.contacts[-1].friction_coeff = mu
    print('Contacts added to simulator Successfully!')

    q = np.zeros((robot.nq, N_SIMULATION+1))*np.nan
    v = np.zeros((robot.nv, N_SIMULATION+1))*np.nan
    f = np.zeros((3*len(conf_pointmass.contact_frames), N_SIMULATION+1))

    sim.init(q0, v0, True)
    q[:,0] = np.copy(sim.q)
    v[:,0] = np.copy(sim.v)

    for t in range(N_SIMULATION):
        q[:,t+1], v[:,t+1], _ = sim.simulate(tau, dt=1.e-3, ndt=10)


    plt.figure('ball height')
    plt.plot(dt*np.arange(N_SIMULATION+1), q[2,:])
    plt.title('ball height')

    plt.figure('ball X pos')
    plt.plot(dt*np.arange(N_SIMULATION+1), q[0,:])
    plt.title('ball X pos')
   
   
   
   
    plt.show()
    