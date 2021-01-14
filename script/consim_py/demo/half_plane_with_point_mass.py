import numpy as np 
import pinocchio as pin 
import consim 
from pinocchio.robot_wrapper import RobotWrapper
import os, sys
from os.path import dirname, join
import matplotlib.pyplot as plt 
import consim_py.utils.plot_utils as plut
import time 


unilateral_contacts = True  
K = 1e5 * np.ones(3)
B = 2e2 * np.ones(3)
N = 250

forward_dyn_method = 3 
compute_predicted_forces = True 
anchor_slipping_method = 1 

dt = 1.e-2
ndt = 100 
mu = 0.3  
 
plane_angle = - 45*np.pi/180 
integration_type = 0 #0: explicit, 1: semi_implicit, 2: classic-explicit

contact_frames = ['root_joint']
nc = len(contact_frames)

reset_anchor_points = True

if __name__=="__main__":
    # build the point mass model 
    urdf_path = os.path.abspath('../../models/urdf/free_flyer.urdf')
    mesh_path = os.path.abspath('../../models')
    robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_path], pin.JointModelFreeFlyer()) 
    print('RobotWrapper Object Created Successfully!')

    simu = consim.build_euler_simulator(dt, ndt, robot.model, robot.data,
                        K, B, mu, forward_dyn_method, integration_type)
    print('Explicit Euler simulator created succesfully')
    half_plane = consim.create_half_plane(K, B, mu, plane_angle)
    print('Half Plane created successfully! ')

    simu.add_object(half_plane)
    print('Half Plane added successfully! ')

    # plane height is 0.5 m at x = -0.5 => start point mass at x,y,z = -.5, 0., 0.75 

    q0 = np.array([-.5, 0., .75, 0., 0., 0., 1.])
    v0 = np.zeros(6)
    tau0 = np.zeros(6)
    f = np.zeros((3, nc, N+1))


    cpts = []
    for cf in contact_frames:
        if not robot.model.existFrame(cf):
            print(("ERROR: Frame", cf, "does not exist"))
        cpts += [simu.add_contact_point(cf, robot.model.getFrameId(cf), unilateral_contacts)]

    print('Contact Point added successfully! ')


    q = np.zeros((robot.nq, N+1))
    v = np.zeros((robot.nv, N+1))
    f = np.zeros((3, nc, N+1))

    consim.stop_watch_reset_all()
    time_start = time.time()

    simu.reset_state(q0, v0, reset_anchor_points)

    q[:,0] = np.copy(q0)
    v[:,0] = np.copy(v0)
    for ci, cp in enumerate(cpts):
        f[:,ci,0] = cp.f



    for t in range(N): 
        simu.step(tau0)
        q[:,t] = simu.get_q()
        v[:,t] = simu.get_v()
        for ci, cp in enumerate(cpts):
            f[:,ci,0] = cp.f



    plt.figure()
    plt.plot(q[0,:], q[2,:])
    plt.show()