"""


"""
from __future__ import print_function
from copy import deepcopy
import numpy as np
#from pinocchio.utils import  zero
from numpy import zeros as zero
import pinocchio as se3
#se3.setNumpyType(np.matrix)

import gepetto.corbaserver
import time
import commands
import os

#import eigenpy
#eigenpy.switchToNumpyMatrix()

import conf_solo as conf
import consim
from example_robot_data.robots_loader import loadSolo
        
print("Test solo cpp sim started")
dt = 1e-3
ndt = 10
T = 3.0

kp = 10
kd = 0.05

N = int(T/dt)
robot = loadSolo() #se3.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ], se3.JointModelFreeFlyer())

robot.data = se3.Data(robot.model)
if not robot.model.check(robot.data):
    print("Python data not consistent with model")

sim = consim.build_simple_simulator(dt, ndt, robot.model, robot.data,
                                    conf.K[0,0], conf.B[0,0], conf.K[1,1], conf.B[1,1],
                                    conf.mu, conf.mu)
cpts = []
for cf in conf.contact_frames:
    if not robot.model.existFrame(cf):
        print("ERROR: Frame", cf, "does not exist")
    cpts += [sim.add_contact_point(robot.model.getFrameId(cf))]

q = conf.q0
conf.q0[2] += 0.1
dq = zero(robot.nv)
tau = zero(robot.nv)
robot.forwardKinematics(q)
sim.reset_state(q, dq, True)

if(conf.use_viewer):
    robot_display = robot #se3.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ], se3.JointModelFreeFlyer())
    l = commands.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
    if int(l[1]) == 0:
        os.system('gepetto-gui &')
    time.sleep(1)
    gepetto.corbaserver.Client()
    robot_display.initViewer(loadModel=True)
    robot_display.viewer.gui.createSceneWithFloor('world')
    robot_display.displayCollisions(False)
    robot_display.displayVisuals(True)
    robot_display.display(q)
    gui = robot_display.viewer.gui
    gui.addFloor('world/floor')
    gui.setLightingMode('world/floor', 'OFF')
    gui.setCameraTransform(0, conf.CAMERA_TRANSFORM)

print("start simulation")
DISPLAY_N = int(conf.DISPLAY_T/dt)
PRINT_N = int(conf.PRINT_T/dt)
display_counter = DISPLAY_N    
t = 0.0 
for it in range(N):
    t0 = time.time()
    tau[6:] = kp*(conf.q0[7:] - sim.q[7:]) - kd*sim.dq[6:]
    sim.step(tau)
    
    if(conf.use_viewer):
        display_counter -= 1
        if display_counter == 0: 
            robot_display.display(np.asmatrix(sim.q).T)
            display_counter = DISPLAY_N     
            
    if(it%PRINT_N==0):
        print("Time %.3f"%(t))
        
    t += dt
    t1 = time.time()
    if(t1-t0<0.9*dt):
        time.sleep(dt-t1+t0)
    
print("end simulation")

for cf in conf.contact_frames:
    print(cf, 1e3*robot.data.oMf[robot.model.getFrameId(cf)].translation.T, 'mm')
    
consim.stop_watch_report(3)
