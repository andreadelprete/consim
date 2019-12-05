"""


"""
from __future__ import print_function
from copy import deepcopy
import numpy as np
from pinocchio.utils import rotate, isapprox, zero
import pinocchio as se3
import gepetto.corbaserver
import time
import commands
import os

import conf_solo as conf
import consim

        
print("Test solo cpp sim started")
dt = 1e-3
ndt = 10
T = .5

kp = 3
kd = 1

N = int(T/dt)
robot = se3.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ], se3.JointModelFreeFlyer())

robot.data = se3.Data(robot.model)
if not robot.model.check(robot.data):
    print("Python data not consistent with model")

print("build_simple_simulator") 
print("model parents", [p for p in robot.model.parents])

sim = consim.build_simple_simulator(dt, ndt, robot.model, robot.data,
                                    conf.K[0,0], conf.B[0,0], conf.K[1,1], conf.B[1,1],
                                    conf.mu, conf.mu)
print("2")
cpts = []
for cf in conf.contact_frames:
    if not robot.model.existFrame(cf):
        print("ERROR: Frame", cf, "does not exist")
    cpts += [sim.add_contact_point(robot.model.getFrameId(cf))]

q = conf.q0
conf.q0[2,0] += 0.1
dq = zero(robot.nv)
tau = zero(robot.nv)
robot.forwardKinematics(q)
sim.reset_state(q, dq, True)

if(conf.use_viewer):
    robot_display = se3.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ], se3.JointModelFreeFlyer())
    l = commands.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
    if int(l[1]) == 0:
        os.system('gepetto-gui &')
    time.sleep(1)
    gepetto.corbaserver.Client()
    robot_display.initViewer(loadModel=True)
    robot_display.displayCollisions(False)
    robot_display.displayVisuals(True)
    robot_display.display(q)
    gui = robot_display.viewer.gui
    gui.addFloor('world/floor')
    gui.setCameraTransform(0, conf.CAMERA_TRANSFORM)

print("start simulation")
DISPLAY_N = int(conf.DISPLAY_T/dt)
display_counter = DISPLAY_N    
t = 0.0 
for it in range(N):
    tau[6:,0] = kp*(conf.q0[7:,0] - sim.q[7:,0]) - kd*sim.dq[6:,0]
    sim.step(tau)
    
    if(conf.use_viewer):
        display_counter -= 1
        if display_counter == 0: 
            robot_display.display(sim.q)
            print("Time %.3f"%(t), robot.data.com[0].T)
            display_counter = DISPLAY_N     
    t += dt
    
print("end simulation")

