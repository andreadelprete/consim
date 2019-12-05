"""


"""
from copy import deepcopy
import numpy as np
from pinocchio.utils import rotate, isapprox, zero
import pinocchio as se3
#from test_utils import TestCaseDefault, degrees2radians, one
from py_dynamics_simulator.robot_wrapper import RobotWrapper

import consim

def assertFalse(condition, string):
    if(condition):
        print string

def assertTrue(condition, string):
    if(not condition):
        print string

def assertIsApprox(a, b, tol=1e-6):
    res = isapprox(a, b, tol)
    if (not res):
        assertTrue(False, "Given and desired input do not match approximately.\n%s\n%s" % (a, b))
        
print "Test ball in the air started"

joints = [
    {'name': "base",
     'placement': {'z': 0.0},
     'shape': "sphere",
     'dimensions': 1.0,
     'mass': 1.0,
     },
]
operational_frames = [
    {'name': "center",
     'parent': "base",
     'placement': {'z': 0.0},
     },
]
ball = {'joints': joints, 'operational_frames': operational_frames}
print "test set up"

robot = RobotWrapper(ball, name='ball', display=None)

print robot.data.com[0]

robot.data = se3.Data(robot.model)

print robot.data.com[0]

if not robot.model.check(robot.data):
    print "Python data not consistent with model"
        
print("Create simple simulator")
sim = consim.build_simple_simulator(
                1e-3, 8, robot.model, robot.data,
                0., 0., 0., 0., 0., 0.);

print("Add contact point")
cpt = sim.add_contact_point(robot.model.getFrameId("center"))

q = zero(robot.nq)
q[6] = 1.0
q[2] = 0.6

dq = zero(robot.nv)
tau = zero(robot.nv)

print "forward kinematics"
robot.forward_joint_kinematics(q)

print "reset state"
sim.reset_state(q, dq, True)

print "start simulation"
for it in range(100):
    sim.step(tau)
    print robot.data.com[0].T
    assertFalse(cpt.active, "Contact point should not be active")
print "end simulation"

q_expected = np.matrix([0., 0., 0.6 - 0.5 * 0.1**2 * 9.81, 0, 0, 0, 1.]).T
assertIsApprox(sim.q, q_expected, 1e-3)

assertIsApprox(sim.q[:3], cpt.x, 1e-6)

# The frame information should be consistent after the step() call.
assertIsApprox(sim.q[:3], robot.data.oMf[robot.model.getFrameId("center")].translation)


