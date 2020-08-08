''' Linear feedback controller
'''
import numpy as np
from simu_cpp_common import state_diff

class LinearFeedbackController:

    def __init__(self, robot, dt, refX, refU, feedBack):
        # load reference trajectories 
        self.robot = robot
        self.refX, self.refU, self.feedBack = refX, refU, feedBack
        self.i = 0
        self.q0, self.v0 = self.refX[0,:robot.nq], self.refX[0,robot.nq:]

    def reset(self, q, v, time_before_start):
        self.i = 0

    def compute_control(self, q, v):
        xact = np.concatenate([q, v])
        diff = state_diff(self.robot, xact, self.refX[self.i])
        u = self.refU[self.i] + self.feedBack[self.i].dot(diff)        
        self.i += 1
        return np.concatenate((np.zeros(6),u))