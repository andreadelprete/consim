# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:02:12 2020

@author: student
"""
import numpy as np
import pinocchio as pin

class Empty:
    pass

dt_ref = 0.01

def interpolate_state(robot, x1, x2, d):
    """ interpolate state for feedback at higher rate that plan """
    x = np.zeros(robot.model.nq+robot.model.nv)
    x[:robot.model.nq] =  pin.interpolate(robot.model, x1[:robot.model.nq], x2[:robot.model.nq], d)
    x[robot.model.nq:] = x1[robot.model.nq:] + d*(x2[robot.model.nq:] - x1[robot.model.nq:])
    return x


def state_diff(robot, x1, x2):
    """ returns x2 - x1 """
    xdiff = np.zeros(2*robot.model.nv)
    xdiff[:robot.model.nv] = pin.difference(robot.model, x1[:robot.model.nq], x2[:robot.model.nq]) 
    xdiff[robot.model.nv:] = x2[robot.model.nq:] - x1[robot.model.nq:]
    return xdiff
    
    
def load_ref_traj(robot, dt):
    ''' Load reference trajectories '''
    whichMotion = 'trot'
    refX_ = np.load('../demo/references/'+whichMotion+'_reference_states.npy').squeeze()
    refU_ = np.load('../demo/references/'+whichMotion+'_reference_controls.npy').squeeze() 
    feedBack_ = np.load('../demo/references/'+whichMotion+'_feedback.npy').squeeze() 
    refX_[:,2] -= 15.37e-3   # ensure contact points are inside the ground at t=0
    N = refU_.shape[0]     
    
    # interpolate reference traj
    ndt_ref = int(dt_ref/dt)
    refX     = np.empty((N*ndt_ref+1, refX_.shape[1]))
    refU     = np.empty((N*ndt_ref, refU_.shape[1]))
    feedBack = np.empty((N*ndt_ref, feedBack_.shape[1], feedBack_.shape[2]))
    for i in range(N):
        for j in range(ndt_ref):
            k = i*ndt_ref+j
            refX[k,:] = interpolate_state(robot, refX_[i], refX_[i+1], j/ndt_ref)
            refU[k,:] = refU_[i]
            feedBack[k,:,:] = feedBack_[i]
    return refX, refU, feedBack