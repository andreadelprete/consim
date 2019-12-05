from simple_biped.utils.test_LDS_integral import compute_integral_x_T, compute_double_integral_x_T, compute_x_T_and_two_integrals
import pinocchio as se3
from pinocchio.utils import zero
import numpy as np
import numpy.matlib as matlib
from numpy import nan
from scipy.linalg import expm
from numpy.linalg import norm as norm
import os
import gepetto.corbaserver
import time
import commands

class Contact:
    def __init__(self, model, data, frame_name, normal, K, B):
        self.model = model
        self.data = data
        self.frame_name = frame_name
        self.normal = normal
        self.K = K
        self.B = B
        self.frame_id = model.getFrameId(frame_name)
        self.reset_contact_position()
        
    def reset_contact_position(self):
        self.p0 = self.data.oMf[self.frame_id].translation.copy()  # Initial (0-load) position of the spring
#        print 'Contact %s p0='%(self.frame_name), self.p0.T
        self.in_contact = True
        
    def compute_force(self):
        M = self.data.oMf[self.frame_id]
        self.p = M.translation
        delta_p = self.p0 - self.p
#        print 'Contact %s p='%(self.frame_name), M.translation.T
#        print 'Contact %s delta_p='%(self.frame_name), delta_p.T
        
        R = se3.SE3(M.rotation, 0*M.translation)
#        v_local = self.model.frames[self.frame_id].placement.inverse()*self.data.v[self.joint_id]
        v_local = se3.getFrameVelocity(self.model, self.data, self.frame_id)
        v_world = (R.act(v_local)).linear
        
        # Doubt: should I use classic or spatial acceleration here?!
        dJv_local = se3.getFrameAcceleration(self.model, self.data, self.frame_id)
        dJv_local.linear += np.cross(v_local.angular, v_local.linear, axis=0)
        dJv_world = (R.act(dJv_local)).linear
        
#        if(delta_p.T * self.normal < -1e-6):
#            self.f = zero(3)
#            self.v = zero(3)
#            self.dJv = zero(3)
##            print 'Contact %s delta_p*Normal='%(self.frame_name), delta_p.T*self.normal
#            if(self.in_contact):
#                self.in_contact = False
#                print "\nINFO: contact %s broken!"%(self.frame_name), delta_p.T, self.normal.T
#        else:
#            if(not self.in_contact):
#                self.in_contact = True
#                print "\nINFO: contact %s made!"%(self.frame_name)
        self.f = self.K*delta_p - self.B*v_world
        self.v = v_world
        self.dJv = dJv_world
        return self.f   

    def getJacobianWorldFrame(self):
#        se3.framesForwardKinematics(self.model,self.data,q)
        M = self.data.oMf[self.frame_id]
        R = se3.SE3(M.rotation, 0*M.translation)
        J_local = se3.getFrameJacobian(self.model, self.data, self.frame_id, se3.ReferenceFrame.LOCAL)
        self.J = (R.action * J_local)[:3,:]
        return self.J
    

class RobotSimulator:
    PRINT_N = 500                   # print every PRINT_N time steps
    DISPLAY_N = 10                  # update robot configuration in viwewer every DISPLAY_N time steps

    def __init__(self, conf, root_joint=None):
        self.conf = conf
        self.assume_A_invertible = False
        self.robot = se3.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ], se3.JointModelFreeFlyer())
        self.model = self.robot.model
        self.data = self.robot.data
        self.t = 0.0
        self.ndt_force = 1 # number of contact force samples for each time step
        self.f = zero(0)
        self.f_log = matlib.zeros((self.ndt_force,0))
        nv = self.model.nv
        if root_joint is None:  na = nv
        else:                   na = nv-6
        self.S = np.hstack((matlib.zeros((na,nv-na)), matlib.eye(na, na)))
        self.contacts = []
        self.display_counter = self.DISPLAY_N
        self.init(conf.q0, None, True)
#        self.init(self.robot.model.neutralConfiguration, None, True)
        
        # for gepetto viewer
        if(conf.use_viewer):
            self.robot_display = se3.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ], se3.JointModelFreeFlyer())
            l = commands.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
            if int(l[1]) == 0:
                os.system('gepetto-gui &')
            time.sleep(1)
            gepetto.corbaserver.Client()
            self.robot_display.initViewer(loadModel=True)
            self.robot_display.displayCollisions(False)
            self.robot_display.displayVisuals(True)
            self.robot_display.display(self.q)
            self.gui = self.robot_display.viewer.gui
            self.gui.setCameraTransform(0, conf.CAMERA_TRANSFORM)

    def init(self, q0, v0=None, reset_contact_positions=False):
        self.first_iter = True
        self.q  = q0.copy()
        if(v0 is None): self.v = zero(self.robot.nv)
        else:           self.v = v0.copy()
        self.dv = zero(self.robot.nv)

        # reset contact position    
        if(reset_contact_positions):  
            se3.forwardKinematics(self.model, self.data, self.q)
            se3.updateFramePlacements(self.model, self.data)
            i = 0
            for c in self.contacts:
                c.reset_contact_position()
                self.p0[i:i+3,0] = c.p0
                i += 3
        
        self.compute_forces(compute_data=True)
        
    def add_contact(self, frame_name, normal, K, B):
        c = Contact(self.model, self.data, frame_name, normal, K, B)
        self.contacts += [c]
        self.nc = len(self.contacts)
        self.nk = 3*self.nc
        self.f = zero(self.nk)
        self.f_log = matlib.zeros((self.nk,self.ndt_force))
        self.Jc = matlib.zeros((self.nk, self.model.nv))
        self.K = matlib.zeros((self.nk, self.nk))
        self.B = matlib.zeros((self.nk, self.nk))
        self.p0  = zero(self.nk)
        self.p   = zero(self.nk)
        self.dp  = zero(self.nk)
        self.dJv = zero(self.nk)
        self.a   = zero(2*self.nk)
        self.A = matlib.zeros((2*self.nk, 2*self.nk))
        self.A[:self.nk, self.nk:] = matlib.eye(self.nk)
        i = 0
        for c in self.contacts:
            self.K[i:i+3,i:i+3] = c.K
            self.B[i:i+3,i:i+3] = c.B
            self.p0[i:i+3,0] = c.p0
            i += 3
        self.D = np.hstack((-self.K, -self.B))
        
    def compute_forces(self, compute_data = True):
        '''Compute the contact forces from q, v and elastic model'''
        if compute_data :
            se3.forwardKinematics(self.model, self.data, self.q, self.v)
            se3.computeAllTerms(self.model, self.data, self.q, self.v)
            se3.updateFramePlacements(self.model, self.data)
            se3.computeJointJacobians(self.model, self.data, self.q)

        i = 0
        for c in self.contacts:
            self.f[i:i+3,0] = c.compute_force()
            i += 3
        
        return self.f
        
    def step(self, u, dt = None, use_exponential_integrator=True):
        if dt is None: dt = self.dt
        
        if self.first_iter: 
            self.compute_forces()
            self.first_iter = False
        
        # dv  = se3.aba(robot.model,robot.data,q,v,tauq,ForceDict(self.forces,NB))
        # (Forces are directly in the world frame, and aba wants them in the end effector frame)
        se3.forwardKinematics(self.model, self.data, self.q, self.v)
        se3.computeAllTerms(self.model, self.data, self.q, self.v)
        se3.updateFramePlacements(self.model, self.data)
        se3.computeJointJacobians(self.model, self.data, self.q)
        M  = self.data.M        #(7,7)
        h  = self.data.nle      #(7,1)
        i = 0
        for c in self.contacts:
            J = c.getJacobianWorldFrame()
            self.Jc[i:i+3,:] = J
            i += 3

        if(not use_exponential_integrator):
            self.dv = np.linalg.solve(M, self.S.T*u - h + self.Jc.T*self.f) #use last forces
            v_mean = self.v + 0.5*dt*self.dv
            self.v += self.dv*dt
            self.q  = se3.integrate(self.model, self.q, v_mean*dt)
        else:
            K_p0 = self.K*self.p0
            dv_bar = np.linalg.solve(M, self.S.T*u - h + self.Jc.T*K_p0)
            i = 0
            for c in self.contacts:
                self.f[i:i+3,0] = c.compute_force()
                self.p[i:i+3,0] = c.p
                self.dp[i:i+3,0] = c.v
                self.dJv[i:i+3,0] = c.dJv
                i += 3
            x0 = np.vstack((self.p, self.dp))
            JMinv = np.linalg.solve(M, self.Jc.T).T
            self.Upsilon = self.Jc*JMinv.T
            self.A[self.nk:,:self.nk] = -self.Upsilon*self.K
            self.A[self.nk:,self.nk:] = -self.Upsilon*self.B
            b = JMinv*(self.S.T*u-h) +self.dJv + self.Upsilon*K_p0
            self.a[self.nk:,0] = b

            if(self.assume_A_invertible):            
                int_x, int2_x = compute_double_integral_x_T(self.A, self.a, x0, dt,
                                                            compute_also_integral=True)
            else:                                            
                int_x  = compute_integral_x_T(self.A, self.a, x0, dt, invertible_A=False)
                int2_x = compute_double_integral_x_T(self.A, self.a, x0, dt,
                                                     invertible_A=False)
#                x, int_x, int2_x = compute_x_T_and_two_integrals(self.A, self.a, x0, dt)
            
            n = self.A.shape[0]
            C = matlib.zeros((n+1, n+1))
            C[0    :n,     0:n]       = self.A
            C[0    :n,     n]         = self.a
            z = matlib.zeros((n+1,1))
            z[:n,0] = x0
            z[-1,0] = 1.0
            e_TC = expm(dt/self.ndt_force*C)
            for i in range(self.ndt_force):
                self.f_log[:,i] = self.K*self.p0 + self.D * z[:n,0]
                z = e_TC*z
            
            Minv_JT_D = JMinv.T*self.D
            v_mean = self.v + 0.5*dt*dv_bar + Minv_JT_D*int2_x/dt
            dv_mean = dv_bar + Minv_JT_D*int_x/dt
            self.v += dt*dv_mean
            self.q  = se3.integrate(self.model, self.q, v_mean*dt)
            self.dv = dv_mean
        
        self.compute_forces()
        self.compute_forces(False)
        self.t += dt
        return self.q, self.v
        
        
    def reset(self):
        self.first_iter = True
        
        
    def simulate(self, u, dt=0.001, ndt=1, use_exponential_integrator=True):
        ''' Perform ndt steps, each lasting dt/ndt seconds '''
#        time_start = time.time()
        
        for i in range(ndt):
            if(not use_exponential_integrator):
                self.f_log[:,i] = self.f
            self.q, self.v = self.step(u, dt/ndt, use_exponential_integrator)
            
                   
        if(self.conf.use_viewer):
            self.display_counter -= 1
            if self.display_counter == 0: 
                self.robot_display.display(self.q)
                self.display_counter = self.DISPLAY_N         
        
#        time_spent = time.time() - time_start
#        if(time_spent < dt): time.sleep(dt-time_spent)
        
        return self.q, self.v, self.f

