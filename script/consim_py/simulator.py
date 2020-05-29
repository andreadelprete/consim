#from consim_py.utils_LDS_integral import compute_x_T
import pinocchio as se3
from pinocchio.utils import zero
import numpy as np
from numpy import nan
from scipy.linalg import expm
from numpy.linalg import norm
import os
import gepetto.corbaserver
import time
import subprocess

from consim_py.utils.utils_LDS_integral import compute_integral_expm #compute_integral_x_T, compute_double_integral_x_T
from consim_py.utils.exponential_matrix_helper import ExponentialMatrixHelper


class Contact:
    def __init__(self, model, data, frame_name, normal, K, B, mu):
        self.model = model
        self.data = data
        self.frame_name = frame_name
        self.normal = normal
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.B = B
        self.frame_id = model.getFrameId(frame_name)
        self.mu = mu
        self.reset_contact_position()
        
        self.t1 = np.cross(normal, np.array([1., 0., 0.]))
        self.t1 /= norm(self.t1)
        self.t2 = np.cross(normal, self.t1)
        self.t2 /= norm(self.t2)
        

    def reset_contact_position(self, p0=None):
        # Initial (0-load) position of the spring
        if(p0 is None):
            self.p0 = self.data.oMf[self.frame_id].translation.copy()
        else:
            self.p0 = np.copy(p0)
        self.dp0 = np.zeros(3)
        self.in_contact = False
        self.slipping = False

    def compute_force(self, project_in_friction_cone):
        M = self.data.oMf[self.frame_id]
        self.p = M.translation
        delta_p = self.p0 - self.p

        R = se3.SE3(M.rotation, 0*M.translation)
        v_local = se3.getFrameVelocity(self.model, self.data, self.frame_id)
        v_world = (R.act(v_local)).linear

        dJv_local = se3.getFrameAcceleration(self.model, self.data, self.frame_id)
        dJv_local.linear += np.cross(v_local.angular, v_local.linear, axis=0)
        dJv_world = (R.act(dJv_local)).linear

        self.v = v_world
        self.dJv = dJv_world
        if(self.slipping):
            self.dp0 = self.v - self.v.dot(self.normal)*self.normal
            
        self.f = self.K@delta_p + self.B@(self.dp0-v_world)
        
        
        if(project_in_friction_cone):
            self.dp0 = zero(3)
            # check whether point is in contact
            if(delta_p.T @ self.normal <= 0.0):
                self.f = zero(3)
                self.v = zero(3)
                self.dJv = zero(3)
                if(self.in_contact):
                    self.in_contact = False
                    print("\nINFO: contact %s broken!"%(self.frame_name), delta_p.T, self.normal.T)
            else:
                if(not self.in_contact):
                    self.in_contact = True
                    print("\nINFO: contact %s made!"%(self.frame_name))
                # check whether contact force is outside friction cone
                f_N = self.f.dot(self.normal)   # norm of normal force
                f_T = self.f - f_N*self.normal  # tangential force (3d)
                f_T_norm = norm(f_T)            # norm of tangential force
                if(f_T_norm > self.mu*f_N):  # contact is slipping 
                    t_dir = f_T / f_T_norm  # direction of tangential force
                    # saturate force at the friction cone boundary
                    f_T = self.mu*f_N*t_dir                    
                    self.f = f_N*self.normal + f_T
                                        
                    # update anchor point assuming anchor point vel is equal to contact point vel
                    self.dp0 = self.v - self.v.dot(self.normal)*self.normal                    
                    # f = K@(p0-p) + B@(v0-v) => p0 = p + f/K - B@(v0-v)/K
                    self.p0 = self.p +self.Kinv @ (self.f - self.B@(self.dp0-self.v))
                    
                    if(self.slipping==False):
                        self.slipping = True
                        print('INFO: contact %s started slipping'%(self.frame_name), f_T_norm-self.mu*f_N)
                    
                elif(self.slipping==True):
                    self.slipping = False
                    print('INFO: contact %s stopped slipping'%(self.frame_name), f_T_norm-self.mu*f_N)
        return self.f
        
            
    def project_force_in_cone(self, f):
        # check whether point is in contact
        f_N = f.dot(self.normal)   # norm of normal force
        if(f_N <= 0.0):
            return zero(3)
            
        # check whether contact force is outside friction cone
        f_T = f - f_N*self.normal       # tangential force (3d)
        f_T_norm = norm(f_T)            # norm of tangential force
        if(f_T_norm > self.mu*f_N):     # contact is slipping 
            t_dir = f_T / f_T_norm      # direction of tangential force
            # saturate force at the friction cone boundary
            f_T = self.mu*f_N*t_dir
            f = f_N*self.normal + f_T                
                
        return f

    def getJacobianWorldFrame(self):
        J_local = se3.getFrameJacobian(self.model, self.data, self.frame_id, se3.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.J = J_local[:3, :]
        return self.J


class RobotSimulator:
    PRINT_N = 500                   # print every PRINT_N time steps
    # update robot configuration in viwewer every DISPLAY_N time steps

    # Class constructor
    def __init__(self, conf, robot, root_joint=None, logFileName=None):
        self.logFileName = logFileName
        if(self.logFileName is not None):
            f = open(logFileName + 'A', 'w')  # Starting from empty file
            f.close()
            f = open(logFileName + 'b', 'w')  # Starting from empty file
            f.close()
            f = open(logFileName + 'xInit', 'w')  # Starting from empty file
            f.close()

        self.conf = conf
        self.expMatHelper = ExponentialMatrixHelper()
        self.assume_A_invertible = False
        self.max_mat_mult = 100
        self.use_second_integral = True
        self.update_expm_N = 1 # update the expm every self.update_expm_N inner simulation steps
        self.fwd_dyn_method = 'Cholseky' # can be either Cholesky, aba, or pinMinv
        self.unilateral_contacts = 'projection' # None, 'QP', 'projection'
        
        # se3.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ], se3.JointModelFreeFlyer())
        self.robot = robot
        self.model = self.robot.model
        self.data = self.robot.data
        self.t = 0.0
        self.f = zero(0)
        nv = self.model.nv  # Dimension of joint velocities vector
        if root_joint is None:  # Basically if we have a floating base
            na = nv
        else:
            na = nv-6  # Remove 6 non-actuated velocities
        # Matrix S used as filter of vetor of inputs U
        self.S = np.hstack((np.zeros((na, nv-na)), np.eye(na, na)))
        self.contacts = []
        self.DISPLAY_T = conf.DISPLAY_T
        self.display_counter = conf.DISPLAY_T
        self.init(conf.q0, None, True)
        
        # for gepetto viewer
        if(conf.use_viewer):
            # se3.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ], se3.JointModelFreeFlyer())
            self.robot_display = robot
            try:
                prompt = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
                if int(prompt[1]) == 0:
                    os.system('gepetto-gui &')
                time.sleep(1)
            except:
                pass
            gepetto.corbaserver.Client()
            self.robot_display.initViewer(windowName='gepetto', loadModel=True)
            self.robot_display.viewer.gui.createSceneWithFloor('world')
            self.robot_display.displayCollisions(False)
            self.robot_display.displayVisuals(True)
            self.robot_display.display(self.q)
            self.gui = self.robot_display.viewer.gui
            self.gui.setCameraTransform('gepetto', conf.CAMERA_TRANSFORM)
            self.gui.setLightingMode('world/floor', 'OFF')

    # Re-initialize the simulator
    def init(self, q0, v0=None, reset_contact_positions=False, p0=None):
        self.first_iter = True
        self.q = q0.copy()
        if(v0 is None):
            self.v = zero(self.robot.nv)
        else:
            self.v = v0.copy()
        self.dv = zero(self.robot.nv)

        if(reset_contact_positions and p0 is None):
            # set the contact anchor points at the current contact positions
            se3.forwardKinematics(self.model, self.data, self.q)
            se3.updateFramePlacements(self.model, self.data)
            for (i,c) in enumerate(self.contacts):
                c.reset_contact_position()
                self.p0[3*i:3*i+3, 0] = c.p0
        elif(p0 is not None):
            # the user is specifying explicitly the anchor points
            for (i,c) in enumerate(self.contacts):
                c.reset_contact_position(p0[3*i:3*i+3])
            self.p0 = np.copy(p0)

        self.compute_forces(compute_data=True)

    # Adds a contact, resets all quantities
    def add_contact(self, frame_name, normal, K, B, mu):
        c = Contact(self.model, self.data, frame_name, normal, K, B, mu)
        self.contacts += [c]
        self.nc = len(self.contacts)
        self.nk = 3*self.nc
        self.f = zero(self.nk)
        self.f_avg = zero(self.nk)
        self.f_avg2 = zero(self.nk)
        self.f_avg_pre_projection = zero(self.nk)
        self.f_avg2_pre_projection = zero(self.nk)
        self.Jc = np.zeros((self.nk, self.model.nv))
        self.K = np.zeros((self.nk, self.nk))
        self.B = np.zeros((self.nk, self.nk))
        self.p0 = zero(self.nk)
        self.dp0 = zero(self.nk)
        self.dp0_qp = zero(self.nk)
        self.p = zero(self.nk)
        self.dp = zero(self.nk)
        self.dJv = zero(self.nk)
        self.a = zero(2*self.nk)
        self.A = np.zeros((2*self.nk, 2*self.nk))
        self.A[:self.nk, self.nk:] = np.eye(self.nk)
        for (i,c) in enumerate(self.contacts):
            self.K[3*i:3*i+3, 3*i:3*i+3] = c.K
            self.B[3*i:3*i+3, 3*i:3*i+3] = c.B
            self.p0[3*i:3*i+3]           = c.p0
        self.D = np.hstack((-self.K, -self.B))
        
        self.debug_dp = zero(self.nk)
        self.debug_dJv = zero(self.nk)
        self.debug_dp_fd  = zero(self.nk)
        self.debug_dJv_fd = zero(self.nk)
        self.x_pred  = zero(2*self.nk)
        self.x_pred2 = zero(2*self.nk)



    def compute_forces(self, compute_data=True):
        '''Compute the contact forces from q, v and elastic model'''
        if compute_data:            
            se3.forwardKinematics(self.model, self.data, self.q, self.v)
            se3.computeJointJacobians(self.model, self.data)
            se3.updateFramePlacements(self.model, self.data)

        for (i,c) in enumerate(self.contacts):
            self.f[3*i:3*i+3] = c.compute_force(self.unilateral_contacts)
            self.p[  3*i:3*i+3] = c.p
            self.p0[ 3*i:3*i+3] = c.p0
            self.dp0[ 3*i:3*i+3] = c.dp0

        return self.f
        
    def forward_dyn(self, tau):
        # self.fwd_dyn_method can be either Cholesky, aba, or pinMinv
        if self.fwd_dyn_method == 'Cholesky':
            return np.linalg.solve(self.data.M, tau - self.data.nle)
        if self.fwd_dyn_method == 'aba':
            return se3.aba(self.model, self.data, self.q, self.v, tau)
        if self.fwd_dyn_method == 'pinMinv':
#            se3.cholesky.decompose(self.model, self.data, self.q)
#            se3.cholesky.computeMinv(self.model, self.data)
            se3.computeMinverse(self.model, self.data, self.q)
            return self.data.Minv @ (tau - self.data.nle)
        raise Exception("Unknown forward dynamics method "+self.fwd_dyn_method)


    def compute_exponential_LDS(self, u, update_expm):
        ''' Compute matrix A and vector a that define the Linear Dynamical System to
            integrate with the matrix exponential.
        '''
        M = self.data.M
        dv_bar = self.forward_dyn(self.S.T@u)
        for (i,c) in enumerate(self.contacts):
            self.p[  3*i:3*i+3] = c.p
            self.p0[ 3*i:3*i+3] = c.p0
            self.dp[ 3*i:3*i+3] = c.v
            self.dJv[3*i:3*i+3] = c.dJv
        # always assume anchor point is not slipping because the spring-damper
        # force is computed based on that assumption and then projected if necessary
        x0 = np.concatenate((self.p-self.p0, self.dp))
        # x0 = np.concatenate((self.p-self.p0, self.dp-self.dp0)) # this works really BAD!
        
        JMinv = np.linalg.solve(M, self.Jc.T).T
        if(update_expm):
            self.Upsilon = self.Jc @ JMinv.T
            self.a[self.nk:] = self.Jc@dv_bar + self.dJv
            self.A[self.nk:, :self.nk] = -self.Upsilon@self.K
            self.A[self.nk:, self.nk:] = -self.Upsilon@self.B
        return x0, dv_bar, JMinv
        
        
    def step(self, u, dt=None, use_exponential_integrator=True, dt_force_pred=None, ndt_force_pred=None,
             update_expm=True):
        if dt is None:
            dt = self.dt

        if self.first_iter:
            self.compute_forces()
            self.first_iter = False

        se3.forwardKinematics(self.model, self.data, self.q, self.v, np.zeros(self.model.nv))
        se3.computeJointJacobians(self.model, self.data)
        se3.updateFramePlacements(self.model, self.data)
        se3.crba(self.model, self.data, self.q)
        se3.nonLinearEffects(self.model, self.data, self.q, self.v)
        
        for (i,c) in enumerate(self.contacts):
            self.Jc[3*i:3*i+3, :] = c.getJacobianWorldFrame()

        # array containing the forces predicting during the time step (meaningful only for exponential integrator)
        if(dt_force_pred is not None):
            f_pred = np.empty((self.nk,ndt_force_pred))*nan
        else:
            f_pred = None
        f_pred_int = None
        
        if(not use_exponential_integrator):
            self.dv = self.forward_dyn(self.S.T@u + self.Jc.T @ self.f)
            v_mean = self.v + 0.5*dt*self.dv
            self.v += self.dv*dt
            self.q = se3.integrate(self.model, self.q, v_mean*dt)            
        else:
            x0, dv_bar, JMinv = self.compute_exponential_LDS(u, update_expm)
                
            # Sanity check on contact point Jacobian and dJv
#            self.compute_dJv_finite_difference()
            # USE THE VALUE COMPUTED WITH FINITE DIFFERENCING FOR NOW
#            self.dJv = np.copy(self.debug_dJv_fd)
#            self.a[self.nk:] = JMinv@(self.S.T@u-h) + self.dJv
            self.logToFile(x0)
            
            if(update_expm):
                if(self.unilateral_contacts=='QP'):
                    self.int_exp_A = compute_integral_expm(self.A, dt)
                    self.dp0_qp = self.solve_friction_QP(x0, dt)
                    x0[self.nk:]     -= self.dp0_qp
                    
                int_x = self.expMatHelper.compute_integral_x_T(self.A, self.a, x0, dt, self.max_mat_mult)
                # store int_x because it may be needed to compute int2_x without updating expm in next iteration
                self.int_x_prev = int_x
            else:
                int_x = self.expMatHelper.compute_next_integral()
            # compute average force
            self.f_avg = self.D @ int_x / dt
            
            if(self.unilateral_contacts=='projection'):
                # project average forces in friction cones
                self.f_avg_pre_projection = np.copy(self.f_avg)
                for (i, c) in enumerate(self.contacts):
                    self.f_avg[3*i:3*i+3] = c.project_force_in_cone(self.f_avg[3*i:3*i+3]) 
            dv_mean = dv_bar + JMinv.T @ self.f_avg
            
            if(self.use_second_integral):
                if(update_expm):
                    int2_x = self.expMatHelper.compute_double_integral_x_T(self.A, self.a, x0, dt, self.max_mat_mult)
                else:
                    int2_x = self.expMatHelper.compute_next_double_integral()
                    int2_x -= dt * self.int_x_prev
                    self.int_x_prev += int_x
                self.f_avg2 = self.D @ int2_x / (0.5*dt*dt)
                
                if(self.unilateral_contacts=='projection'):
                    # project average forces in friction cones
                    self.f_avg2_pre_projection = np.copy(self.f_avg2)
                    for (i, c) in enumerate(self.contacts):
                        self.f_avg2[3*i:3*i+3] = c.project_force_in_cone(self.f_avg2[3*i:3*i+3])
                v_mean = self.v + 0.5*dt*(dv_bar + JMinv.T @ self.f_avg2)
            else:
                v_mean  = self.v + 0.5*dt*dv_mean

            if(dt_force_pred is not None):
                # predict intermediate forces using linear dynamical system (i.e. matrix exponential)
                n = self.A.shape[0]
                C = np.zeros((n+1, n+1))
                C[0:n,   0:n] = self.A
                C[0:n,     n] = self.a
                z = np.concatenate((x0, [1.0]))
                e_TC = expm(dt_force_pred/ndt_force_pred*C)
                for i in range(ndt_force_pred):
                    f_pred[:, i] = self.D @ z[:n]
                    z = e_TC @ z
                    
                # predict also what forces we would get by integrating with the force prediction
                int_x = self.expMatHelper.compute_integral_x_T(self.A, self.a, x0, dt_force_pred, self.max_mat_mult, store=False)
                int2_x = self.expMatHelper.compute_double_integral_x_T(self.A, self.a, x0, dt_force_pred, self.max_mat_mult, store=False)
                D_int_x = self.D @ int_x
                D_int2_x = self.D @ int2_x
                v_mean_pred = self.v + 0.5*dt_force_pred*dv_bar + JMinv.T@D_int2_x/dt_force_pred
                dv_mean_pred = dv_bar + JMinv.T @ D_int_x/dt_force_pred
                v_pred = self.v + dt_force_pred*dv_mean_pred
                q_pred = se3.integrate(self.model, self.q, v_mean_pred*dt_force_pred)
                
                se3.forwardKinematics(self.model, self.data, q_pred, v_pred)
                se3.updateFramePlacements(self.model, self.data)
                f_pred_int = np.zeros(self.nk)
                # comment these lines because they were messing up the anchor point updates
#                for (i,c) in enumerate(self.contacts):
#                    f_pred_int[3*i:3*i+3] = c.compute_force(self.unilateral_contacts)
                    
                # DEBUG: predict forces integrating contact point dynamics while updating robot dynamics M and h
#                t = dt_force_pred/ndt_force_pred
#                f_pred[:, 0] = K_p0 + self.D @ x0
#                x = np.copy(x0)
#                q, v = np.copy(self.q), np.copy(self.v)
#                for i in range(1,ndt_force_pred):
#                    # integrate robot state
#                    dv = np.linalg.solve(M, self.S.T@u - h + self.Jc.T@f_pred[:,i-1])
#                    v_tmp = v + 0.5*dt*dv
#                    v += dv*dt
#                    q = se3.integrate(self.model, q, v_tmp*dt)
#                    # update kinematics
#                    se3.forwardKinematics(self.model, self.data, q, v, np.zeros(self.model.nv))
#                    se3.computeJointJacobians(self.model, self.data)
#                    se3.updateFramePlacements(self.model, self.data)
#                    ii = 0
#                    for c in self.contacts:
#                        J = c.getJacobianWorldFrame()
#                        self.Jc[ii:ii+3, :] = J
#                        self.p[i:i+3] = c.p
#                        self.dp[i:i+3] = c.v
#                        self.dJv[i:i+3] = c.dJv
#                        ii += 3
#                    x0 = np.concatenate((self.p, self.dp))
#                    # update M and h
#                    se3.crba(self.model, self.data, q)
#                    se3.nonLinearEffects(self.model, self.data, q, v)
#                    M = self.data.M
#                    h = self.data.nle
#                    # recompute A and a
#                    JMinv = np.linalg.solve(M, self.Jc.T).T
#                    self.Upsilon = self.Jc@JMinv.T
#                    self.a[self.nk:] = JMinv@(self.S.T@u - h) + self.dJv + self.Upsilon@K_p0
#                    self.A[self.nk:, :self.nk] = -self.Upsilon@self.K
#                    self.A[self.nk:, self.nk:] = -self.Upsilon@self.B
#                    # integrate LDS
#                    dx = self.A @ x + self.a
#                    x += t * dx
#                    f_pred[:, i] = f_pred[:,i-1] + t*self.D @ dx
                    
                # DEBUG: predict forces integrating contact point dynamics with Euler
#                t = dt_force_pred/ndt_force_pred
#                f_pred[:, 0] = K_p0 + self.D @ x0
#                x = np.copy(x0)
#                for i in range(1,ndt_force_pred):
#                    dx = self.A @ x + self.a
#                    x += t * dx
#                    f_pred[:, i] = f_pred[:,i-1] + t*self.D @ dx
                    
                # DEBUG: predict forces assuming constant contact point acceleration (works really bad)
#                df_debug = self.D @ (self.A @ x0 + self.a)
#                dv = np.linalg.solve(M, self.S.T@u - h + self.Jc.T@self.f)
#                ddp = self.Jc @ dv + self.dJv
#                df = -self.K @ self.dp - self.B @ ddp
#                if(norm(df - df_debug)>1e-6):
#                    print("Error:", norm(df - df_debug))
#                f0 = K_p0 + self.D @ x0
#                t = 0.0
#                for i in range(ndt_force_pred):
#                    f_pred[:, i] = f0 + t*df
#                    t += dt_force_pred/ndt_force_pred
                    
                # DEBUG: predict forces assuming constant contact point velocity (works reasonably well)
#                df = -self.K @ self.dp
#                f0 = K_p0 + self.D @ x0
#                t = 0.0
#                for i in range(ndt_force_pred):
#                    f_pred[:, i] = f0 + t*df
#                    t += dt_force_pred/ndt_force_pred

            self.v += dt*dv_mean
            self.q = se3.integrate(self.model, self.q, v_mean*dt)
            self.dv = dv_mean

        # compute forces at the end so that user has access to updated forces
        self.compute_forces()
        self.t += dt
        return self.q, self.v, f_pred, f_pred_int


    def reset(self):
        self.first_iter = True


    def simulate(self, u, dt=0.001, ndt=1, use_exponential_integrator=True):
        ''' Perform ndt steps, each lasting dt/ndt seconds '''
        #        time_start = time.time()
        
        # APPROACH 1
        # I have to compute ndt inner simulation steps
        # I have to log force values for ndt_force time steps, with ndt_force>=ndt
        # this means that for each simulation step I have to compute ndt_force/ndt forces
        # to simplify things I will assume that ndt_force/ndt is an integer
#        n_f_pred = int(self.ndt_force/ndt)
        
        # APPROACH 2:
        # Predict all forces during first inner simulation step
        
        # forces computed in the inner simulation steps (ndt)
        self.f_inner = np.zeros((self.nk, ndt))
        self.F_avg  = np.zeros((self.nk, ndt))
        self.F_avg2 = np.zeros((self.nk, ndt))
        self.F_avg_pre_projection  = np.zeros((self.nk, ndt))
        self.F_avg2_pre_projection = np.zeros((self.nk, ndt))
        
        # forces predicted at the first of the inner simulation steps
#        self.f_pred = np.zeros((self.nk, self.ndt_force))
        update_expm_counter = 1
        for i in range(ndt):
            update_expm_counter -= 1
            if(update_expm_counter==0):
                update_expm_counter = self.update_expm_N
                update_expm = True
            else:
                update_expm = False
                
            if(i==0):
                self.q, self.v, self.f_pred, self.f_pred_int = self.step(u, dt/ndt, 
                                                    use_exponential_integrator, dt, ndt, update_expm=True)
                self.f_inner[:,0] = self.f_pred[:,0]
            else:
                self.q, self.v, tmp1, tmp2 = self.step(u, dt/ndt, use_exponential_integrator, update_expm=update_expm)
                self.f_inner[:,i] = np.copy(self.f)
            
            self.F_avg[:,i]  = np.copy(self.f_avg)
            self.F_avg2[:,i] = np.copy(self.f_avg2)
            self.F_avg_pre_projection[:,i]  = np.copy(self.f_avg_pre_projection)
            self.F_avg2_pre_projection[:,i] = np.copy(self.f_avg2_pre_projection)

            self.display(self.q, dt/ndt)

        return self.q, self.v, self.f
        
        
    def display(self, q, dt):
        if(self.conf.use_viewer):
            self.display_counter -= dt
            if self.display_counter <= 0:
                self.robot_display.display(q)
                self.display_counter = self.DISPLAY_T
                
    
    def solve_friction_QP(self, x0, dt):
        from quadprog import solve_qp
        '''
        Solve a strictly convex quadratic program
        
        Minimize     1/2 x^T G x - a^T x
        Subject to   C.T x >= b
        
        Input Parameters:
        G : array, shape=(n, n)
        a : array, shape=(n,)
        C : array, shape=(n, m) matrix defining the constraints
        b : array, shape=(m), default=None, vector defining the constraints
        meq : int, default=0
            the first meq constraints are treated as equality constraints,
            all further as inequality constraints
        Output: a tuple, where the first element is the optimal x.
        '''
        nc, nk = self.nc, self.nk
        # constraint matrix and vector
        C = np.zeros((nc*4, nc*3))
        c = np.zeros(nc*4)
        # matrices of normal and tangential contact directions
        N = np.zeros((nc, nc*3))
        T1 = np.zeros((nc, nc*3))
        T2 = np.zeros((nc, nc*3))
        # cost Hessian matrix and gradient vector
        G = np.identity(2*nc)
        g = np.zeros(2*nc)
        # fill in constraint matrix
        S0 = np.zeros((2*nk,2*nc))
        S1 = np.zeros((2*nk,2*nc))
        for (i, c) in enumerate(self.contacts):
            N[i,:] = c.normal
            T1[i,:] = c.t1
            T2[i,:] = c.t2
            S0[3*i:3*i+3, 2*i+0]        = c.t1
            S0[3*i:3*i+3, 2*i+1]        = c.t2
            S1[nk+3*i:nk+3*i+3, 2*i+0]  = c.t1
            S1[nk+3*i:nk+3*i+3, 2*i+1]  = c.t2

        # TO FIX: align linearized friction cones with current tangential force directions
        muu = self.conf.mu #/np.sqrt(2.)
        
        # Compute matrix to constrain contact forces: C*f >= 0
        C[0*nc:1*nc,:] = +T1 + muu*N
        C[1*nc:2*nc,:] = +T2 + muu*N
        C[2*nc:3*nc,:] = -T1 + muu*N
        C[3*nc:4*nc,:] = -T2 + muu*N
        # compute matrix M that maps anchor point vel dp0 to contact forces
        # at the end of the time step: f = f0 + M*dp0
        Mx = (S1 + self.int_exp_A @ self.A @ S1)
        M = -self.D @ Mx
        xNs = (x0 + self.int_exp_A @ (self.A@x0 + self.a))
        f0 = self.D @ xNs
        # put things together: C*M*dp0 >= -C*f0
        c = -C@f0
        C = C@M
            
        # DEBUG: check we get same prediction with these two methods
#        self.x_pred = self.expMatHelper.compute_x_T(self.A, self.a, x0, dt, self.max_mat_mult)
#        self.x_pred2 = x0 + self.int_exp_A@(self.A@x0 + self.a)
        # END DEBUG
        
        solution = solve_qp(G, g, C.T, c, 0)
        dp_0 = solution[0]            
        dp0_3d = S0[:nk,:] @ dp_0
        
        # DEBUG
        self.x_pred = self.expMatHelper.compute_x_T(self.A, self.a, x0-S1@dp_0, dt, self.max_mat_mult)
        self.x_pred2 = xNs - Mx@dp_0
            
        return dp0_3d
        
        
    def compute_dJv_finite_difference(self):
        # Sanity check on contact point Jacobian and dJv
        self.debug_dp = self.Jc @ self.v
        self.debug_dJv = self.dJv
        eps = 1e-8
        q_next = se3.integrate(self.model, self.q, (self.v)*eps)
        v_next = self.v
        se3.forwardKinematics(self.model, self.data, q_next, v_next, np.zeros(self.model.nv))
        se3.computeJointJacobians(self.model, self.data)
        se3.updateFramePlacements(self.model, self.data)
        p_next = np.zeros_like(self.p)
        dp_next = np.zeros_like(self.dp)
        for (i,c) in enumerate(self.contacts):
            c.compute_force(self.unilateral_contacts)
            p_next[ 3*i:3*i+3] = c.p
            dp_next[3*i:3*i+3] = c.v
        self.debug_dp_fd  = (p_next-self.p)/eps
        self.debug_dJv_fd = (dp_next-self.dp)/eps
        
    def logToFile(self, x0):
        # To improve: file is opened and closed at each iteration
        if (self.logFileName is not None):
            # Writing down A
            with open(self.logFileName + 'A', 'a+') as f:
                np.savetxt(f, self.A.flatten(), '%.18f', '\t')
            with open(self.logFileName + 'b', 'a+') as f:
                np.savetxt(f, [np.asarray(self.a)[:, 0]], '%.18f', '\t')  # All this mess to print it as a row, and no transposing does not help
            with open(self.logFileName + 'xInit', 'a+') as f:
                np.savetxt(f, [np.asarray(x0)[:, 0]], '%.18f', '\t')



'''
def solve_dense_expo_system(self, U, K, B, a, x0, dt):
        n = U.shape[0]
        A = np.zeros((2*n, 2*n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -U@K
        A[n:, n:] = -U@B

#        print "A", A
        if(self.assume_A_invertible):
            int_x, int2_x = compute_double_integral_x_T(A, a, x0, dt,
                                                        compute_also_integral=True,
                                                        invertible_A=True)
        else:
            x = compute_x_T(A, a, x0, dt, invertible_A=False)
            int_x = compute_integral_x_T(A, a, x0, dt, invertible_A=False)
            int2_x = compute_double_integral_x_T(A, a, x0, dt, invertible_A=False)
        return x, int_x, int2_x

    def solve_sparse_exp(self, x0, b, dt):
        debug = False
        if(int(self.t*1e3) % 100 == 0):
            debug = True

        D_x = zero(self.nk)
        D_int_x = zero(self.nk)
        D_int2_x = zero(self.nk)
        dx0 = self.A@x0

        # normalize rows of Upsilon
        U = self.Upsilon.copy()
        for i in range(U.shape[0]):
            U[i, :] = np.abs(U[i, :])/np.sum(np.abs(U[i, :]))

        done = False
        left = range(self.nk)
#        if debug: print "\nU\n", U
        while not done:
            # find elements have at least 10% effect of current item derivative (compared to sum of all elements)
            #            if debug: print "Analyze row", left[0], ':', U[left[0],:]
            ii = np.where(U[left[0], :].A1 > 0.1)[0]
#            ii = np.array([left[0]])
#            ii = np.array([left[0], left[1], left[2]])
#            ii = np.array(left)
            if debug:
                print("Select elements:", ii, "which give %.1f %% coverage" % (1e2*np.sum(U[left[0], ii])))

            k = ii.shape[0]
            Ux = self.Upsilon[ii, :][:, ii]
            Kx = self.K[ii, :][:, ii]
            Bx = self.B[ii, :][:, ii]
            ax = zero(2*k)
            ax[:k] = dx0[ii]
            ax[k:] = dx0[self.nk+ii]
            ax -= np.vstack((self.dp[ii], -Ux@Kx @ self.p[ii] - Ux@Bx@self.dp[ii]))
            ax[k:] += b[ii]
            x0x = np.vstack((self.p[ii], self.dp[ii]))
            x, int_x, int2_x = self.solve_dense_expo_system(Ux, Kx, Bx, ax, x0x, dt)
            Dx = np.hstack((-Kx, -Bx))
            D_x[ii] = Dx @ x
            D_int_x[ii] = Dx @ int_x
            D_int2_x[ii] = Dx @ int2_x

            try:
                for i in ii:
                    left.remove(i)
            except:
                print("\nU\n", U)
                raise

            if len(left) == 0:
                done = True

        return D_x, D_int_x, D_int2_x
'''


''' Moved because of readability
           D_x = zero(self.nk)
           D_int_x = zero(self.nk)
           D_int2_x = zero(self.nk)
           dx0 = self.A*x0

           Ux = self.Upsilon[0::3,0::3]
           Kx = self.K[0::3, 0::3]
           Bx = self.B[0::3, 0::3]
           ax = dx0[0::3, 0]
           ax -= np.vstack((self.dp[0::3,0], -Ux@Kx@self.p[0::3,0] -Ux@Bx@self.dp[0::3,0]))
           ax[4:,0] += b[0::3, 0]
           x0x = np.vstack((self.p[0::3,0], self.dp[0::3,0]))
           x, int_x, int2_x = self.solve_dense_expo_system(Ux, Kx, Bx, ax, x0x, dt)
           D_x[0::3,0] = np.hstack((-Kx, -Bx)) @ x
           D_int_x[0::3,0] = np.hstack((-Kx, -Bx)) @ int_x
           D_int2_x[0::3,0] = np.hstack((-Kx, -Bx)) @ int2_x

           for i in range(4):
               ii = 3*i+1
               Ux = self.Upsilon[ii:ii+2,ii:ii+2]
               Kx = self.K[ii:ii+2, ii:ii+2]
               Bx = self.B[ii:ii+2, ii:ii+2]
               ax = zero(4)
               ax[:2,0] = dx0[ii:ii+2, 0]
               ax[2:,0] = dx0[self.nk+ii:self.nk+ii+2, 0]
               ax -= np.vstack((self.dp[ii:ii+2,0], -Ux@Kx@self.p[ii:ii+2,0] -Ux@Bx@self.dp[ii:ii+2,0]))
               ax[2:,0] += b[ii:ii+2, 0]
               x0x = np.vstack((self.p[ii:ii+2,0], self.dp[ii:ii+2,0]))
               x, int_x, int2_x = self.solve_dense_expo_system(Ux, Kx, Bx, ax, x0x, dt)
               D_x[ii:ii+2,0] = np.hstack((-Kx, -Bx)) @ x
               D_int_x[ii:ii+2,0] = np.hstack((-Kx, -Bx)) @ int_x
               D_int2_x[ii:ii+2,0] = np.hstack((-Kx, -Bx)) @ int2_x
'''
