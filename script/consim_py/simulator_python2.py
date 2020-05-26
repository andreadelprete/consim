""" This is for the sole purpose of debugging the QP method for updating the friction cone  """

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

from cvxopt.solvers import  qp as solverQP
from cvxopt import matrix
#from consim_py.utils.utils_LDS_integral import compute_integral_x_T, compute_double_integral_x_T
from consim_py.utils.exponential_matrix_helper_py2 import ExponentialMatrixHelper

from cvxopt import solvers
solvers.options['show_progress'] = False

class Contact:
    def __init__(self, model, data, frame_name, normal, K, B):
        self.model = model
        self.data = data
        self.frame_name = frame_name
        self.normal = normal
        # hard coded for flat ground 
        self.tangentA = np.array([1., 0., 0.])
        self.tangentB = np.array([0., 1., 0.]) 
        self.K = K
        self.invK = np.linalg.inv(K)
        self.B = B
        self.frame_id = model.getFrameId(frame_name)
        
        self.unilateral = True
        self.active = False  
        self.friction_coeff = 0.3
        self.f = np.zeros(3)
        self.p0 = np.zeros(3)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        # computed through exponentials  exponentials 
        self.pNext = np.zeros(3)
        self.vNext = np.zeros(3)
        self.fNext = np.zeros(3)

        # self.reset_contact_position()
    

    def check_contact(self): 
        if self.data.oMf[self.frame_id].translation[2]<=0.:
            if self.unilateral:
                if not self.active: 
                    self.reset_contact_position()
                    self.active = True  
                    self.in_contact = True  
            else: 
                if not self.active:
                    self.reset_contact_position()
                    self.active = True 
        else: 
            if self.unilateral:
                self.active = False   
                self.in_contact = False   
                self.f = zero(3)
                

    def reset_contact_position(self, p0=None):
        # Initial (0-load) position of the spring
        if not self.active:
            if(p0 is None):
                self.p0 = self.data.oMf[self.frame_id].translation.copy()
            else:
                self.p0 = np.copy(p0)
            self.in_contact = True

    def compute_force(self):
        if self.unilateral and not self.active:
            self.f = np.zeros(3)
            return self.f 
        M = self.data.oMf[self.frame_id]
        self.p = M.translation
        delta_p = self.p0 - self.p

        R = se3.SE3(M.rotation, 0*M.translation)
        v_local = se3.getFrameVelocity(self.model, self.data, self.frame_id)
        v_world = (R.act(v_local)).linear

        dJv_local = se3.getFrameAcceleration(self.model, self.data, self.frame_id)
        dJv_local.linear += np.cross(v_local.angular, v_local.linear, axis=0)
        dJv_world = (R.act(dJv_local)).linear

        self.f = self.K.dot(delta_p) - self.B.dot(v_world)
        self.v = v_world
        self.dJv = dJv_world
        return self.f

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

        self.dt = 1.e-3 
        self.conf = conf
        self.expMatHelper = ExponentialMatrixHelper()
        self.assume_A_invertible = False
        self.max_mat_mult = 100
        self.use_second_integral = True
        self.update_expm_N = 1 # update the expm every self.update_expm_N inner simulation steps
        self.fwd_dyn_method = 'pinMinv' # can be either Cholesky, aba, or pinMinv
        
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
        self.n_active_ = 0 
        self.cone_violation_ = False 
        self.cone_method = "average"  # options = ["qp", "average"]
        self.withEquality = False
        
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
            self.check_contacts() 
        elif(p0 is not None):
            # the user is specifying explicitly the anchor points
            for (i,c) in enumerate(self.contacts):
                c.reset_contact_position(p0[3*i:3*i+3])
            self.p0 = np.copy(p0)

        self.compute_forces(compute_data=True)

    # Adds a contact, resets all quantities
    def add_contact(self, frame_name, normal, K, B):
        c = Contact(self.model, self.data, frame_name, normal, K, B)
        self.contacts += [c]
        self.nc = len(self.contacts)
        self.nk = 3*self.nc
        self.f = zero(self.nk)
        self.Jc = np.zeros((self.nk, self.model.nv))
        self.K = np.zeros((self.nk, self.nk))
        self.B = np.zeros((self.nk, self.nk))
        self.p0 = zero(self.nk)
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

    def check_contacts(self):
        """ loops over all contacts to check active status and update matrix size """
        self.n_active_ = 0 
        for (i,c) in enumerate(self.contacts):
            c.check_contact()
            if c.active:
                self.n_active_ += 1 
        if not (self.f.shape[0]==3*self.n_active_) or self.first_iter:
            self.f = zero(3*self.n_active_)
            self.Jc = np.zeros((3*self.n_active_, self.model.nv))
            self.K = np.zeros((3*self.n_active_, 3*self.n_active_))
            self.B = np.zeros((3*self.n_active_, 3*self.n_active_))
            self.p0 = zero(3*self.n_active_)
            self.p = zero(3*self.n_active_)
            self.dp = zero(3*self.n_active_)
            self.dJv = zero(3*self.n_active_)
            self.a = zero(6*self.n_active_)
            self.A = np.zeros((6*self.n_active_, 6*self.n_active_))
            self.A[:3*self.n_active_, 3*self.n_active_:] = np.eye(3*self.n_active_)
            self.cone_constraints = np.zeros([4*self.n_active_, 3*self.n_active_])
            self.anchor_integrator= np.vstack([.5 * np.eye(3*self.n_active_), np.eye(3*self.n_active_)/self.dt ]) 
            self.qp_Q = matrix(2.*np.identity(3*self.n_active_))
            self.qp_q = matrix([0.]*(3*self.n_active_))
            self.cone_equality = np.zeros([self.n_active_, 3*self.n_active_])
            i_active = 0
            for (i,c) in enumerate(self.contacts):
                if c.active:
                    self.K[3*i_active:3*i_active+3, 3*i_active:3*i_active+3] = c.K.copy()
                    self.B[3*i_active:3*i_active+3, 3*i_active:3*i_active+3] = c.B.copy()
                    self.p0[3*i_active:3*i_active+3]           = c.p0.copy()
                    if self.cone_method == "qp":
                        self.cone_constraints[4*i_active, 3*i_active:3*i_active+3] = (c.friction_coeff/np.sqrt(2))*c.normal
                        self.cone_constraints[4*i_active, 3*i_active:3*i_active+3] -= c.tangentA
                        self.cone_constraints[4*i_active+1, 3*i_active:3*i_active+3] = (c.friction_coeff/np.sqrt(2))*c.normal
                        self.cone_constraints[4*i_active+1, 3*i_active:3*i_active+3] += c.tangentA
                        self.cone_constraints[4*i_active+2, 3*i_active:3*i_active+3] = (c.friction_coeff/np.sqrt(2))*c.normal
                        self.cone_constraints[4*i_active+2, 3*i_active:3*i_active+3] -= c.tangentB
                        self.cone_constraints[4*i_active+3, 3*i_active:3*i_active+3] = (c.friction_coeff/np.sqrt(2))*c.normal
                        self.cone_constraints[4*i_active+3, 3*i_active:3*i_active+3] += c.tangentB
                        self.cone_equality[i_active,:] = c.normal
                    i_active += 1 
            self.D = np.hstack((-self.K, -self.B))



    def compute_forces(self, compute_data=True):
        '''Compute the contact forces from q, v and elastic model'''
        if compute_data:            
            se3.forwardKinematics(self.model, self.data, self.q, self.v)
            se3.computeJointJacobians(self.model, self.data)
            se3.updateFramePlacements(self.model, self.data)

        self.check_contacts()
        i_active = 0 
        for (i,c) in enumerate(self.contacts):
            if c.active: 
                self.f[3*i_active:3*i_active+3] = c.compute_force()
                i_active+= 1
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
            return self.data.Minv.dot(tau - self.data.nle)
        raise Exception("Unknown forward dynamics method "+self.fwd_dyn_method)


    def compute_exponential_LDS(self, u):
        ''' Compute matrix A and vector a that define the Linear Dynamical System to
            integrate with the matrix exponential.
        '''
        M, h = self.data.M, self.data.nle
        dv_bar = self.forward_dyn(self.S.T.dot(u))
        i_active = 0 
        for (i,c) in enumerate(self.contacts):
            if c.active:
                self.p[  3*i_active:3*i_active+3] = c.p
                self.dp[ 3*i_active:3*i_active+3] = c.v
                self.dJv[3*i_active:3*i_active+3] = c.dJv
                i_active += 1 
        self.x0 = np.concatenate((self.p, self.dp))
        # TODO: I removed p0 from self.x0, must be added here 
        JMinv = np.linalg.solve(M, self.Jc.T).T
        self.Upsilon = self.Jc.dot(JMinv.T)
        self.a[3*self.n_active_:] = JMinv.dot((self.S.T.dot(u)-h)) + self.dJv + self.Upsilon.dot(self.K.dot(self.p0))
        self.A[3*self.n_active_:, :3*self.n_active_] = -self.Upsilon.dot(self.K)
        self.A[3*self.n_active_:, 3*self.n_active_:] = -self.Upsilon.dot(self.B)
        return self.x0, dv_bar, JMinv


    def step(self, u, dt=None, use_exponential_integrator=True, dt_force_pred=None, ndt_force_pred=None,
             update_expm=True):
        if dt is None:
            dt = self.dt
        else:
            self.dt = dt 


        if self.first_iter:
            self.compute_forces()
            self.first_iter = False

        se3.forwardKinematics(self.model, self.data, self.q, self.v, np.zeros(self.model.nv))
        se3.computeJointJacobians(self.model, self.data)
        se3.updateFramePlacements(self.model, self.data)
        se3.crba(self.model, self.data, self.q)
        se3.nonLinearEffects(self.model, self.data, self.q, self.v)
        
        i_active = 0 
        for (i,c) in enumerate(self.contacts):
            if c.active:
                self.Jc[3*i_active:3*i_active+3, :] = c.getJacobianWorldFrame()
                i_active += 1 

        # array containing the forces predicting during the time step (meaningful only for exponential integrator)
        if(dt_force_pred is not None):
            f_pred = np.empty((3*self.n_active_,ndt_force_pred))*nan
        else:
            f_pred = None
        f_pred_int = None
        

        if(not use_exponential_integrator):            
            self.dv = self.forward_dyn(self.S.T.dot(u) + self.Jc.T.dot(self.f))
            v_mean = self.v + 0.5*dt*self.dv
            self.v += self.dv*dt
            self.q = se3.integrate(self.model, self.q, v_mean*dt)
            
        else:

            if self.n_active_ > 0:
                x0, dv_bar, JMinv = self.compute_exponential_LDS(u)
                self.logToFile(x0)
                int_x = self.expMatHelper.compute_integral_x_T(self.A, self.a, x0, dt, self.max_mat_mult)
                int2_x = self.expMatHelper.compute_double_integral_x_T(self.A, self.a, x0, dt, self.max_mat_mult)
                D_int_x = self.D.dot(int_x)
                D_int2_x = self.D.dot(int2_x)

                f_projection = self.checkFrictionCone(D_int_x ,dt)

                if self.cone_violation_:
                    dv_mean = dv_bar + JMinv.T.dot(f_projection) 
                    v_mean  = self.v + 0.5*dt*dv_mean
                else:
                    dv_mean = dv_bar + JMinv.T.dot(D_int_x)/dt
                    v_mean = self.v + .5 *dt*dv_bar + JMinv.T.dot(D_int2_x)/dt

                self.v += dt*dv_mean
                self.q = se3.integrate(self.model, self.q, v_mean*dt)
                self.dv = dv_mean
            else:
                self.dv = self.forward_dyn(self.S.T.dot(u))
                v_mean = self.v + .5*dt*self.dv 
                self.q = se3.integrate(self.model, self.q, v_mean*dt)
                self.v += dt*self.dv

        # compute forces at the end so that user has access to updated forces
        self.compute_forces()
        self.t += dt
        return self.q, self.v, f_pred, f_pred_int

    def reset(self):
        self.first_iter = True

    def simulate(self, u, dt=0.001, ndt=1, use_exponential_integrator=True):
        ''' Perform ndt steps, each lasting dt/ndt seconds '''

        self.f_inner = np.zeros((self.nk, ndt))
        update_expm_counter = 1
        for i in range(ndt):
            self.q, self.v, self.f_pred, self.f_pred_int = self.step(u, dt/ndt, 
                                                    use_exponential_integrator, dt, ndt, update_expm=True)
            # self.f_inner[:,0] = self.f_pred[:,0]
            
            self.display(self.q, dt/ndt)

        return self.q, self.v, self.f
        
    def display(self, q, dt):
        if(self.conf.use_viewer):
            self.display_counter -= dt
            if self.display_counter <= 0:
                self.robot_display.display(q)
                self.display_counter = self.DISPLAY_T
                
    
    def compute_dJv_finite_difference(self):
        # Sanity check on contact point Jacobian and dJv
        self.debug_dp = self.Jc.dot(self.v)
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
            c.compute_force()
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

    def checkFrictionCone(self,D_int_x ,dt): 
        """ loops over contact forces and checks friction cone, then updates with QP """
        self.cone_violation_ = False
        f_average = self.K.dot(self.p0) + D_int_x/dt  # average force over integration interval 
        # loop over contacts 
        self.f_projection = zero(3*self.n_active_)
        self.computePredictedXandF(dt)
        i_active = 0 
        for i,c in enumerate(self.contacts):
            if not c.active:
                continue
            if not c.unilateral:
                f_projection[3*i_active:3*i_active+3] = f_average[3*i_active:3*i_active+3]
                continue

            normal_force_value = c.normal.T.dot(f_average[3*i_active:3*i_active+3])
            normal_force = normal_force_value*c.normal
            if normal_force_value<0.:
                # pulling force 
                self.cone_violation_ = True 
                self.f_projection[3*i_active:3*i_active+3] = zero(3)
                continue
            else:
                "checking for tangent force "
                tangent_force = f_average[3*i_active:3*i_active+3] - normal_force
                tangent_norm = np.linalg.norm(tangent_force)
                if (tangent_norm>c.friction_coeff*normal_force_value):
                    if self.cone_method=="average":
                        f_projection[3*i_active:3*i_active+3]  = normal_force + (c.friction_coeff*normal_force_value/tangent_norm)*tangent_force
                        c.p0 = c.invK.dot(f_projection[3*i_active:3*i_active+3] + c.K.dot(c.pNext) + c.B.dot(c.vNext))
                    elif self.cone_method=="qp":
                        pass 
                    else:
                        raise Exception("Cone Update Method not Recognized")
                    self.cone_violation_ = True 
                else:
                    self.f_projection[3*i_active:3*i_active+3] = f_average[3*i_active:3*i_active+3]

            i_active += 1 
        
        # method 2 is applied when all constraints are satisfied 
        if self.cone_violation_ and  self.cone_method=="qp":
            self.qpAnchorPointUpdate(f_average)

            
        return self.f_projection


    def computePredictedXandF(self, dt):
        """ cmpute pNext, vNext and fNext that shoud occure at the end of the integration step 
            * computes e^{dt*A}
            * computes \int{e^{dt*A}}
            * computes predictedXf = edtA x0 + int_edtA_ * b 
            * updates predicted x & v in each contact point 
        """
        self.e_Adt = self.expMatHelper.expm(dt*self.A)
        self.int_e_Adt = self.expMatHelper.compute_integral_expm(self.A, dt)
        xEnd = self.e_Adt.dot(self.x0) + self.int_e_Adt.dot(self.a) 
        fEnd = self.K.dot(self.p0) + self.D.dot(xEnd) 
        i_active = 0
        for i, c in enumerate(self.contacts):
            if c.active:
                c.pNext = xEnd[3*i_active:3*i_active+3].copy() 
                c.vNext = xEnd[3*(i_active+self.n_active_):3*(i_active+self.n_active_)+3].copy()
                c.fNext = fEnd[3*i_active:3*i_active+3].copy()
                i_active += 1 


    def qpAnchorPointUpdate(self, f_average):
        "updating anchor through QP"
        
        D_inte_integrator = self.D.dot(self.int_e_Adt).dot(self.anchor_integrator)  

        Aineq = matrix(-self.cone_constraints.dot(D_inte_integrator)) 
        bineq = matrix(self.cone_constraints.dot(f_average))
        Aeq = matrix(self.cone_equality.dot(D_inte_integrator))
        beq = matrix(np.zeros(self.n_active_))

    
        if self.withEquality:
            soln = solverQP(self.qp_Q, self.qp_q, G=-Aineq, h=bineq, A=Aeq, b=beq)
        else:
            soln = solverQP(self.qp_Q, self.qp_q, G=-Aineq, h=bineq)


        dp = np.resize(np.array(soln['x']), 3*self.n_active_)

        i_active = 0 
        for i,c in enumerate(self.contacts):
            if not c.active: 
                continue
            c.p0 += .5 * self.dt * dp[3*i_active:3*i_active+3] 
            self.f_projection[3*i_active:3*i_active+3] = c.K.dot(c.p0 - c.pNext) - c.B.dot(c.vNext)
            i_active += 1 
