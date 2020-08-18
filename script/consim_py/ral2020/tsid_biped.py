import pinocchio as pin
# from pinocchio import libpinocchio_pywrap as pin 
import tsid
import numpy as np
import os
import gepetto.corbaserver
import time
import subprocess


class TsidBiped:
    ''' Standard TSID formulation for a biped robot standing on its rectangular feet.
        - Center of mass task
        - Postural task
        - 6d rigid contact constraint for both feet
        - Regularization task for contact forces
    '''
    
    def __init__(self, conf, dt, urdf, modelPath, srdf):
        self.conf = conf
        self.dt = dt
        vector = pin.StdVec_StdString()
        vector.extend(item for item in modelPath)
        self.robot = tsid.RobotWrapper(urdf, vector, pin.JointModelFreeFlyer(), False)
        robot = self.robot
        self.model = robot.model()
        pin.loadReferenceConfigurations(self.model, srdf, False)        
        try:
            self.q0 = conf.q0
            q = np.copy(conf.q0)
        except:
            self.q0 = self.model.referenceConfigurations["half_sitting"]
            q = np.copy(self.q0)
        self.v0 = v = np.zeros(robot.nv)
        
        assert self.model.existFrame(conf.rf_frame_name)
        assert self.model.existFrame(conf.lf_frame_name)
        
        formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        formulation.computeProblemData(0.0, q, v)
        data = formulation.data()
        contact_Point = np.ones((3,4)) * conf.lz
        contact_Point[0, :] = [-conf.lxn, -conf.lxn, conf.lxp, conf.lxp]
        contact_Point[1, :] = [-conf.lyn, conf.lyp, -conf.lyn, conf.lyp]
        
        contactRF =tsid.Contact6d("contact_rfoot", robot, conf.rf_frame_name, contact_Point, 
                                  conf.contact_normal, conf.mu, conf.fMin, conf.fMax)
        contactRF.setKp(conf.kp_contact * np.ones(6))
        contactRF.setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(6))
        self.RF = robot.model().getFrameId(conf.rf_frame_name)
        H_rf_ref = robot.framePosition(formulation.data(), self.RF)
#        print('H RF\n', H_rf_ref.translation)
        
        # modify initial robot configuration so that foot is on the ground (z=0)
        q[2] -= H_rf_ref.translation[2] + conf.lz
        formulation.computeProblemData(0.0, q, v)
        data = formulation.data()
        H_rf_ref = robot.framePosition(data, self.RF)
#        print('H RF\n', H_rf_ref)
        contactRF.setReference(H_rf_ref)
        if(conf.w_contact>=0.0):
            formulation.addRigidContact(contactRF, conf.w_forceRef, conf.w_contact, 1)
        else:
            formulation.addRigidContact(contactRF, conf.w_forceRef)
        
        contactLF =tsid.Contact6d("contact_lfoot", robot, conf.lf_frame_name, contact_Point, 
                                  conf.contact_normal, conf.mu, conf.fMin, conf.fMax)
        contactLF.setKp(conf.kp_contact * np.ones(6))
        contactLF.setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(6))
        self.LF = robot.model().getFrameId(conf.lf_frame_name)
        H_lf_ref = robot.framePosition(formulation.data(), self.LF)
#        print('H LF\n', H_lf_ref)
        contactLF.setReference(H_lf_ref)
        if(conf.w_contact>=0.0):
            formulation.addRigidContact(contactLF, conf.w_forceRef, conf.w_contact, 1)
        else:
            formulation.addRigidContact(contactLF, conf.w_forceRef)
        
        comTask = tsid.TaskComEquality("task-com", robot)
        comTask.setKp(conf.kp_com * np.ones(3))
        comTask.setKd(2.0 * np.sqrt(conf.kp_com) * np.ones(3))
        if(conf.w_com>0):
            formulation.addMotionTask(comTask, conf.w_com, 1, 0.0)
        
        postureTask = tsid.TaskJointPosture("task-posture", robot)
        try:
            postureTask.setKp(conf.kp_posture)
            postureTask.setKd(2.0 * np.sqrt(conf.kp_posture))
        except:
            postureTask.setKp(conf.kp_posture * np.ones(robot.nv-6))
            postureTask.setKd(2.0 * np.sqrt(conf.kp_posture) * np.ones(robot.nv-6))
        if(conf.w_posture>0):
            formulation.addMotionTask(postureTask, conf.w_posture, 1, 0.0)
        
        self.leftFootTask = tsid.TaskpinEquality("task-left-foot", self.robot, self.conf.lf_frame_name)
        self.leftFootTask.setKp(self.conf.kp_foot * np.ones(6))
        self.leftFootTask.setKd(2.0 * np.sqrt(self.conf.kp_foot) * np.ones(6))
        self.trajLF = tsid.TrajectorypinConstant("traj-left-foot", H_lf_ref)
        if(conf.w_foot>0):
            formulation.addMotionTask(self.leftFootTask, self.conf.w_foot, 1, 0.0)
#        
        self.rightFootTask = tsid.TaskpinEquality("task-right-foot", self.robot, self.conf.rf_frame_name)
        self.rightFootTask.setKp(self.conf.kp_foot * np.ones(6))
        self.rightFootTask.setKd(2.0 * np.sqrt(self.conf.kp_foot) * np.ones(6))
        self.trajRF = tsid.TrajectorypinConstant("traj-right-foot", H_rf_ref)
        if(conf.w_foot>0):
            formulation.addMotionTask(self.rightFootTask, self.conf.w_foot, 1, 0.0)

        self.waistTask = tsid.TaskpinEquality("task-waist", self.robot, self.conf.waist_frame_name)
        self.waistTask.setKp(self.conf.kp_waist * np.ones(6))
        self.waistTask.setKd(2.0 * np.sqrt(self.conf.kp_waist) * np.ones(6))
        self.waistTask.setMask(np.array([0,0,0,1,1,1.]))
        waistID = robot.model().getFrameId(conf.waist_frame_name)
        H_waist_ref = robot.framePosition(formulation.data(), waistID)
        self.trajWaist = tsid.TrajectorypinConstant("traj-waist", H_waist_ref)
        if(conf.w_waist>0):
            formulation.addMotionTask(self.waistTask, self.conf.w_waist, 1, 0.0)
                
        self.tau_max = conf.tau_max_scaling*robot.model().effortLimit[-robot.na:]
        self.tau_min = -self.tau_max
        actuationBoundsTask = tsid.TaskActuationBounds("task-actuation-bounds", robot)
        actuationBoundsTask.setBounds(self.tau_min, self.tau_max)
        if(conf.w_torque_bounds>0.0):
            formulation.addActuationTask(actuationBoundsTask, conf.w_torque_bounds, 0, 0.0)
            
        jointBoundsTask = tsid.TaskJointBounds("task-joint-bounds", robot, dt)
        self.v_max = conf.v_max_scaling * robot.model().velocityLimit[-robot.na:]
        self.v_min = -self.v_max
        jointBoundsTask.setVelocityBounds(self.v_min, self.v_max)
        if(conf.w_joint_bounds>0.0):
            formulation.addMotionTask(jointBoundsTask, conf.w_joint_bounds, 0, 0.0)
        
        com_ref = robot.com(formulation.data())
        self.trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)
        self.sample_com = self.trajCom.computeNext()
        
        q_ref = q[7:]
        self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)
        postureTask.setReference(self.trajPosture.computeNext())
        
        self.sampleLF  = self.trajLF.computeNext()
        self.sample_LF_pos = self.sampleLF.pos()
        self.sample_LF_vel = self.sampleLF.vel()
        self.sample_LF_acc = self.sampleLF.acc()
        
        self.sampleRF  = self.trajRF.computeNext()
        self.sample_RF_pos = self.sampleRF.pos()
        self.sample_RF_vel = self.sampleRF.vel()
        self.sample_RF_acc = self.sampleRF.acc()
        
        self.solver = tsid.SolverHQuadProgFast("qp solver")
        self.solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)
        
        self.comTask = comTask
        self.postureTask = postureTask
        self.contactRF = contactRF
        self.contactLF = contactLF
        self.actuationBoundsTask = actuationBoundsTask
        self.jointBoundsTask = jointBoundsTask
        self.formulation = formulation
        self.q = q
        self.v = v
        self.q0 = np.copy(self.q)
        
        self.contact_LF_active = True
        self.contact_RF_active = True
        
#        self.reset()
        
        data = np.load(conf.DATA_FILE_TSID)
        # assume dt>dt_ref
        r = int(dt/conf.dt_ref)
        
        self.N = int(data['com'].shape[1]/r)
                
        self.contact_phase = data['contact_phase'][::r]
        self.com_pos_ref = np.asarray(data['com'])[:,::r]
        self.com_vel_ref = np.asarray(data['dcom'])[:,::r]
        self.com_acc_ref = np.asarray(data['ddcom'])[:,::r]
        self.x_RF_ref    = np.asarray(data['x_RF'])[:,::r]
        self.dx_RF_ref   = np.asarray(data['dx_RF'])[:,::r]
        self.ddx_RF_ref  = np.asarray(data['ddx_RF'])[:,::r]
        self.x_LF_ref    = np.asarray(data['x_LF'])[:,::r]
        self.dx_LF_ref   = np.asarray(data['dx_LF'])[:,::r]
        self.ddx_LF_ref  = np.asarray(data['ddx_LF'])[:,::r]
        self.cop_ref     = np.asarray(data['cop'])[:,::r]
        
        x_rf   = self.get_placement_RF().translation
        offset = x_rf - self.x_RF_ref[:,0]
#        print("offset", offset)
        for i in range(self.N):
            self.com_pos_ref[:,i] += offset
            self.x_RF_ref[:,i] += offset
            self.x_LF_ref[:,i] += offset
            
        
    def reset(self, q, v, time_before_start):
        print("Reset controller state")
        self.i = -int(time_before_start/self.dt)
        self.t = 0.0
        self.set_com_ref(self.com_pos_ref[:,0], 0*self.com_vel_ref[:,0], 0*self.com_acc_ref[:,0])
        self.formulation.computeProblemData(self.t, q, v)
        if(not self.contact_RF_active):
            self.add_contact_RF()
        if(not self.contact_LF_active):
            self.add_contact_LF()
        
    def compute_control(self, q, v):
        i = self.i
        if(i==0):
            print("Time %.3f Starting to walk"%self.t)
            if self.contact_phase[i] == 'left':
                self.remove_contact_RF()
            else:
                self.remove_contact_LF()
        elif i>0 and i<self.N-1:
            if self.contact_phase[i] != self.contact_phase[i-1]:
                print("Time %.3f Changing contact phase from %s to %s"%(self.t, self.contact_phase[i-1], self.contact_phase[i]))
                if self.contact_phase[i] == 'left':
                    self.add_contact_LF()
                    self.remove_contact_RF()
                else:
                    self.add_contact_RF()
                    self.remove_contact_LF()
        
        if i<0:
            self.set_com_ref(self.com_pos_ref[:,0], 0*self.com_vel_ref[:,0], 0*self.com_acc_ref[:,0])
        elif i<self.N:
            self.set_com_ref(self.com_pos_ref[:,i], self.com_vel_ref[:,i], self.com_acc_ref[:,i])
            self.set_LF_3d_ref(self.x_LF_ref[:,i], self.dx_LF_ref[:,i], self.ddx_LF_ref[:,i])
            self.set_RF_3d_ref(self.x_RF_ref[:,i], self.dx_RF_ref[:,i], self.ddx_RF_ref[:,i])
        
        HQPData = self.formulation.computeProblemData(self.t, q, v)
    
        sol = self.solver.solve(HQPData)
        if(sol.status!=0):
            print("QP problem could not be solved! Error code:", sol.status)
            return None

        u = self.formulation.getActuatorForces(sol)
        
        self.i += 1
        self.t += self.dt
        return np.concatenate((np.zeros(6),u))
        
    def integrate_dv(self, q, v, dv, dt):
        v_mean = v + 0.5*dt*dv
        v += dt*dv
        q = pin.integrate(self.model, q, dt*v_mean)
        return q,v
        
    def get_placement_LF(self):
        return self.robot.framePosition(self.formulation.data(), self.LF)
        
    def get_placement_RF(self):
        return self.robot.framePosition(self.formulation.data(), self.RF)
        
    def set_com_ref(self, pos, vel, acc):
        self.sample_com.pos(pos)
        self.sample_com.vel(vel)
        self.sample_com.acc(acc)
        self.comTask.setReference(self.sample_com)
        
    def set_RF_3d_ref(self, pos, vel, acc):
        self.sample_RF_pos[:3] = pos
        self.sample_RF_vel[:3] = vel
        self.sample_RF_acc[:3] = acc
        self.sampleRF.pos(self.sample_RF_pos)
        self.sampleRF.vel(self.sample_RF_vel)
        self.sampleRF.acc(self.sample_RF_acc)        
        self.rightFootTask.setReference(self.sampleRF)
        
    def set_LF_3d_ref(self, pos, vel, acc):
        self.sample_LF_pos[:3] = pos
        self.sample_LF_vel[:3] = vel
        self.sample_LF_acc[:3] = acc
        self.sampleLF.pos(self.sample_LF_pos)
        self.sampleLF.vel(self.sample_LF_vel)
        self.sampleLF.acc(self.sample_LF_acc)        
        self.leftFootTask.setReference(self.sampleLF)
        
    def get_LF_3d_pos_vel_acc(self, dv):
        data = self.formulation.data()
        H  = self.robot.framePosition(data, self.LF)
        v  = self.robot.frameVelocity(data, self.LF)
        a = self.leftFootTask.getAcceleration(dv)
        return H.translation, v.linear, a[:3]
        
    def get_RF_3d_pos_vel_acc(self, dv):
        data = self.formulation.data()
        H  = self.robot.framePosition(data, self.RF)
        v  = self.robot.frameVelocity(data, self.RF)
        a = self.rightFootTask.getAcceleration(dv)
        return H.translation, v.linear, a[:3]
        
    def remove_contact_RF(self, transition_time=0.0):
#        print("Time %.3f remove contact RF"%self.t)
        H_rf_ref = self.robot.framePosition(self.formulation.data(), self.RF)
        self.trajRF.setReference(H_rf_ref)
        self.rightFootTask.setReference(self.trajRF.computeNext())
    
        self.formulation.removeRigidContact(self.contactRF.name, transition_time)
        self.contact_RF_active = False
        
    def remove_contact_LF(self, transition_time=0.0):   
#        print("Time %.3f remove contact LF"%self.t)
        H_lf_ref = self.robot.framePosition(self.formulation.data(), self.LF)
        self.trajLF.setReference(H_lf_ref)
        self.leftFootTask.setReference(self.trajLF.computeNext())
        
        self.formulation.removeRigidContact(self.contactLF.name, transition_time)
        self.contact_LF_active = False
        
    def add_contact_RF(self, transition_time=0.0):   
#        print("Time %.3f add contact RF"%self.t)
        H_rf_ref = self.robot.framePosition(self.formulation.data(), self.RF)
        self.contactRF.setReference(H_rf_ref)
        if(self.conf.w_contact>=0.0):
            self.formulation.addRigidContact(self.contactRF, self.conf.w_forceRef, self.conf.w_contact, 1)
        else:
            self.formulation.addRigidContact(self.contactRF, self.conf.w_forceRef)
        
        self.contact_RF_active = True
        
    def add_contact_LF(self, transition_time=0.0):      
#        print("Time %.3f add contact LF"%self.t)
        H_lf_ref = self.robot.framePosition(self.formulation.data(), self.LF)
        self.contactLF.setReference(H_lf_ref)
        if(self.conf.w_contact>=0.0):
            self.formulation.addRigidContact(self.contactLF, self.conf.w_forceRef, self.conf.w_contact, 1)
        else:
            self.formulation.addRigidContact(self.contactLF, self.conf.w_forceRef)
        
        self.contact_LF_active = True