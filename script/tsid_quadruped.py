import pinocchio as se3
import tsid
import numpy as np
import numpy.matlib as matlib
import os
import gepetto.corbaserver
import time
import commands


class TsidQuadruped:
    ''' Standard TSID formulation for a quadruped robot standing on its point feet.
        - Center of mass task
        - Postural task
        - 3d rigid contact constraint for the specified feet
        - Regularization task for contact forces
    '''
    
    def __init__(self, conf, robot, q0=None, viewer=True):
        self.conf = conf
        self.robot = tsid.RobotWrapper(robot.model, False)
        
        if q0 is None:
            q = robot.model.neutralConfiguration
        else :
            q = np.copy(q0)
#        q = se3.getNeutralConfiguration(robot.model(), conf.srdf, False)
#        q = robot.model().referenceConfigurations["half_sitting"]
        v = np.matrix(np.zeros(robot.nv)).T
                
        # for gepetto viewer
        if(viewer):
            self.robot_display = robot #se3.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ], se3.JointModelFreeFlyer())
            l = commands.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
            if int(l[1]) == 0:
                os.system('gepetto-gui &')
            time.sleep(1)
            gepetto.corbaserver.Client()
            self.robot_display.initViewer(loadModel=True)
            self.robot_display.displayCollisions(False)
            self.robot_display.displayVisuals(True)
            self.robot_display.display(q)
            self.gui = self.robot_display.viewer.gui
            self.gui.setCameraTransform(0, conf.CAMERA_TRANSFORM)
            
        robot = self.robot
        self.model = robot.model()
        
        assert [robot.model().existFrame(name) for name in conf.contact_frames]
        
        formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        formulation.computeProblemData(0.0, q, v)
        data = formulation.data()
        
        contacts = len(conf.contact_frames)*[None]
        self.contact_ids = {}
        for i, name in enumerate(conf.contact_frames):
            contacts[i] =tsid.ContactPoint(name, robot, name, conf.contact_normal, 
                                            conf.mu, conf.fMin, conf.fMax)
            contacts[i].setKp(conf.kp_contact * matlib.ones(3).T)
            contacts[i].setKd(2.0 * np.sqrt(conf.kp_contact) * matlib.ones(3).T)
            self.contact_ids[name] = robot.model().getFrameId(name)
            H_ref = robot.framePosition(data, robot.model().getFrameId(name))
            contacts[i].setReference(H_ref)
            contacts[i].useLocalFrame(False)
            formulation.addRigidContact(contacts[i], conf.w_forceRef, 1.0, 1)
        
        # modify initial robot configuration so that foot is on the ground (z=0)
#        q[2] -= H_rf_ref.translation[2,0] # 0.84    # fix this
#        formulation.computeProblemData(0.0, q, v)
#        data = formulation.data()
#        H_rf_ref = robot.position(data, self.RF)
#        contactRF.setReference(H_rf_ref)
#        formulation.addRigidContact(contactRF, conf.w_forceRef)
                
        comTask = tsid.TaskComEquality("task-com", robot)
        comTask.setKp(conf.kp_com * matlib.ones(3).T)
        comTask.setKd(2.0 * np.sqrt(conf.kp_com) * matlib.ones(3).T)
        formulation.addMotionTask(comTask, conf.w_com, 1, 0.0)
        
        postureTask = tsid.TaskJointPosture("task-posture", robot)
        postureTask.setKp(conf.kp_posture * matlib.ones(robot.nv-6).T)
        postureTask.setKd(2.0 * np.sqrt(conf.kp_posture) * matlib.ones(robot.nv-6).T)
        formulation.addMotionTask(postureTask, conf.w_posture, 1, 0.0)
        
#        self.leftFootTask = tsid.TaskSE3Equality("task-left-foot", self.robot, self.conf.lf_frame_name)
#        self.leftFootTask.setKp(self.conf.kp_foot * np.matrix(np.ones(6)).T)
#        self.leftFootTask.setKd(2.0 * np.sqrt(self.conf.kp_foot) * np.matrix(np.ones(6)).T)
#        self.trajLF = tsid.TrajectorySE3Constant("traj-left-foot", H_lf_ref)
#        
#        self.rightFootTask = tsid.TaskSE3Equality("task-right-foot", self.robot, self.conf.rf_frame_name)
#        self.rightFootTask.setKp(self.conf.kp_foot * np.matrix(np.ones(6)).T)
#        self.rightFootTask.setKd(2.0 * np.sqrt(self.conf.kp_foot) * np.matrix(np.ones(6)).T)
#        self.trajRF = tsid.TrajectorySE3Constant("traj-right-foot", H_rf_ref)
#        
        com_ref = robot.com(data)
        trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)
        
        q_ref = q[7:]
        trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)
        
        comTask.setReference(trajCom.computeNext())
        postureTask.setReference(trajPosture.computeNext())
        
        solver = tsid.SolverHQuadProgFast("qp solver")
        solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)
        
        self.trajCom = trajCom
        self.trajPosture = trajPosture
        self.comTask = comTask
        self.postureTask = postureTask
        self.contacts = contacts
        self.formulation = formulation
        self.solver = solver
        self.q = q
        self.v = v
        
        self.contacts_active = {}
        for name in conf.contact_frames: self.contacts_active[name] = True
        
        
    def integrate_dv(self, q, v, dv, dt):
        v_mean = v + 0.5*dt*dv
        v += dt*dv
        q = se3.integrate(self.model, q, dt*v_mean)
        return q,v
        
    def remove_contact(self, name, transition_time=0.0):
#        self.formulation.addMotionTask(self.rightFootTask, self.conf.w_foot, 1, 0.0)
#        H_ref = self.robot.position(self.formulation.data(), self.contact_ids[name])
#        self.traj[name].setReference(H_ref)
#        self.footTask[name].setReference(self.traj[name].computeNext())
    
        self.formulation.removeRigidContact(name, transition_time)
        self.contacts_active[name] = False
        
    def add_contact(self, name, transition_time=0.0):
#        self.formulation.removeTask(self.rightFootTask.name, 0.0)
        H_ref = self.robot.position(self.formulation.data(), self.contact_ids[name])
        self.contacts[name].setReference(H_ref)
        self.formulation.addRigidContact(self.contacts[name], self.conf.w_forceRef)
        
        self.contacts_active[name] = True
        