import pinocchio as se3
import tsid
import numpy as np
import os
# import gepetto.corbaserver
import time
import subprocess


class TsidQuadruped:
    ''' Standard TSID formulation for a quadruped robot standing on its point feet.
        - Center of mass task
        - Postural task
        - 3d rigid contact constraint for the specified feet
        - Regularization task for contact forces
    '''

    def __init__(self, conf, dt, robot, com_offset, com_frequency, com_amplitude, q0=None, viewer=False):
        self.conf = conf
        self.dt = dt
        self.robot = tsid.RobotWrapper(robot.model, False)

        if q0 is None:
            q = robot.model.neutralConfiguration
        else:
            q = np.copy(q0)
#        q = se3.getNeutralConfiguration(robot.model(), conf.srdf, False)
#        q = robot.model().referenceConfigurations["half_sitting"]
        v = np.zeros(robot.nv)

        # for gepetto viewer
        if(viewer):
            # se3.RobotWrapper.BuildFromURDF(conf.urdf, [conf.path, ], se3.JointModelFreeFlyer())
            self.robot_display = robot
            prompt = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
            if int(prompt[1]) == 0:
                os.system('gepetto-gui &')
            time.sleep(1)
            gepetto.corbaserver.Client()
            self.robot_display.initViewer(loadModel=True)
            self.robot_display.displayCollisions(False)
            self.robot_display.displayVisuals(True)
            self.robot_display.display(q)
            self.gui = self.robot_display.viewer.gui
            self.gui.setCameraTransform('gepetto', conf.CAMERA_TRANSFORM)

        robot = self.robot
        self.model = robot.model()

#        assert [robot.model().existFrame(name) for name in conf.contact_frames]

        formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        formulation.computeProblemData(0.0, q, v)
        data = formulation.data()

        contacts = len(conf.contact_frames)*[None]
        self.contact_ids = {}
        for i, name in enumerate(conf.contact_frames):
            contacts[i] = tsid.ContactPoint(name, robot, name, conf.contact_normal, conf.mu, conf.fMin, conf.fMax)
            contacts[i].setKp(conf.kp_contact * np.ones(3))
            contacts[i].setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(3))
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
        comTask.setKp(conf.kp_com * np.ones(3))
        comTask.setKd(2.0 * np.sqrt(conf.kp_com) * np.ones(3))
        formulation.addMotionTask(comTask, conf.w_com, 1, 0.0)

        postureTask = tsid.TaskJointPosture("task-posture", robot)
        postureTask.setKp(conf.kp_posture * np.ones(robot.nv-6))
        postureTask.setKd(2.0 * np.sqrt(conf.kp_posture) * np.ones(robot.nv-6))
        formulation.addMotionTask(postureTask, conf.w_posture, 1, 0.0)

#        self.leftFootTask = tsid.TaskSE3Equality("task-left-foot", self.robot, self.conf.lf_frame_name)
#        self.leftFootTask.setKp(self.conf.kp_foot * np.array(np.ones(6)).T)
#        self.leftFootTask.setKd(2.0 * np.sqrt(self.conf.kp_foot) * np.array(np.ones(6)).T)
#        self.trajLF = tsid.TrajectorySE3Constant("traj-left-foot", H_lf_ref)
#
#        self.rightFootTask = tsid.TaskSE3Equality("task-right-foot", self.robot, self.conf.rf_frame_name)
#        self.rightFootTask.setKp(self.conf.kp_foot * np.array(np.ones(6)).T)
#        self.rightFootTask.setKd(2.0 * np.sqrt(self.conf.kp_foot) * np.array(np.ones(6)).T)
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
        for name in conf.contact_frames:
            self.contacts_active[name] = True
            
#        com_pos = np.empty((3, N_SIMULATION))*nan
#        com_vel = np.empty((3, N_SIMULATION))*nan
#        com_acc = np.empty((3, N_SIMULATION))*nan
#        com_pos_ref = np.empty((3, N_SIMULATION))*nan
#        com_vel_ref = np.empty((3, N_SIMULATION))*nan
#        com_acc_ref = np.empty((3, N_SIMULATION))*nan
#        # acc_des = acc_ref - Kp*pos_err - Kd*vel_err
#        com_acc_des = np.empty((3, N_SIMULATION))*nan
        self.offset = com_offset + self.robot.com(self.formulation.data())
        self.two_pi_f = np.copy(com_frequency)
        self.amp = np.copy(com_amplitude)
        self.two_pi_f_amp = self.two_pi_f * self.amp
        self.two_pi_f_squared_amp = self.two_pi_f * self.two_pi_f_amp
        self.sampleCom = self.trajCom.computeNext()
        self.reset(q, v, 0.0)
        
    def reset(self, q, v, time_before_start):
        self.i = 0
        self.t = 0.0
        
    def compute_control(self, q, v):
        self.sampleCom.pos(self.offset + self.amp * np.sin(self.two_pi_f*self.t))
        self.sampleCom.vel(self.two_pi_f_amp * np.cos(self.two_pi_f*self.t))
        self.sampleCom.acc(-self.two_pi_f_squared_amp * np.sin(self.two_pi_f*self.t))
        self.comTask.setReference(self.sampleCom)

        HQPData = self.formulation.computeProblemData(self.t, q, v)
        sol = self.solver.solve(HQPData)
        if(sol.status != 0):
            print("[%d] QP problem could not be solved! Error code:" % (self.i), sol.status)
            return None

        u = self.formulation.getActuatorForces(sol)
        self.i += 1
        self.t += self.dt
        return np.concatenate((np.zeros(6),u))

    def integrate_dv(self, q, v, dv, dt):
        v_mean = v + 0.5*dt*dv
        v += dt*dv
        q = se3.integrate(self.model, q, dt*v_mean)
        return q, v

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
