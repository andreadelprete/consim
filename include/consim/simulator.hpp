#pragma once

#include <Eigen/Eigen>
#include<Eigen/Cholesky>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/motion.hpp>

#include <MatrixExponential.hpp>
#include <LDSUtility.hpp>

#include "consim/object.hpp"
#include "consim/contact.hpp"
#include "eiquadprog/eiquadprog-fast.hpp"

#define CONSIM_PROFILER
#ifndef CONSIM_PROFILER
#define CONSIM_START_PROFILER(name)
#define CONSIM_STOP_PROFILER(name)
#else
#define CONSIM_START_PROFILER(name) getProfiler().start(name)
#define CONSIM_STOP_PROFILER(name) getProfiler().stop(name)
#endif





namespace consim {

  class AbstractSimulator {
    public:
      AbstractSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps); 
      ~AbstractSimulator(){};

      /**
        * Defines a pinocchio frame as a contact point for contact interaction checking.
        * A contact is a struct containing all the contact information 
      */
      const ContactPoint &addContactPoint(int frame_id, bool unilateral);

      /**
        * Returns the contact points reference 
      */

      const ContactPoint &getContact(int index);

      /**
        * Adds an object to the simulator for contact interaction checking.
      */
      void addObject(Object &obj);

      /**
       * Convenience method to perform a single dt timestep of the simulation. The
       * q and dq after the step are available form the sim.q and sim.dq property.
       * The acceloration during the last step is available from data.ddq;
       * 
       * The jacobian, frames etc. of data is update after the final q/dq values
       * are computed. This allows to use the data object after calling step
       * without the need to re-run the computeXXX methods etc.
       */

      void resetState(const Eigen::VectorXd &q, const Eigen::VectorXd &dq, bool reset_contact_state);

      void setJointFriction(const Eigen::VectorXd &joint_friction);

      

      
      /**
       * Convenience method to perform a single dt timestep of the simulation. 
       * Computes q, dq, ddq, and contact forces for a single time step 
       * results are stored in sim.q, sim.dq, sim.ddq, sim.f 
       */

      virtual void step(const Eigen::VectorXd &tau)=0;

      const Eigen::VectorXd& get_q() const {return q_;};
      const Eigen::VectorXd& get_v() const {return v_;};
      const Eigen::VectorXd& get_dv() const {return dv_;};
      // Eigen::VectorXd get_f() const {return f_};

    
    protected:

      Eigen::VectorXd q_;  
      Eigen::VectorXd qnext_;
      Eigen::VectorXd v_;
      Eigen::VectorXd dv_;
      // Eigen::VectorXd f_;
      Eigen::VectorXd vMean_;
      Eigen::VectorXd tau_;
      unsigned int nc_=0;
      int nactive_; // number of active contacts
      bool resetflag_ = false;
      /**
        * loops over contact points, checks active contacts and sets reference contact positions 
      */
      void checkContact();
      /**
      * computes all relative dynamic and kinematic terms, then checks for the contacts  
      */
      void computeContactState();


      virtual void computeContactForces(const Eigen::VectorXd &dq)=0;

      inline void contactLinearJacobian(unsigned int frame_id);

      const pinocchio::Model *model_;
      pinocchio::Data *data_;

      double dt_;
      int n_integration_steps_;
  
      std::vector<ContactPoint *> contacts_;
      std::vector<Object *> objects_;

      Eigen::MatrixXd frame_Jc_;
      pinocchio::Data::Matrix6x J_;

      Eigen::VectorXd joint_friction_;
      bool joint_friction_flag_ = 0;
      /**
       * contact anchor point update type 
       * 0: x_start = x  
       * 1: every point is moved separately such that the force lies on the cone boundary 
       * 2: qp solution to update contact points consistently 
      **/

      int contact_update = 1; 

  }; // class AbstractSimulator

  class EulerSimulator : public AbstractSimulator
  {
    public: 
      EulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps); 
      ~EulerSimulator(){};

    /**
     * Explicit Euler first oder step 
    */

      void step(const Eigen::VectorXd &tau) override;

    protected:
      const double sub_dt;
      void computeContactForces(const Eigen::VectorXd &dq) override;
      
      


  }; // class EulerSimulator

  class ExponentialSimulator : public AbstractSimulator
  {
    public:
      ExponentialSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, 
                          bool sparse=false, bool invertibleA=false); 
      ~ExponentialSimulator(){};
      void step(const Eigen::VectorXd &tau) override;



    protected:
      /**
       * AbstractSimulator::computeContactState() must be called before  
       * calling ExponentialSimulator::computeContactForces()
       */
      void computeContactForces(const Eigen::VectorXd &dq) override; 
      /**
       * computes average contact force during one integration step 
       * loops over the average force to compute tangential and normal force per contact 
       * projects any violations of the cone onto its boundaries 
       * sets a flag to to switch integration mode to include saturated forces 
       */
      void checkFrictionCone(); 

      bool sparse_; 
      bool invertibleA_;
      const double sub_dt;
      
      Eigen::VectorXd f_;  // total force 
      Eigen::MatrixXd Jc_; // contact jacobian for all contacts 
      Eigen::VectorXd p0_; // reference position for contact 
      Eigen::VectorXd p_; // current contact position 
      Eigen::VectorXd dp_; // contact velocity
      Eigen::VectorXd x0_;
      Eigen::VectorXd a_;
      Eigen::VectorXd b_;
      Eigen::VectorXd xt_;  // containts p and dp for all active contact points 
      Eigen::VectorXd intxt_;
      Eigen::VectorXd int2xt_;
      Eigen::VectorXd kp0_; 
      Eigen::VectorXd dv_bar; 

      
      void resizeVectorsAndMatrices();
      void computeFrameKinematics(unsigned int contact_id); 
      // convenience method to compute terms needed in integration  
      void computeIntegrationTerms();

      // contact acceleration components 
      Eigen::VectorXd dJv_;  
      // contnact frame components
      pinocchio::Motion vilocal_ = pinocchio::Motion::Zero();
      pinocchio::Motion dJvilocal_ = pinocchio::Motion::Zero(); 
      Eigen::VectorXd dJvi_; 
      pinocchio::SE3 frameSE3_ = pinocchio::SE3::Identity();
      //
      Eigen::MatrixXd K;
      Eigen::MatrixXd B;
      Eigen::MatrixXd D;
      Eigen::MatrixXd A; 
      Eigen::MatrixXd Minv_;
      Eigen::MatrixXd JMinv_;
      Eigen::MatrixXd MinvJcT_;
      Eigen::MatrixXd Upsilon_;
      Eigen::MatrixXd JcT_; 
      // expokit 
      expokit::LDSUtility<double, Dynamic> utilDense_;
      // 
      Eigen::VectorXd dvMean_;
      Eigen::VectorXd temp01_;
      Eigen::VectorXd temp02_;
      Eigen::VectorXd temp03_;
      Eigen::VectorXd temp04_;
      Eigen::MatrixXd tempStepMat_; 
      // friction cone 
      Eigen::VectorXd f_avg;  // average force for cone 
      Eigen::VectorXd fpr_;   // projected force on cone boundaries 
      bool cone_flag_ = false; // cone violation status 
      double cone_direction_; // angle of tangential(to contact surface) force 
      double ftan_; 
      unsigned int i_active_; // index of the active contact      

      /**
       * solves a QP to update anchor points of sliding contacts 
       * min || opt_p - p0 ||^2  written as  0.5* opt_p.T * Q * opt_p  + g0.T * opt_p
       * s.t.  Cineq_ * opt_p  + cineq_ >= 0  (inner linear approximation of friction cone)
      **/
      eiquadprog::solvers::EiquadprogFast qp;
      Eigen::MatrixXd Q_cone; 
      Eigen::VectorXd g0_cone; 
      Eigen::MatrixXd Cineq_cone; 
      Eigen::VectorXd cineq_cone; 
      Eigen::VectorXd optP_cone; 
      Eigen::MatrixXd invK;
      

      Eigen::Vector3d xstart_new; // invK*cone_force_offset_03

      // void updateAnchorPoint(const int &contact_index,const int &active_index); 

  }; // class ExponentialSimulator



} // namespace consim 
