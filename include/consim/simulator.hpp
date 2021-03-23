//
//  Copyright (c) 2020-2021 UNITN, NYU
//
//  This file is part of consim
//  consim is free software: you can redistribute it
//  and/or modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation, either version
//  3 of the License, or (at your option) any later version.
//  consim is distributed in the hope that it will be
//  useful, but WITHOUT ANY WARRANTY; without even the implied warranty
//  of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
//  General Lesser Public License for more details. You should have
//  received a copy of the GNU Lesser General Public License along with
//  consim If not, see
//  <http://www.gnu.org/licenses/>.

#pragma once

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/motion.hpp>

#include "consim/object.hpp"
#include "consim/contact.hpp"
#include "eiquadprog/eiquadprog-fast.hpp"

#include "utils/stop-watch.h"

#define CONSIM_PROFILER
#ifndef CONSIM_PROFILER
#define CONSIM_START_PROFILER(name)
#define CONSIM_STOP_PROFILER(name)
#else
#define CONSIM_START_PROFILER(name) getProfiler().start(name)
#define CONSIM_STOP_PROFILER(name) getProfiler().stop(name)
#endif


namespace consim {
  enum EulerIntegrationType{ EXPLICIT=0, SEMI_IMPLICIT=1, CLASSIC_EXPLICIT=2};

  typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagonalMatrixXd;

  /**
   * Detect active/inactive contact points
   */
  int detectContacts_imp(pinocchio::Data &data, std::vector<ContactPoint *> &contacts, std::vector<ContactObject*> &objects);

  /**
   * Compute the contact forces associated to the specified list of contacts and objects. 
   * Moreover, it computes their net effect on the generalized joint torques tau_f.
   */
  int computeContactForces_imp(const pinocchio::Model &model, pinocchio::Data &data, 
                            const Eigen::VectorXd &q, const Eigen::VectorXd &v, Eigen::VectorXd &tau_f, 
                            std::vector<ContactPoint*> &contacts, std::vector<ContactObject*> &objects);

  /** 
   * Integrate in state space.
   */
  void integrateState(const pinocchio::Model &model, const Eigen::VectorXd &x, const Eigen::VectorXd &dx, 
                      double dt, Eigen::VectorXd &xNext);

  /**
   * Compute the difference between x1 and x0, i.e. x1-x0, where x0 and x1 might live on a Lie group.
   */
  void differenceState(const pinocchio::Model &model, const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, 
                      Eigen::VectorXd &dx);

  /** 
   * Derivatives of the function that integrates in state space.
   */
  void DintegrateState(const pinocchio::Model &model, const Eigen::VectorXd &x, const Eigen::VectorXd &dx, 
                      double dt, Eigen::MatrixXd &J);

  /**
   * Derivatives of the function that computes the difference between x1 and x0, i.e. x1-x0, where x0 and x1 might live on a Lie group.
   */
  void DdifferenceState_x0(const pinocchio::Model &model, const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, 
                           Eigen::MatrixXd &J);

  void DdifferenceState_x1(const pinocchio::Model &model, const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, 
                           Eigen::MatrixXd &J);


  class AbstractSimulator {
    public:
      AbstractSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD, EulerIntegrationType type); 
      ~AbstractSimulator(){};

      /**
        * Defines a pinocchio frame as a contact point for contact interaction checking.
        * A contact is a struct containing all the contact information 
      */
      const ContactPoint &addContactPoint(const std::string & name, int frame_id, bool unilateral);

      /**
        * Returns the contact points reference 
      */

      const ContactPoint &getContact(const std::string & name);

      /**
       * checks if contact is active 
       * if active it updates p0 for the specified contact and returns true
       * can update all contact forces
       * can be called after sim.step to update contact forces for next simulation step 
       **/ 
      bool resetContactAnchorPoint(const std::string & name, const Eigen::Vector3d &p0, bool updateContactForces, bool slipping);

      /**
        * Adds an object to the simulator for contact interaction checking.
      */
      void addObject(ContactObject &obj);


      /**
       * Convenience method to perform a single dt timestep of the simulation. The
       * q and dq after the step are available form the sim.q and sim.v property.
       * The acceloration during the last step is available from data.dv;
       * 
       * The jacobian, frames etc. of data is update after the final q/v values
       * are computed. This allows to use the data object after calling step
       * without the need to re-run the computeXXX methods etc.
       */

      void resetState(const Eigen::VectorXd &q, const Eigen::VectorXd &dq, bool reset_contact_state);

      void setJointFriction(const Eigen::VectorXd &joint_friction);

      /**
       * Convenience method to perform a single dt timestep of the simulation. 
       * Computes q, dq, ddq, and contact forces for a single time step 
       * results are stored in sim.q, sim.v, sim.dv, sim.f 
       */

      virtual void step(const Eigen::VectorXd &tau)=0;

      const Eigen::VectorXd& get_q() const {return q_;};
      const Eigen::VectorXd& get_v() const {return v_;};
      const Eigen::VectorXd& get_dv() const {return dv_;};

    protected:
      const pinocchio::Model *model_;
      pinocchio::Data *data_;

      double dt_;
      int n_integration_steps_;
      const double sub_dt;
      
      /** which forward dynamics to use 
       *  1: pinocchio::computeMinverse()
       *  2: pinocchio::aba()
       *  3: cholesky decompostion 
       **/ 
      const int whichFD_; 
      EulerIntegrationType integration_type_; // explicit euler, semi-implicit euler or classic explicit euler

      Eigen::Vector3d  f_tmp;
      
      Eigen::VectorXd q_;  
      Eigen::VectorXd qnext_;
      Eigen::VectorXd v_;
      Eigen::VectorXd dv_;
      Eigen::VectorXd vMean_;
      Eigen::VectorXd tau_;
      unsigned int nc_=0;
      int nactive_; 
      int newActive_;
      double elapsedTime_;  
      bool resetflag_ = false;
      bool contactChange_; 

      std::vector<ContactPoint *> contacts_;
      std::vector<ContactObject *> objects_;

      Eigen::VectorXd joint_friction_;
      bool joint_friction_flag_ = 0;
      
      Eigen::LLT<Eigen::MatrixXd> lltM_; /*!< used for Cholesky FD */ 
      Eigen::MatrixXd inverseM_;  /*!< used for pinocchio::computeMinverse() */ 
      Eigen::VectorXd mDv_; 
      Eigen::VectorXd fkDv_; // filled with zeros for second order kinematics  
      
      /**
        * loops over contact points, checks active contacts and sets reference contact positions 
      */
      void detectContacts(std::vector<ContactPoint *> &contacts);

      void forwardDynamics(Eigen::VectorXd &tau, Eigen::VectorXd &dv, const Eigen::VectorXd *q=NULL, const Eigen::VectorXd *v=NULL); 
      virtual void computeContactForces()=0;
  }; // class AbstractSimulator

/*_______________________________________________________________________________*/

  class EulerSimulator : public AbstractSimulator
  {
    public: 
      EulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD, EulerIntegrationType type); 
      ~EulerSimulator(){};

    /**
     * Explicit Euler first oder step 
    */
      void step(const Eigen::VectorXd &tau) override;

    protected:
      void computeContactForces() override;
      
      Eigen::VectorXd tau_f_; // joint torques due to external forces
  }; // class EulerSimulator

  /*_______________________________________________________________________________*/


  class RK4Simulator : public EulerSimulator
  {
    public: 
      RK4Simulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD);  
      ~RK4Simulator(){};

    /**
     * Runge Kutta 4th order, only applied for integrating acceleration to velocity 
    */

      void step(const Eigen::VectorXd &tau) override;

    protected:
      int computeContactForces(const Eigen::VectorXd &q, const Eigen::VectorXd &v, std::vector<ContactPoint*> &contacts);

    private: 
      //\brief : vectors for the RK4 integration will be allocated in the constructor, depends on state dimension
      std::vector<Eigen::VectorXd> qi_;
      std::vector<Eigen::VectorXd> vi_;
      std::vector<Eigen::VectorXd> dvi_;
      std::vector<double> rk_factors_;

      // std::vector<Eigen::VectorXd> dyi_;
      std::vector<ContactPoint *> contactsCopy_;
  }; // class RK4Simulator

} // namespace consim 
