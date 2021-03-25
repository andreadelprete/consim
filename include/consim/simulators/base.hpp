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

#include "consim/simulators/common.hpp"


namespace consim 
{

  class AbstractSimulator 
  {
    public:
      AbstractSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, 
                        int whichFD, EulerIntegrationType type); 
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
      int nactive_;         // number of active contact points
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

} // namespace consim 
