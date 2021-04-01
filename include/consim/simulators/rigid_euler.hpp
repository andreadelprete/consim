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
#include <Eigen/Cholesky>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/motion.hpp>

#include "consim/simulators/explicit_euler.hpp"

namespace consim 
{
/*_______________________________________________________________________________*/

  class RigidEulerSimulator : public EulerSimulator
  {
    public: 
      RigidEulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps); 
      ~RigidEulerSimulator(){};

      /**
       * Rigid-contact simulation step 
      */
      void step(const Eigen::VectorXd &tau) override;

      double get_avg_iteration_number() const;
      void set_contact_stabilization_gains(double kp, double kd);
      void set_integration_scheme(int value);

    protected:      
      void computeContactForces(const Eigen::VectorXd &x, std::vector<ContactPoint *> &contacts);
      void computeDynamics(const Eigen::VectorXd &tau, const Eigen::VectorXd &x, Eigen::VectorXd &f);
            
      int integration_scheme_;  // id of the integration scheme (1: Euler, 4: RK4)
      Eigen::MatrixXd Jc_;
      Eigen::VectorXd dJv_;
      Eigen::VectorXd x_;       // system state
      Eigen::VectorXd x_next_;  // next system state
      Eigen::VectorXd f_;       // system dynamics

      std::vector<Eigen::VectorXd> xi_;
      std::vector<Eigen::VectorXd> fi_;
      Eigen::VectorXd rk_factors_a_;
      Eigen::VectorXd rk_factors_b_;

      std::vector<ContactPoint *> contactsCopy_;
      double avg_iteration_number_; // average number of iterations during last call to step
      double regularization_;       // regularization parameter
      double kp_, kd_;              // feedback gains for contact stabilization
  }; // class RigidEulerSimulator

} // namespace consim 
