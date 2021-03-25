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
       * Implicit Euler first oder step 
      */
      void step(const Eigen::VectorXd &tau) override;

      double get_avg_iteration_number() const;
      void set_contact_stabilization_gains(double kp, double kd);

    protected:      
      void computeContactForces();
            
      Eigen::MatrixXd KKT_mat_;
      Eigen::VectorXd KKT_vec_;      // Newton step expressed in tangent space

      Eigen::MatrixXd MinvJcT_;
      Eigen::MatrixXd Jc_;
      Eigen::VectorXd dJv_;
      DiagonalMatrixXd K_;
      DiagonalMatrixXd B_;
      Eigen::VectorXd lambda_;       // contact forces
      Eigen::VectorXd tau_plus_JT_f_;

      Eigen::PartialPivLU<Eigen::MatrixXd> KKT_LU_;

      std::vector<ContactPoint *> contactsCopy_;
      double avg_iteration_number_; // average number of iterations during last call to step
      double regularization_;       // regularization parameter
      double kp_, kd_;              // feedback gains for contact stabilization
  }; // class RigidEulerSimulator

} // namespace consim 
