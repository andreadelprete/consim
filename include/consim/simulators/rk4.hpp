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
#include "consim/simulators/explicit_euler.hpp"

namespace consim 
{

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
