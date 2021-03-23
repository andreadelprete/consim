
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

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp> // se3.integrate
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/cholesky.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>

#include "consim/object.hpp"
#include "consim/contact.hpp"
#include "consim/simulators/explicit_euler.hpp"

#include <iostream>

using namespace Eigen;

namespace consim 
{

EulerSimulator::EulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD, EulerIntegrationType type):
AbstractSimulator(model, data, dt, n_integration_steps, whichFD, type) 
{
  tau_f_.resize(model.nv); tau_f_.setZero();
}


void EulerSimulator::computeContactForces() 
{
  // with Euler the contact forces need only to be computed for the current state, so we
  // can directly call the method with the current value of q, v and contacts
  nactive_ = computeContactForces_imp(*model_, *data_, q_, v_, tau_f_, contacts_, objects_);
  tau_ += tau_f_;
}


void EulerSimulator::step(const Eigen::VectorXd &tau) 
{
  
  if(!resetflag_){
    throw std::runtime_error("resetState() must be called first !");
  }
  CONSIM_START_PROFILER("euler_simulator::step");
  assert(tau.size() == model_->nv);
  for (int i = 0; i < n_integration_steps_; i++)
    {
      Eigen::internal::set_is_malloc_allowed(false);
      CONSIM_START_PROFILER("euler_simulator::substep");
      // \brief add input control 
      tau_ += tau;
      // \brief joint damping 
      if (joint_friction_flag_){
        tau_ -= joint_friction_.cwiseProduct(v_);
      }
      
      forwardDynamics(tau_, dv_); 
    
      CONSIM_START_PROFILER("euler_simulator::integration");
      /*!< integrate twice */ 
      switch(integration_type_)
      {
        case SEMI_IMPLICIT: 
          vMean_ = v_ + sub_dt*dv_;
          break;
        case EXPLICIT:
          vMean_ = v_ + 0.5*sub_dt*dv_;
          break;
        case CLASSIC_EXPLICIT:
          vMean_ = v_;
          break;
      }
      pinocchio::integrate(*model_, q_, vMean_ * sub_dt, qnext_);
      v_ += dv_ * sub_dt;
      q_ = qnext_;
      CONSIM_STOP_PROFILER("euler_simulator::integration");
      
      // \brief adds contact forces to tau_
      tau_.setZero();
      computeContactForces(); 
      Eigen::internal::set_is_malloc_allowed(true);
      CONSIM_STOP_PROFILER("euler_simulator::substep");
      elapsedTime_ += sub_dt; 
    }
  CONSIM_STOP_PROFILER("euler_simulator::step");
}

}  // namespace consim 
