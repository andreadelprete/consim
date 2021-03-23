
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
#include "consim/simulators/rk4.hpp"

#include <iostream>

using namespace Eigen;

// TODO: sqr already defined in contact.cpp 
#define sqr(x) (x * x)

namespace consim {

/* ____________________________________________________________________________________________*/
/** 
 * RK4 Simulator Class
 * for one integration step, once cotact status and forces are determined they don't change 
 *  
*/

RK4Simulator::RK4Simulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD):
EulerSimulator(model, data, dt, n_integration_steps, whichFD, EXPLICIT) {
  for(int i = 0; i<4; i++){
    qi_.push_back(Eigen::VectorXd::Zero(model.nq));
    vi_.push_back(Eigen::VectorXd::Zero(model.nv));
    dvi_.push_back(Eigen::VectorXd::Zero(model.nv)); 
  }

  rk_factors_.push_back(1.); rk_factors_.push_back(.5); rk_factors_.push_back(.5); rk_factors_.push_back(1.); 
}

int RK4Simulator::computeContactForces(const Eigen::VectorXd &q, const Eigen::VectorXd &v, std::vector<ContactPoint*> &contacts) 
{
  // with RK4 the contact forces must be computed also for 3 intermediate states (2 in the middle of the time step and 1 at the end)
  // so we need to specify different values of q, v and contacts
  int newActive = computeContactForces_imp(*model_, *data_, q, v, tau_f_, contacts, objects_);
  tau_ += tau_f_;
  return newActive;
}


void RK4Simulator::step(const Eigen::VectorXd &tau) 
{
  if(!resetflag_){
    throw std::runtime_error("resetState() must be called first !");
  }
  CONSIM_START_PROFILER("rk4_simulator::step");
  assert(tau.size() == model_->nv);
  for (int i = 0; i < n_integration_steps_; i++)
    {
      // Eigen::internal::set_is_malloc_allowed(false);
      CONSIM_START_PROFILER("rk4_simulator::substep");
      // \brief add input control 
      tau_ += tau;
      // \brief joint damping 
      if (joint_friction_flag_){
        tau_ -= joint_friction_.cwiseProduct(v_);
      }

      qi_[0] = q_; 
      vi_[0] = v_; 

      vMean_.setZero(); dv_.setZero();

      for(int j = 0; j<3; j++){
        forwardDynamics(tau_, dvi_[j], &qi_[j], &vi_[j]); 
        pinocchio::integrate(*model_,  q_, vi_[j] * sub_dt * rk_factors_[j+1], qi_[j+1]);
        vi_[j+1] = v_ +  dvi_[j] * sub_dt * rk_factors_[j+1]  ; 

        vMean_.noalias() +=  vi_[j]/(rk_factors_[j]*6) ; 
        dv_.noalias()    += dvi_[j]/(rk_factors_[j]*6) ; 

        // create a copy of the current contacts
        for(auto &cp: contactsCopy_){
          delete cp;
        }
        contactsCopy_.clear();
        for(auto &cp: contacts_){
          contactsCopy_.push_back(new ContactPoint(*cp));
        }
        // compute contact forces and add J^T*f to tau
        tau_ = tau;
        computeContactForces(qi_[j+1], vi_[j+1], contactsCopy_); 
      }

      forwardDynamics(tau_, dvi_[3], &qi_[3], &vi_[3]); 

      vMean_.noalias() +=  vi_[3]/(rk_factors_[3]*6) ; 
      dv_.noalias()    += dvi_[3]/(rk_factors_[3]*6) ; 

      v_ += dv_ * sub_dt;
      pinocchio::integrate(*model_, q_, vMean_ * sub_dt, qnext_);
      q_ = qnext_;
      
      // compute contact forces and add J^T*f to tau
      tau_.setZero();
      nactive_ = computeContactForces(q_, v_, contacts_);

      // Eigen::internal::set_is_malloc_allowed(true);
      CONSIM_STOP_PROFILER("rk4_simulator::substep");
      elapsedTime_ += sub_dt; 
    }
  CONSIM_STOP_PROFILER("rk4_simulator::step");
}

}  // namespace consim 
