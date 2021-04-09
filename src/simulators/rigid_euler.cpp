
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
#include <pinocchio/algorithm/jacobian.hpp>

#include "consim/object.hpp"
#include "consim/contact.hpp"
#include "consim/simulators/rigid_euler.hpp"

#include <iostream>

using namespace Eigen;
using namespace std;

namespace consim 
{

/* ____________________________________________________________________________________________*/
/** 
 * RigidEulerSimulator Class 
*/

RigidEulerSimulator::RigidEulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps):
EulerSimulator(model, data, dt, n_integration_steps, 3, EXPLICIT),
integration_scheme_(1),
regularization_(1e-12),
kp_(0.0),
kd_(0.0)
{
  const int nv = model.nv, nq=model.nq;
  int nx = nq+nv;
  int ndx = 2*nv;
  int nf=0;

  tau_f_.resize(nv); tau_f_.setZero();
  x_.resize(nx); x_.setZero();
  x_next_.resize(nx); x_next_.setZero();
  f_.resize(ndx); f_.setZero();
  Jc_.resize(3 * nf, nv); Jc_.setZero();
  dJv_.resize(3 * nf); dJv_.setZero();

  for(int i = 0; i<4; i++){
    xi_.push_back(Eigen::VectorXd::Zero(nx));
    fi_.push_back(Eigen::VectorXd::Zero(ndx));
  }
  rk_factors_a_.resize(4);
  rk_factors_b_.resize(4);
  rk_factors_a_ << 1., 0.5, 0.5, 1.0; 
  rk_factors_b_ << 1./6, 1./3, 1./3, 1./6; 
  
  // rk_factors_a_ << 0., 0.25, 0.5, 0.75; 
  // rk_factors_b_ << 1./4, 1./4, 1./4, 1./4; 
}

double RigidEulerSimulator::get_avg_iteration_number() const { return avg_iteration_number_; }

void RigidEulerSimulator::set_integration_scheme(int value){ integration_scheme_=value; }

void RigidEulerSimulator::set_contact_stabilization_gains(double kp, double kd)
{
  kp_ = kp;
  kd_ = kd;
}

void RigidEulerSimulator::computeContactForces(const Eigen::VectorXd &x, std::vector<ContactPoint *> &contacts)
{
  /**
   * computes the kinematics at the end of the integration step, 
   * runs contact detection 
   * resizes matrices to match the number of active contacts if needed 
   * compute the contact forces of the active contacts 
   **/  
  const int nq = model_->nq, nv = model_->nv;

  CONSIM_START_PROFILER("rigid_euler_simulator::kinematics");
  pinocchio::forwardKinematics(*model_, *data_, x.head(nq), x.tail(nv), fkDv_);
  pinocchio::computeJointJacobians(*model_, *data_);
  pinocchio::updateFramePlacements(*model_, *data_);
  CONSIM_STOP_PROFILER("rigid_euler_simulator::kinematics");

  CONSIM_START_PROFILER("rigid_euler_simulator::contactDetection");
  detectContacts(contacts); /*!<inactive contacts get automatically filled with zero here */
  CONSIM_STOP_PROFILER("rigid_euler_simulator::contactDetection");

  if (nactive_>0){
    if (dJv_.size()!=3*nactive_){
      CONSIM_START_PROFILER("rigid_euler_simulator::resizeVectorsAndMatrices");
      dJv_.resize(3 * nactive_); dJv_.setZero();
      Jc_.resize(3 * nactive_, nv); Jc_.setZero();
      CONSIM_STOP_PROFILER("rigid_euler_simulator::resizeVectorsAndMatrices");
    }
    
    CONSIM_START_PROFILER("rigid_euler_simulator::contactKinematics");
    int i_active_ = 0; 
    for(auto &cp : contacts)
    {
      if (!cp->active) continue;
      
      cp->firstOrderContactKinematics(*data_);
      cp->optr->computePenetration(*cp);
      cp->secondOrderContactKinematics(*data_);
      /*!< computeForce updates the anchor point */ 
      // cp->optr->contact_model_->computeForce(*cp);
      Jc_.block(3*i_active_,0,3,nv) = cp->world_J_;
      dJv_.segment<3>(3*i_active_) = cp->dJv_ - kp_*cp->delta_x + kd_*cp->v; 

      i_active_ += 1;  
    }
    CONSIM_STOP_PROFILER("rigid_euler_simulator::contactKinematics");
  }
} // RigidEulerSimulator::computeContactForces

void RigidEulerSimulator::computeDynamics(const Eigen::VectorXd &tau, const Eigen::VectorXd &x, Eigen::VectorXd &f)
{
  CONSIM_START_PROFILER("rigid_euler_simulator::forwardDynamics");
  const int nq = model_->nq, nv = model_->nv;
  computeContactForces(x, contacts_);
  pinocchio::crba(*model_, *data_, x.head(nq));
  pinocchio::nonLinearEffects(*model_, *data_, x.head(nq), x.tail(nv));
  pinocchio::forwardDynamics(*model_, *data_, tau, Jc_, dJv_, regularization_);
  f.head(nv) = x.tail(nv);
  f.tail(nv) = data_-> ddq;
  CONSIM_STOP_PROFILER("rigid_euler_simulator::forwardDynamics");
}

void RigidEulerSimulator::step(const Eigen::VectorXd &tau) 
{
  const int nq = model_->nq, nv = model_->nv;
  if(!resetflag_){
    throw std::runtime_error("resetState() must be called first !");
  }
  CONSIM_START_PROFILER("rigid_euler_simulator::step");
  assert(tau.size() == model_->nv);

  avg_iteration_number_ = 0.0;
  x_.head(nq) = q_;
  x_.tail(nv) = v_;

  for (int i = 0; i < n_integration_steps_; i++)
  {
    // Eigen::internal::set_is_malloc_allowed(false);
    CONSIM_START_PROFILER("rigid_euler_simulator::substep");
    if (joint_friction_flag_){
      tau_ -= joint_friction_.cwiseProduct(v_);
    }
    
    if(integration_scheme_==1)
    {
      /*!< integrate twice with explicit Euler */ 
      computeDynamics(tau, x_, f_);
    }
    else if(integration_scheme_==2)
    {
      // integrate with RK2
      computeDynamics(tau, x_, f_);
      integrateState(*model_, x_, f_, sub_dt*0.5, xi_[1]);
      computeDynamics(tau, xi_[1], f_);
    }
    else if(integration_scheme_==4)
    {
      // integrate with RK4
      xi_[0] = x_; 
      f_.setZero();
      for(int j = 0; j<3; j++){
        computeDynamics(tau, xi_[j], fi_[j]);
        integrateState(*model_, xi_[0], fi_[j], sub_dt*rk_factors_a_[j+1], xi_[j+1]);
        f_.noalias() +=  fi_[j]*rk_factors_b_[j]; 
      }
      computeDynamics(tau, xi_[3], fi_[3]);
      f_.noalias() +=  fi_[3]*rk_factors_b_[3]; 
    }

    integrateState(*model_, x_, f_, sub_dt, x_next_);
    x_ = x_next_;
    // Eigen::internal::set_is_malloc_allowed(true);
    CONSIM_STOP_PROFILER("rigid_euler_simulator::substep");
    elapsedTime_ += sub_dt; 
  }
  avg_iteration_number_ /= n_integration_steps_;
  q_ = x_.head(nq);
  v_ = x_.tail(nv);

  CONSIM_STOP_PROFILER("rigid_euler_simulator::step");
}

}  // namespace consim 
