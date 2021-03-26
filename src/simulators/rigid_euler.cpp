
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
regularization_(1e-8),
kp_(0.0),
kd_(0.0)
{
  const int nv = model.nv, nq=model.nq;
  int nx = nq+nv;
  int ndx = 2*nv;
  int nf=0;
  int nkkt = nv+3*nf;

  tau_f_.resize(nv); tau_f_.setZero();
  tau_plus_JT_f_.resize(nv); tau_plus_JT_f_.setZero();

  // KKT_mat_.resize(nkkt, nkkt); KKT_mat_.setZero();
  // KKT_vec_.resize(nkkt); KKT_vec_.setZero();
  // KKT_LU_ = PartialPivLU<MatrixXd>(nkkt);

  // lambda_.resize(3 * nf); lambda_.setZero();
  // K_.resize(3 * nf); K_.setZero();
  // B_.resize(3 * nf); B_.setZero();
  Jc_.resize(3 * nf, nv); Jc_.setZero();
  dJv_.resize(3 * nf); dJv_.setZero();
  // MinvJcT_.resize(nv, 3*nf); MinvJcT_.setZero();
}

double RigidEulerSimulator::get_avg_iteration_number() const { return avg_iteration_number_; }

void RigidEulerSimulator::set_contact_stabilization_gains(double kp, double kd)
{
  kp_ = kp;
  kd_ = kd;
}

void RigidEulerSimulator::computeContactForces()
{
  /**
   * computes the kinematics at the end of the integration step, 
   * runs contact detection 
   * resizes matrices to match the number of active contacts if needed 
   * compute the contact forces of the active contacts 
   **/  
  
  CONSIM_START_PROFILER("rigid_euler_simulator::kinematics");
  pinocchio::forwardKinematics(*model_, *data_, q_, v_, fkDv_);
  pinocchio::computeJointJacobians(*model_, *data_);
  pinocchio::updateFramePlacements(*model_, *data_);
  CONSIM_STOP_PROFILER("rigid_euler_simulator::kinematics");

  CONSIM_START_PROFILER("rigid_euler_simulator::contactDetection");
  detectContacts(contacts_); /*!<inactive contacts get automatically filled with zero here */
  CONSIM_STOP_PROFILER("rigid_euler_simulator::contactDetection");

  if (nactive_>0){
    if (lambda_.size()!=3*nactive_){
      CONSIM_START_PROFILER("rigid_euler_simulator::resizeVectorsAndMatrices");
      // resizeVectorsAndMatrices();
      // nactive_ = newActive_;
      // int nkkt = nv + 3*nactive_;
      // KKT_mat_.resize(nkkt, nkkt); KKT_mat_.setZero();
      // KKT_vec_.resize(nkkt); KKT_vec_.setZero();
      // KKT_LU_ = PartialPivLU<MatrixXd>(nkkt);

      // lambda_.resize(3 * nactive_); lambda_.setZero();
      // K_.resize(3 * nactive_); K_.setZero();
      // B_.resize(3 * nactive_); B_.setZero();
      dJv_.resize(3 * nactive_); dJv_.setZero();
      Jc_.resize(3 * nactive_, model_->nv); Jc_.setZero();
      MinvJcT_.resize(model_->nv, 3*nactive_); MinvJcT_.setZero();
      CONSIM_STOP_PROFILER("rigid_euler_simulator::resizeVectorsAndMatrices");
    }
    
    CONSIM_START_PROFILER("rigid_euler_simulator::contactKinematics");
    int i_active_ = 0; 
    for(auto &cp : contacts_)
    {
      if (!cp->active) continue;
      
      cp->firstOrderContactKinematics(*data_);
      cp->optr->computePenetration(*cp);
      cp->secondOrderContactKinematics(*data_);
      /*!< computeForce updates the anchor point */ 
      cp->optr->contact_model_->computeForce(*cp);
      Jc_.block(3*i_active_,0,3,model_->nv) = cp->world_J_;
      dJv_.segment<3>(3*i_active_) = cp->dJv_ - kp_*cp->delta_x + kd_*cp->v; 

      i_active_ += 1;  
    }
    CONSIM_STOP_PROFILER("rigid_euler_simulator::contactKinematics");
  }
} // RigidEulerSimulator::computeContactForces


void RigidEulerSimulator::step(const Eigen::VectorXd &tau) 
{
  if(!resetflag_){
    throw std::runtime_error("resetState() must be called first !");
  }
  CONSIM_START_PROFILER("rigid_euler_simulator::step");
  assert(tau.size() == model_->nv);

  avg_iteration_number_ = 0.0;
  for (int i = 0; i < n_integration_steps_; i++)
  {
    // Eigen::internal::set_is_malloc_allowed(false);
    CONSIM_START_PROFILER("rigid_euler_simulator::substep");
    if (joint_friction_flag_){
      tau_ -= joint_friction_.cwiseProduct(v_);
    }
    
    /*!< integrate twice with explicit Euler */ 
    pinocchio::crba(*model_, *data_, q_);
    pinocchio::nonLinearEffects(*model_, *data_, q_, v_);
    pinocchio::forwardDynamics(*model_, *data_, tau, Jc_, dJv_, regularization_);
    // cout<<"\ni="<<i<<endl;
    // cout<<"tau="<<tau.transpose()<<endl;
    // cout<<"dJv="<<dJv_.transpose()<<endl;
    // cout<<"J:\n"<<Jc_<<endl;
    // cout<<"dv="<<data_->ddq.transpose()<<endl;
    dv_ = data_->ddq;
    
    pinocchio::integrate(*model_, q_, v_ * sub_dt, qnext_);
    q_ = qnext_;
    v_ += sub_dt*dv_;
    
    computeContactForces();
    // Eigen::internal::set_is_malloc_allowed(true);
    CONSIM_STOP_PROFILER("rigid_euler_simulator::substep");
    elapsedTime_ += sub_dt; 
  }
  avg_iteration_number_ /= n_integration_steps_;
  CONSIM_STOP_PROFILER("rigid_euler_simulator::step");
}

}  // namespace consim 
