
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
#include "consim/simulators/base.hpp"

#include <iostream>

using namespace Eigen;

namespace consim 
{

/** 
 * AbstractSimulator Class 
*/

AbstractSimulator::AbstractSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, 
int whichFD, EulerIntegrationType type): 
model_(&model), data_(&data), dt_(dt), n_integration_steps_(n_integration_steps), sub_dt(dt / ((double)n_integration_steps)), 
whichFD_(whichFD), integration_type_(type) {
  q_.resize(model.nq); q_.setZero();
  v_.resize(model.nv); v_.setZero();
  dv_.resize(model.nv); dv_.setZero();
  vMean_.resize(model.nv); vMean_.setZero();
  tau_.resize(model.nv); tau_.setZero();
  qnext_.resize(model.nq); qnext_.setZero();
  inverseM_.resize(model.nv, model.nv); inverseM_.setZero();
  mDv_.resize(model.nv); mDv_.setZero();
  fkDv_.resize(model_->nv); fkDv_.setZero();
} 


const ContactPoint &AbstractSimulator::addContactPoint(const std::string & name, int frame_id, bool unilateral)
{
  ContactPoint *cptr = new ContactPoint(*model_, name, frame_id, model_->nv, unilateral);
	contacts_.push_back(cptr);
  nc_ += 1; /*!< total number of defined contact points */ 
  resetflag_ = false; /*!< cannot call Simulator::step() if resetflag is false */ 
  return getContact(name);
}


const ContactPoint &AbstractSimulator::getContact(const std::string & name)
{
  for (auto &cptr : contacts_) {
    if (cptr->name_==name){
      return *cptr; 
    } 
  }
  throw std::runtime_error("Contact name not recongnized ");
}


bool AbstractSimulator::resetContactAnchorPoint(const std::string & name, const Eigen::Vector3d &p0, bool updateContactForces, bool slipping){
  bool active_contact_found = false;
  for (auto &cptr : contacts_) {
    if (cptr->name_==name){
      if (cptr->active){
        active_contact_found = true;
        cptr->resetAnchorPoint(p0, slipping); 
      }
      break; 
    }
  }
  if (active_contact_found && updateContactForces){
    computeContactForces();
  }
  return active_contact_found;
}

void AbstractSimulator::addObject(ContactObject& obj) {
  objects_.push_back(&obj);
}

void AbstractSimulator::resetState(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, bool reset_contact_state)
{
  q_ = q;
  v_ = dq;
  vMean_ = dq;
  if (reset_contact_state) {
    for (auto &cptr : contacts_) {
      cptr->active = false;
      cptr->f.fill(0);
    }
  }
  computeContactForces();
  for (unsigned int i=0; i<nc_; ++i){
    contacts_[i]->predictedX_ = data_->oMf[contacts_[i]->frame_id].translation(); 
  }
  // elapsedTime_ = 0.;  
  resetflag_ = true;
}

void AbstractSimulator::detectContacts(std::vector<ContactPoint *> &contacts)
{
  contactChange_ = false;  
  newActive_ = detectContacts_imp(*data_, contacts, objects_);
  if(newActive_!= nactive_){
    contactChange_ = true; 
    // std::cout <<elapsedTime_  <<" Number of active contacts changed from "<<nactive_<<" to "<<newActive_<<std::endl; 
  }
  nactive_ = newActive_;
}

void AbstractSimulator::setJointFriction(const Eigen::VectorXd& joint_friction)
{
  joint_friction_flag_= true;
  joint_friction_ = joint_friction;
}

void AbstractSimulator::forwardDynamics(Eigen::VectorXd &tau, Eigen::VectorXd &dv, const Eigen::VectorXd *q, const Eigen::VectorXd *v)
{
  /**
   * Solving the Forward Dynamics  
   *  1: pinocchio::computeMinverse()
   *  2: pinocchio::aba()
   *  3: cholesky decompostion 
   **/  

  // if q and v are not specified -> use the current state
  if(q==NULL)
    q = &q_;
  if(v==NULL)
    v = &v_;

  switch (whichFD_)
      {
        case 1: // slower than ABA
          pinocchio::nonLinearEffects(*model_, *data_, *q, *v);
          mDv_ = tau - data_->nle;
          inverseM_ = pinocchio::computeMinverse(*model_, *data_, *q);
          inverseM_.triangularView<Eigen::StrictlyLower>()
          = inverseM_.transpose().triangularView<Eigen::StrictlyLower>(); // need to fill the Lower part of the matrix
          dv.noalias() = inverseM_*mDv_;
          break;
          
        case 2: // fast
          pinocchio::aba(*model_, *data_, *q, *v, tau);
          dv = data_-> ddq;
          break;
          
        case 3: // fast if some results are reused
          pinocchio::nonLinearEffects(*model_, *data_, *q, *v);
          dv = tau - data_->nle; 
          // Sparse Cholesky factorization
          pinocchio::crba(*model_, *data_, *q);
          pinocchio::cholesky::decompose(*model_, *data_);
          pinocchio::cholesky::solve(*model_,*data_, dv);
          break;
          
        default:
          throw std::runtime_error("Forward Dynamics Method not recognized");
      }
}

}  // namespace consim 
