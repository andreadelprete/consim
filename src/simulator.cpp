#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp> // se3.integrate
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>

#include "consim/object.hpp"
#include "consim/contact.hpp"
#include "consim/simulator.hpp"
#include "consim/utils/stop-watch.hpp"

#include <iostream>

namespace consim {

Simulator::Simulator(float dt, int n_integration_steps, const pinocchio::Model& model, pinocchio::Data& data):
  dt_(dt), n_integration_steps_(n_integration_steps), model_(&model), data_(&data)
{
  q_.resize(model.nq);
  dq_.resize(model.nv);
  dqMean_.resize(model.nv);
  tau_.resize(model.nv);
  J_.resize(6, model.nv);
  frame_Jc_.resize(3, model.nv);
};


const Contact& Simulator::add_contact_point(int frame_id)
{
	Contact* cptr = new Contact();
	cptr->active = false;
  cptr->f.fill(0);
  cptr->friction_flag = false;
  cptr->frame_id = frame_id;
	contacts_.push_back(cptr);
  return get_contact(contacts_.size() - 1);
}


void Simulator::check_contact_() {
  // Loop over all the contact points, over all the objects.
  for (auto &cp : contacts_) {
    cp->x = data_->oMf[cp->frame_id].translation();
    if (cp->active) {
      if (!cp->optr->check_contact(*cp)) {
        cp->active = false;
        cp->f.fill(0);
        cp->friction_flag = false;
      } else {
        // If the contact point is still active, then no need to search for
        // other contacting object (we assume there is only one object acting
        // on a contact point at each timestep).
        continue;
      }
    }

    for (auto &optr : objects_) {
      if (optr->check_contact(*cp)) {
        cp->active = true;
        cp->optr = optr;
        break;
      }
    }
  }
}

inline void Simulator::contact_linear_jacobian_(int frame_id)
{
//  pinocchio::Data::Matrix6x J(6, model_->nv);
  J_.setZero();
//  pinocchio::getFrameJacobian<pinocchio::LOCAL>(*model_, *data_, frame_id, J);
  pinocchio::getFrameJacobian(*model_, *data_, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J_);

  // Rotate the jacobian to have it aligned with the world coordinate frame.
  // Eigen::Matrix3d rotation = data_->oMf[frame_id].rotation();
  // Eigen::Vector3d translation = Eigen::Vector3d::Zero();

  // frame_Jc_ = (pinocchio::SE3(rotation, translation).toActionMatrix() * J_).topRows(3);
  frame_Jc_ = J_.topRows(3);
}

void Simulator::compute_contact_forces_and_torques_(const Eigen::VectorXd& dq) {
  // subtract joint frictions
  if (joint_friction_flag_){
    tau_ -= joint_friction_.cwiseProduct(dq);
  }

  for (auto &cp : contacts_) {
    if (!cp->active) continue;

    // If the contact point is active, compute it's velocity and call the
    // contact model function on the object.
    // TODO: Is there a faster way to compute the contact point velocity than
    //       multiply the jacobian with the generalized velocity from pinocchio?
    contact_linear_jacobian_(cp->frame_id);
    cp->v = frame_Jc_ * dq;
    // printf("contact point position %f\n", cp->normal(2));
    // printf("contact point velocity %f\n", cp->normvel(2));
    // contact force computation is called in th
    cp->optr->contact_model(*cp);
    // printf("contact point force %f\n", cp->f(2));

    tau_ += frame_Jc_.transpose() * cp->f;
    // printf("integration force %f\n", tau_(2));
  }
}

const Contact& Simulator::get_contact(int index)
{
  return *contacts_[index];
}

void Simulator::step(const Eigen::VectorXd& tau) {
  getProfiler().start("simulator::step");
  const double sub_dt = dt_ / ((double)n_integration_steps_);

  assert(tau.size() == model_->nv);

  for (int i = 0; i < n_integration_steps_; i++) {
    // TODO: Support friction models at the joints.

    // Add the user torque;
    tau_ += tau;

    // Compute the acceloration ddq.
    getProfiler().start("pinocchio::aba");
    pinocchio::aba(*model_, *data_, q_, dq_, tau_);
    getProfiler().stop("pinocchio::aba");

    // Integrate the system forward in time.
    dqMean_ = dq_ + data_->ddq * .5 * sub_dt;
    q_ = pinocchio::integrate(*model_, q_, dqMean_ * sub_dt);
    dq_ += data_->ddq * sub_dt;


    // Compute the new data values and contact information after the integration
    // step. This way, if this method returns, the values computed in data and
    // on the contact state are consistent with the q, dq and ddq values.
    compute_terms_and_contact_state_();
  }
  getProfiler().stop("simulator::step");
}

void Simulator::compute_terms_and_contact_state_()
{
  tau_.fill(0);

  // Compute all the terms (mass matrix, jacobians, ...)
  data_->M.fill(0);
  getProfiler().start("pinocchio::computeAllTerms");
  pinocchio::computeAllTerms(*model_, *data_, q_, dq_);
  pinocchio::updateFramePlacements(*model_, *data_);
  getProfiler().stop("pinocchio::computeAllTerms");

  // Contact handling: Detect contact, compute contact forces, compute resulting torques.
  getProfiler().start("compute_contact_forces_and_torques");
  check_contact_();
  compute_contact_forces_and_torques_(dq_);
  getProfiler().stop("compute_contact_forces_and_torques");
}

void Simulator::reset_state(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, bool reset_contact_state)
{
  q_ = q;
  dq_ = dq;
  dqMean_ = dq;

  if (reset_contact_state) {
    for (auto &cptr : contacts_) {
      cptr->active = false;
      cptr->f.fill(0);
      cptr->friction_flag = false;
    }
  }

  compute_terms_and_contact_state_();
}

void Simulator::add_object(Object& obj) {
  objects_.push_back(&obj);
}

void Simulator::set_joint_friction(const Eigen::VectorXd& joint_friction)
{
  joint_friction_flag_=1;
  joint_friction_ = joint_friction;
}


}
