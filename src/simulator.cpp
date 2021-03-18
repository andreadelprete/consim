
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
#include "consim/simulator.hpp"

#include <iostream>

using namespace Eigen;

// TODO: sqr already defined in contact.cpp 
#define sqr(x) (x * x)

namespace consim {

int detectContacts_imp(pinocchio::Data &data, std::vector<ContactPoint *> &contacts, std::vector<ContactObject*> &objects)
{
  // counter of number of active contacts
  int newActive = 0;
  // Loop over all the contact points, over all the objects.
  for (auto &cp : contacts) {
    cp->updatePosition(data);
    if(cp->active)
    {
      // for unilateral active contacts check if they are still in contact with same object
      if (cp->unilateral) {
        // optr: object pointer
        if (!cp->optr->checkCollision(*cp))
        {
          // if not => set them to inactive and move forward to searching a colliding object
          cp->active = false;
          cp->f.fill(0);
          // cp->friction_flag = false;
        } else {
          newActive += 1;
          // If the contact point is still active, then no need to search for
          // other contacting object (we assume there is only one object acting
          // on a contact point at each timestep).
          continue;
        }
      }
      else {
        // bilateral contacts never break
        newActive += 1;
        continue;
      }
    }
    // if a contact is bilateral and active => no need to search
    // for colliding object because bilateral contacts never break
    if(cp->unilateral || !cp->active) {  
      for (auto &optr : objects) {
        if (optr->checkCollision(*cp))
        {
          cp->active = true;
          newActive += 1; 
          cp->optr = optr;
          if(!cp->unilateral){
            std::cout<<"Bilateral contact with object "<<optr->getName()<<" at point "<<cp->x.transpose()<<std::endl;
          }
          break;
        }
      }
    }
  }
  return newActive;
}

int computeContactForces_imp(const pinocchio::Model &model, pinocchio::Data &data, const Eigen::VectorXd &q, 
                         const Eigen::VectorXd &v, Eigen::VectorXd &tau_f, 
                         std::vector<ContactPoint*> &contacts, std::vector<ContactObject*> &objects) 
{
  pinocchio::forwardKinematics(model, data, q, v);
  pinocchio::computeJointJacobians(model, data);
  pinocchio::updateFramePlacements(model, data);
  /*!< loops over all contacts and objects to detect contacts and update contact positions*/
  
  int newActive = detectContacts_imp(data, contacts, objects);
  CONSIM_START_PROFILER("compute_contact_forces");
  tau_f.setZero();
  for (auto &cp : contacts) {
    if (!cp->active) continue;
    cp->firstOrderContactKinematics(data); /*!<  must be called before computePenetration() it updates cp.v and jacobian*/   
    cp->optr->computePenetration(*cp); 
    cp->optr->contact_model_->computeForce(*cp);
    tau_f.noalias() += cp->world_J_.transpose() * cp->f; 
    // if (contactChange_){
    //     std::cout<<cp->name_<<" p ["<< cp->x.transpose() << "] v ["<< cp->v.transpose() << "] f ["<<  cp->f.transpose() <<"]"<<std::endl; 
    //   }
  }
  CONSIM_STOP_PROFILER("compute_contact_forces");
  return newActive;
}

void integrateState(const pinocchio::Model &model, const Eigen::VectorXd &x, const Eigen::VectorXd &dx, 
                    double dt, Eigen::VectorXd &xNext)
{
  pinocchio::integrate(model, x.head(model.nq), dx.head(model.nv) * dt, xNext.head(model.nq));
  xNext.tail(model.nv) = x.tail(model.nv) + dx.tail(model.nv) * dt;
}

void differenceState(const pinocchio::Model &model, const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, 
                      Eigen::VectorXd &dx)
{
  pinocchio::difference(model, x0.head(model.nq), x1.head(model.nq), dx.head(model.nv));
  dx.tail(model.nv) = x1.tail(model.nv) - x0.tail(model.nv);
}

void DintegrateState(const pinocchio::Model &model, const Eigen::VectorXd &x, const Eigen::VectorXd &dx, 
                      double dt, Eigen::MatrixXd &J)
{
  pinocchio::dIntegrate(model, x.head(model.nq), dx.head(model.nv) * dt, J.topLeftCorner(model.nv, model.nv), pinocchio::ArgumentPosition::ARG1);
  J.bottomRightCorner(model.nv, model.nv) = MatrixXd::Identity(model.nv, model.nv); // * dt;
  J.topRightCorner(model.nv, model.nv).setZero();
  J.bottomLeftCorner(model.nv, model.nv).setZero();
}

void DdifferenceState_x0(const pinocchio::Model &model, const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, 
                        Eigen::MatrixXd &J)
{
  pinocchio::dDifference(model, x0.head(model.nq), x1.head(model.nq), J.topLeftCorner(model.nv, model.nv), pinocchio::ArgumentPosition::ARG0);
  J.bottomRightCorner(model.nv, model.nv) = -MatrixXd::Identity(model.nv, model.nv);
  J.topRightCorner(model.nv, model.nv).setZero();
  J.bottomLeftCorner(model.nv, model.nv).setZero();
}

void DdifferenceState_x1(const pinocchio::Model &model, const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, 
                        Eigen::MatrixXd &J)
{
  pinocchio::dDifference(model, x0.head(model.nq), x1.head(model.nq), J.topLeftCorner(model.nv, model.nv), pinocchio::ArgumentPosition::ARG1);
  J.bottomRightCorner(model.nv, model.nv).setIdentity(model.nv, model.nv);
  J.topRightCorner(model.nv, model.nv).setZero();
  J.bottomLeftCorner(model.nv, model.nv).setZero();
}

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


/* ____________________________________________________________________________________________*/
/** 
 * EulerSimulator Class 
*/

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
