#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp> // se3.integrate
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "consim/object.hpp"
#include "consim/contact.hpp"
#include "consim/simulator.hpp"
// #include "consim/utils/stop-watch.hpp"

#include <iostream>

// TODO: sqr already defined in contact.cpp 
#define sqr(x) (x * x)

namespace consim {

/** 
 * AbstractSimulator Class 
*/

AbstractSimulator::AbstractSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps): 
model_(&model), data_(&data), dt_(dt), n_integration_steps_(n_integration_steps) {
  q_.resize(model.nq); q_.setZero();
  v_.resize(model.nv); v_.setZero();
  dv_.resize(model.nv); dv_.setZero();
  vMean_.resize(model.nv);
  tau_.resize(model.nv);
  frame_Jc_.resize(3, model.nv);
  J_.resize(6, model.nv);
  qnext_.resize(model.nq);
} 

const ContactPoint &AbstractSimulator::addContactPoint(int frame_id, bool unilateral)
{
  ContactPoint *cptr = new ContactPoint();
  cptr->active = false;
  cptr->unilateral = unilateral;
  cptr->f.fill(0);
  cptr->friction_flag = false;
  cptr->frame_id = frame_id;
	contacts_.push_back(cptr);
  nc_ += 1; // increase contact points count  
  resetflag_ = false; // enforce resetState() after adding a contact point
  return getContact(contacts_.size() - 1);
}

const ContactPoint &AbstractSimulator::getContact(int index)
{
  return *contacts_[index];
}

void AbstractSimulator::addObject(Object& obj) {
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
      cptr->friction_flag = false;
    }
  }

  computeContactState();
  computeContactForces(v_);
  resetflag_ = true;
}

void AbstractSimulator::checkContact()
{
  nactive_ = 0; 
  // Loop over all the contact points, over all the objects.
  for (auto &cp : contacts_) {
    cp->x = data_->oMf[cp->frame_id].translation();    

    if(cp->active)
    {
      // for unilateral active contacts check if they are still in contact with same object
      if (cp->unilateral) {
        // optr: object pointer
        if (!cp->optr->checkContact(*cp))
        {
          // if not => set them to inactive and move forward to searching a colliding object
          cp->active = false;
          cp->f.fill(0);
          cp->friction_flag = false;
        } else {
          nactive_ += 1;
          // If the contact point is still active, then no need to search for
          // other contacting object (we assume there is only one object acting
          // on a contact point at each timestep).
          continue;
        }
      }
      else {
        // bilateral contacts never break
        nactive_ += 1;
        continue;
      }
    }
    // if a contact is bilateral and active => no need to search
    // for colliding object because bilateral contacts never break
    if(cp->unilateral || !cp->active) {  
      for (auto &optr : objects_) {
        if (optr->checkContact(*cp))
        {
          cp->active = true;
          nactive_ += 1; 
          cp->optr = optr;
          if(!cp->unilateral){
            cp->x_start = cp->x;
            cout<<"Bilateral contact with object "<<optr->getName()<<" at point "<<cp->x.transpose()<<endl;
          }
          break;
        }
      }
    }
  }
}

void AbstractSimulator::computeContactState()
{
  tau_.fill(0);

  // Compute all the terms (mass matrix, jacobians, ...)
  data_->M.fill(0);
  CONSIM_START_PROFILER("pinocchio::computeAllTerms");
  pinocchio::computeAllTerms(*model_, *data_, q_, v_);
  pinocchio::updateFramePlacements(*model_, *data_);
  CONSIM_STOP_PROFILER("pinocchio::computeAllTerms");

  // Contact handling: Detect contact, compute contact forces, compute resulting torques.
  CONSIM_START_PROFILER("check_contact_state");
  checkContact();
  // computeContactForces(v_);
  CONSIM_STOP_PROFILER("check_contact_state");
}

void AbstractSimulator::setJointFriction(const Eigen::VectorXd& joint_friction)
{
  joint_friction_flag_=true ;
  joint_friction_ = joint_friction;
}

inline void AbstractSimulator::contactLinearJacobian(unsigned int frame_id)
{
  J_.setZero();
  pinocchio::getFrameJacobian(*model_, *data_, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J_);
  frame_Jc_ = J_.topRows<3>();
}

/* ____________________________________________________________________________________________*/
/** 
 * EulerSimulator Class 
*/

EulerSimulator::EulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps):
AbstractSimulator(model, data, dt, n_integration_steps), sub_dt(dt / ((double)n_integration_steps)) {

}

void EulerSimulator::computeContactForces(const Eigen::VectorXd &dq) 
{
  CONSIM_START_PROFILER("compute_contact_forces");
  // subtract joint frictions
  if (joint_friction_flag_){
    tau_ -= joint_friction_.cwiseProduct(dq);
  }

  for (auto &cp : contacts_) {
    if (!cp->active) continue;
    // If the contact point is active, compute its velocity and call the
    // contact model function on the object.
    // TODO: Is there a faster way to compute the contact point velocity than
    //       multiply the jacobian with the generalized velocity from pinocchio?
    contactLinearJacobian(cp->frame_id);
    cp->v = frame_Jc_ * dq;
    cp->optr->contactModel(*cp);
    tau_ += frame_Jc_.transpose() * cp->f;
  }
  CONSIM_STOP_PROFILER("compute_contact_forces");
}


void EulerSimulator::step(const Eigen::VectorXd &tau) 
{
  CONSIM_START_PROFILER("euler_simulator::step");
  assert(tau.size() == model_->nv);
  for (int i = 0; i < n_integration_steps_; i++)
    {
      CONSIM_START_PROFILER("euler_simulator::substep");
      // TODO: Support friction models at the joints.

      // Add the user torque;
      tau_ += tau;
      // Compute the acceloration ddq.
      CONSIM_START_PROFILER("pinocchio::aba");
      pinocchio::aba(*model_, *data_, q_, v_, tau_);
      CONSIM_STOP_PROFILER("pinocchio::aba");
      // Integrate the system forward in time.
      vMean_ = v_ + .5 * sub_dt * data_->ddq;
      pinocchio::integrate(*model_, q_, vMean_ * sub_dt, qnext_);
      q_ = qnext_;
      v_ += data_->ddq * sub_dt;
      // Compute the new data values and contact information after the integration
      // step. This way, if this method returns, the values computed in data and
      // on the contact state are consistent with the q, dq and ddq values.
      computeContactState();
      computeContactForces(v_);
      CONSIM_STOP_PROFILER("euler_simulator::substep");
    }
  CONSIM_STOP_PROFILER("euler_simulator::step");
}





/* ____________________________________________________________________________________________*/
/** 
 * ExponentialSimulator Class 
*/

ExponentialSimulator::ExponentialSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps,
                                           bool sparse, bool invertibleA) : AbstractSimulator(model, data, dt, n_integration_steps), 
                                           sparse_(sparse), invertibleA_(invertibleA), sub_dt(dt / ((double)n_integration_steps))
{
  dJvi_.resize(3);
  dvMean_.resize(model_->nv);
  Minv_.resize(model_->nv, model_->nv); Minv_.setZero();
  dv_bar.resize(model_->nv); dv_bar.setZero();
  temp01_.resize(model_->nv);
  temp02_.resize(model_->nv);
}




void ExponentialSimulator::step(const Eigen::VectorXd &tau){
  CONSIM_START_PROFILER("exponential_simulator::step");
  if(!resetflag_){
    throw std::runtime_error("resetState() must be called first !");
  }

  for (int i = 0; i < n_integration_steps_; i++){
    CONSIM_START_PROFILER("exponential_simulator::substep");
    tau_ += tau; 
    if (nactive_> 0){
      Eigen::internal::set_is_malloc_allowed(false);
      // if (sparse_){
      //   throw std::runtime_error("Sparse integration not implemented yet");
      // } // sparse 
      // else{
      //   if(invertibleA_){
      //     throw std::runtime_error("Invertible and dense integration not implemented yet");
      //   } //invertible dense 
      //   else{
      //     a_.tail(3*nactive_) = b_;
      //     CONSIM_START_PROFILER("exponential_simulator::computeIntegralXt");
      //     utilDense_.ComputeIntegralXt(A, a_, x0_, sub_dt, intxt_);
      //     CONSIM_STOP_PROFILER("exponential_simulator::computeIntegralXt");
      //   } // non-invertable dense
      // }
      computeIntegrationTerms();
      CONSIM_START_PROFILER("exponential_simulator::computeIntegralXt");
      intxt_.fill(0); 
      utilDense_.ComputeIntegralXt(A, a_, x0_, sub_dt, intxt_);
      CONSIM_STOP_PROFILER("exponential_simulator::computeIntegralXt");
      
      CONSIM_START_PROFILER("exponential_simulator::checkFrictionCone");
      checkFrictionCone();
      CONSIM_STOP_PROFILER("exponential_simulator::checkFrictionCone");
      if(cone_flag_){
        temp01_.noalias() = JcT_*fpr_; 
        temp02_ = tau - data_->nle + temp01_;
        dvMean_.noalias() = Minv_*temp02_; 
        vMean_ = v_ + sub_dt* dvMean_; 
      } // force violates friction cone 
      else{
        temp03_.noalias() = D*intxt_; 
        temp01_.noalias() = MinvJcT_ * temp03_;
        dvMean_.noalias() = dv_bar + temp01_/sub_dt ; 
        CONSIM_START_PROFILER("exponential_simulator::ComputeDoubleIntegralXt");
        int2xt_.fill(0);
        utilDense_.ComputeDoubleIntegralXt(A, a_, x0_, sub_dt, int2xt_); 
        CONSIM_STOP_PROFILER("exponential_simulator::ComputeDoubleIntegralXt");
        temp03_.noalias() = D*int2xt_; 
        temp01_.noalias() = MinvJcT_ * temp03_;
        vMean_ = v_ + .5 * sub_dt * dv_bar + temp01_/sub_dt; 
      } // force within friction cone 
    } // active contacts > 0 
    else{
      pinocchio::aba(*model_, *data_, q_, v_, tau_);
      dvMean_ = data_->ddq; 
      vMean_ = v_ + dv_ * .5 * sub_dt;
    } // no active contacts 

    CONSIM_START_PROFILER("exponential_simulator::subIntegration");      
    pinocchio::integrate(*model_, q_, vMean_ * sub_dt, qnext_);
    q_ = qnext_;
    v_ += sub_dt*dvMean_;
    dv_ = dvMean_; 
    CONSIM_STOP_PROFILER("exponential_simulator::subIntegration");
    /** 
     * pinocchio::computeAllTerms already calls first order FK 
     * https://github.com/stack-of-tasks/pinocchio/blob/3f4d9e8504ff4e05dbae0ede0bd808d025e4a6d8/src/algorithm/compute-all-terms.hpp#L20
     * we need second order FK for pinocchio::getFrameAcceleration
     * seems a bit inefficient to call forward kinematics twice   
    */
    
    CONSIM_START_PROFILER("exponential_simulator::computeContactState");
    computeContactState();  // this sets tau_ to zero 
    CONSIM_STOP_PROFILER("exponential_simulator::computeContactState");
    CONSIM_START_PROFILER("exponential_simulator::computeContactForces");
    computeContactForces(v_);
    CONSIM_STOP_PROFILER("exponential_simulator::computeContactForces");
    CONSIM_STOP_PROFILER("exponential_simulator::substep");
    Eigen::internal::set_is_malloc_allowed(true);
  }  // sub_dt loop
  CONSIM_STOP_PROFILER("exponential_simulator::step");
} // ExponentialSimulator::step


void ExponentialSimulator::computeIntegrationTerms(){
  CONSIM_START_PROFILER("exponential_simulator::computeMinverse");
  Minv_ = pinocchio::computeMinverse(*model_, *data_, q_);
  CONSIM_STOP_PROFILER("exponential_simulator::computeMinverse");
  // CONSIM_START_PROFILER("exponential_simulator::MinvEigen");
  // Minv_ = data_->M.inverse();
  // CONSIM_STOP_PROFILER("exponential_simulator::MinvEigen");
  JMinv_.noalias() = Jc_ * Minv_;
  MinvJcT_.noalias() = JMinv_.transpose(); 
  JcT_.noalias() = Jc_.transpose(); 
  Upsilon_.noalias() =  Jc_*MinvJcT_;
  temp01_.noalias() = JcT_ * kp0_;
  temp02_ = tau_ - data_->nle + temp01_;
  dv_bar.noalias() = Minv_ * temp02_; 
  tempStepMat_.noalias() =  Upsilon_ * K;
  A.block(3*nactive_, 0, 3*nactive_, 3*nactive_).noalias() = -tempStepMat_;  
  tempStepMat_.noalias() = Upsilon_ * B; 
  A.block(3*nactive_, 3*nactive_, 3*nactive_, 3*nactive_).noalias() = -tempStepMat_; 
  temp01_ = tau_ - data_->nle; 
  temp04_.noalias() = JMinv_*temp01_;  
  temp03_.noalias() =  Upsilon_*kp0_;
  b_.noalias() = temp04_ + dJv_ + temp03_; 
  a_.tail(3*nactive_) = b_;
  x0_.head(3*nactive_) = p_; 
  x0_.tail(3*nactive_) = dp_; 
}



  void ExponentialSimulator::computeContactForces(const Eigen::VectorXd &dq)
{
  if (f_.size()!=3*nactive_){
    CONSIM_START_PROFILER("exponential_simulator::resizeVectorsAndMatrices");
    resizeVectorsAndMatrices();
    CONSIM_STOP_PROFILER("exponential_simulator::resizeVectorsAndMatrices");
  } // only reallocate memory if number of active contacts changes
  // tau_ was already set to zero in checkContactStates()
  if (joint_friction_flag_)
  {
    tau_ -= joint_friction_.cwiseProduct(dq);
  }

  pinocchio::forwardKinematics(*model_, *data_, q_, v_, dv_);  // frame accelerations  
  i_active_ = 0; 
  for(unsigned int i=0; i<nc_; i++){
    if (!contacts_[i]->active) continue;
    // compute force using the model  
    contacts_[i]->optr->contactModel(*contacts_[i]); 
    f_.segment(3*i_active_,3) = contacts_[i]->f; 
    p0_.segment(3*i_active_,3)=contacts_[i]->x_start; 
    p_.segment(3*i_active_,3)=contacts_[i]->x; 
    dp_.segment(3*i_active_,3)=contacts_[i]->v; 
    // compute jacobian for active contact and store in frame_Jc_
    contactLinearJacobian(contacts_[i]->frame_id); 
    Jc_.block(3*i_active_,0,3,model_->nv) = frame_Jc_;
    computeFrameKinematics(i); 
    dJv_.segment(3*i_active_,3) = dJvi_; // cross component of acceleration
    // fill Kp0_ 
    kp0_(3*i_active_) = contacts_[i]->optr->getTangentialStiffness() * p0_(3*i_active_);
    kp0_(1+3*i_active_) = contacts_[i]->optr->getTangentialStiffness() * p0_(1+3*i_active_);
    kp0_(2+3*i_active_) = contacts_[i]->optr->getNormalStiffness() * p0_(2+3*i_active_);
    i_active_ += 1;  
  }
} // ExponentialSimulator::computeContactForces



void ExponentialSimulator::computeFrameKinematics(unsigned int contact_id)
{
  CONSIM_START_PROFILER("exponential_simulator::computeFrameAcceleration");
  dJvi_.setZero();
  vilocal_ = pinocchio::getFrameVelocity(*model_, *data_, contacts_[contact_id]->frame_id); 
  dJvilocal_ = pinocchio::getFrameAcceleration(*model_, *data_, contacts_[contact_id]->frame_id); 
  dJvilocal_.linear() += vilocal_.angular().cross(vilocal_.linear());
  frameSE3_.rotation() = data_->oMf[contacts_[contact_id]->frame_id].rotation();
  dJvi_ = frameSE3_.act(dJvilocal_).linear();
  contacts_[contact_id]->v.noalias() = frameSE3_.act(vilocal_).linear();
  CONSIM_STOP_PROFILER("exponential_simulator::computeFrameAcceleration");
} //computeFrameAcceleration



void ExponentialSimulator::checkFrictionCone(){
  temp03_.noalias() = D*intxt_;
  f_avg= kp0_ + temp03_/sub_dt; 
  i_active_ = 0; 
  cone_flag_ = true; 
  for(unsigned int i=0; i<nc_; i++){
    if (!contacts_[i]->active) continue;
    if (!contacts_[i]->unilateral) {
      fpr_.segment(3*i_active_,3) = f_avg.segment(3*i_active_,3); 
      i_active_ += 1; 
      continue;
    }

    ftan_ = sqrt(sqr(f_avg(3*i_active_)) + sqr(f_avg(1+3*i_active_))); 
    if (ftan_<(f_avg(2+3*i_active_) * contacts_[i]->optr->getFrictionCoefficient())) // fi_tan < mu*fi_z 
    {
      fpr_.segment(3*i_active_,3) = f_avg.segment(3*i_active_,3); 
    } // no violation 
    else{
      // if fi_z is pulling in world frame => fi_z < 0, this will also be activated  
      if(f_avg(2+3*i_active_)<0){
        fpr_(3*i_active_) =  0.; 
        fpr_(1+3*i_active_) = 0.;
        fpr_(2+3*i_active_) = 0.;
      } // pulling force case 
      else{
        cone_direction_ = atan2(f_avg(1+3*i_active_), f_avg(3*i_active_)); 
        fpr_(3*i_active_) = cos(cone_direction_)*f_avg(2+3*i_active_) * contacts_[i]->optr->getFrictionCoefficient();
        fpr_(1+3*i_active_) = sin(cone_direction_)*f_avg(2+3*i_active_) * contacts_[i]->optr->getFrictionCoefficient();
        fpr_(2+3*i_active_) = f_avg(2+3*i_active_); 
      } 
      cone_flag_ = true; 
    } // project onto cone boundaries 
    i_active_ += 1; 
  }
} // ExponentialSimulator::checkFrictionCone



void ExponentialSimulator::resizeVectorsAndMatrices()
{
  // Operations below need optimization, this is a first attempt
  // resize matrices and fillout contact information
  // TODO: change to use templated header dynamic_algebra.hpp
  Eigen::internal::set_is_malloc_allowed(true);
  if (nactive_>0){
    f_.resize(3 * nactive_); f_.setZero();
    p0_.resize(3 * nactive_); p0_.setZero();
    p_.resize(3 * nactive_); p_.setZero();
    dp_.resize(3 * nactive_); dp_.setZero();
    a_.resize(6 * nactive_); a_.setZero();
    b_.resize(3 * nactive_); b_.setZero();
    x0_.resize(6 * nactive_); x0_.setZero();
    xt_.resize(6 * nactive_); xt_.setZero();
    intxt_.resize(6 * nactive_); intxt_.setZero();
    int2xt_.resize(6 * nactive_); int2xt_.setZero();
    kp0_.resize(3 * nactive_); kp0_.setZero();
    K.resize(3 * nactive_, 3 * nactive_); K.setZero();
    B.resize(3 * nactive_, 3 * nactive_); B.setZero();
    D.resize(3 * nactive_, 6 * nactive_); D.setZero();
    A.resize(6 * nactive_, 6 * nactive_); A.setZero();
    A.block(0, 3*nactive_, 3*nactive_, 3*nactive_) = Eigen::MatrixXd::Identity(3*nactive_, 3*nactive_); 
    Jc_.resize(3 * nactive_, model_->nv); Jc_.setZero();
    JcT_.resize(model_->nv, 3 * nactive_); JcT_.setZero();
    Upsilon_.resize(3 * nactive_, 3 * nactive_); Upsilon_.setZero();
    JMinv_.resize(3 * nactive_, model_->nv); JMinv_.setZero();
    MinvJcT_.resize(model_->nv, 3*nactive_); MinvJcT_.setZero();
    dJv_.resize(3 * nactive_); dJv_.setZero();
    utilDense_.resize(6 * nactive_);
    f_avg.resize(3 * nactive_); f_avg.setZero();
    fpr_.resize(3 * nactive_); fpr_.setZero();
    tempStepMat_.resize(3 * nactive_, 3 * nactive_); tempStepMat_.setZero();
    temp03_.resize(3*nactive_); temp03_.setZero();
    temp04_.resize(3*nactive_); temp04_.setZero();
    // fillout K & B only needed whenever number of active contacts changes 
    i_active_ = 0; 
    for(unsigned int i=0; i<nc_; i++){
      if (!contacts_[i]->active) continue;
      K(3*i_active_, 3*i_active_) = contacts_[i]->optr->getTangentialStiffness();
      K(3*i_active_ + 1, 3*i_active_ + 1) = contacts_[i]->optr->getTangentialStiffness();
      K(3*i_active_ + 2, 3 * i_active_ + 2) = contacts_[i]->optr->getNormalStiffness();
      B(3*i_active_, 3*i_active_) = contacts_[i]->optr->getTangentialDamping();
      B(3*i_active_ + 1, 3*i_active_ + 1) = contacts_[i]->optr->getTangentialDamping();
      B(3*i_active_ + 2, 3*i_active_ + 2) = contacts_[i]->optr->getNormalDamping();
      i_active_ += 1; 
    }
    // fillout D 
    D.block(0,0, 3*nactive_, 3*nactive_).noalias() = -K;
    D.block(0,3*nactive_, 3*nactive_, 3*nactive_).noalias() = -B; 
  } // nactive_ > 0
  Eigen::internal::set_is_malloc_allowed(false);
} // ExponentialSimulator::resizeVectorsAndMatrices


/*____________________________________________________________________________________________*/



}  // namespace consim 
