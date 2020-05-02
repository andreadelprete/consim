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
  vMean_.resize(model.nv); vMean_.setZero();
  tau_.resize(model.nv); tau_.setZero();
  qnext_.resize(model.nq); qnext_.setZero();
} 

const ContactPoint &AbstractSimulator::addContactPoint(std::string name, int frame_id, bool unilateral)
{
  ContactPoint *cptr = new ContactPoint(*model_, name, frame_id, model_->nv, unilateral);
	contacts_.push_back(cptr);
  nc_ += 1; // increase contact points count  
  resetflag_ = false; // enforce resetState() after adding a contact point
  return getContact(contacts_.size() - 1);
}

const ContactPoint &AbstractSimulator::getContact(int index)
{
  return *contacts_[index];
}

void AbstractSimulator::addObject(ContactObject& obj) {
  objects_.push_back(&obj);
}

void AbstractSimulator::resetState(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, bool reset_contact_state)
{
  q_ = q;
  v_ = dq;
  vMean_ = dq;
  tau_.fill(0);
  if (reset_contact_state) {
    for (auto &cptr : contacts_) {
      cptr->active = false;
      cptr->f.fill(0);
    }
  }
  computeContactForces();
  resetflag_ = true;
}

void AbstractSimulator::detectContacts()
{
  nactive_ = 0; 
  // Loop over all the contact points, over all the objects.
  for (auto &cp : contacts_) {
    cp->updatePosition(*data_);
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
        if (optr->checkCollision(*cp))
        {
          cp->active = true;
          nactive_ += 1; 
          cp->optr = optr;
          if(!cp->unilateral){
            cout<<"Bilateral contact with object "<<optr->getName()<<" at point "<<cp->x.transpose()<<endl;
          }
          break;
        }
      }
    }
  }
}

void AbstractSimulator::setJointFriction(const Eigen::VectorXd& joint_friction)
{
  joint_friction_flag_= true;
  joint_friction_ = joint_friction;
}


/* ____________________________________________________________________________________________*/
/** 
 * EulerSimulator Class 
*/

EulerSimulator::EulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps):
AbstractSimulator(model, data, dt, n_integration_steps), sub_dt(dt / ((double)n_integration_steps)) {}


void EulerSimulator::computeContactForces() 
{
  data_->M.fill(0);
  CONSIM_START_PROFILER("pinocchio::computeAllTerms");
  pinocchio::computeAllTerms(*model_, *data_, q_, v_);
  pinocchio::updateFramePlacements(*model_, *data_);
  detectContacts();
  CONSIM_START_PROFILER("compute_contact_forces");
  for (auto &cp : contacts_) {
    if (!cp->active) continue;
    cp->firstOrderContactKinematics(*data_); // must be called before penetration, has velocity in it 
    cp->optr->computePenetration(*cp); 
    cp->optr->contact_model_->computeForce(*cp);
    tau_ += cp->world_J_.transpose() * cp->f; 
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
      // \brief add input control 
      tau_ += tau;
      // \brief joint damping 
      if (joint_friction_flag_){
        tau_ -= joint_friction_.cwiseProduct(v_);
      }
      // Compute the acceloration ddq.
      CONSIM_START_PROFILER("pinocchio::aba");
      pinocchio::aba(*model_, *data_, q_, v_, tau_);
      CONSIM_STOP_PROFILER("pinocchio::aba");
      vMean_ = v_ + .5 * sub_dt * data_->ddq;
      pinocchio::integrate(*model_, q_, vMean_ * sub_dt, qnext_);
      q_ = qnext_;
      v_ += data_->ddq * sub_dt;
      
      tau_.fill(0);
      // \brief adds contact forces to tau_
      computeContactForces(); 
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
  dvMean_.resize(model_->nv);
  Minv_.resize(model_->nv, model_->nv); Minv_.setZero();
  dv_bar.resize(model_->nv); dv_bar.setZero();
  temp01_.resize(model_->nv); temp01_.setZero();
  temp02_.resize(model_->nv); temp02_.setZero();
}


void ExponentialSimulator::step(const Eigen::VectorXd &tau){
  CONSIM_START_PROFILER("exponential_simulator::step");
  if(!resetflag_){
    throw std::runtime_error("resetState() must be called first !");
  }

  for (int i = 0; i < n_integration_steps_; i++){
    CONSIM_START_PROFILER("exponential_simulator::substep");
    // \brief add input control 
    tau_.fill(0);
    tau_ += tau;
    // \brief joint damping 
    if (joint_friction_flag_){
      tau_ -= joint_friction_.cwiseProduct(v_);
    } 

  
    if (nactive_> 0){
      Eigen::internal::set_is_malloc_allowed(false);
      computeIntegrationTerms();
      CONSIM_START_PROFILER("exponential_simulator::computeIntegralXt");
      utilDense_.ComputeIntegralXt(A, a_, x0_, sub_dt, intxt_);
      CONSIM_STOP_PROFILER("exponential_simulator::computeIntegralXt");
      CONSIM_START_PROFILER("exponential_simulator::checkFrictionCone");
      checkFrictionCone();
      CONSIM_STOP_PROFILER("exponential_simulator::checkFrictionCone");
      if(cone_flag_){
        computeSlipping();
        temp01_.noalias() = JcT_*fpr_; 
        temp02_ = tau_ - data_->nle + temp01_;
        dvMean_.noalias() = Minv_*temp02_; 
        vMean_ = v_ + sub_dt* dvMean_; 
      } // force violates friction cone 
      else{
        temp03_.noalias() = D*intxt_; 
        temp01_.noalias() = MinvJcT_ * temp03_;
        dvMean_.noalias() = dv_bar + temp01_/sub_dt ; 
        CONSIM_START_PROFILER("exponential_simulator::ComputeDoubleIntegralXt");
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
    // 
    computeContactForces();
    CONSIM_STOP_PROFILER("exponential_simulator::substep");
    Eigen::internal::set_is_malloc_allowed(true);
  }  // sub_dt loop
  CONSIM_STOP_PROFILER("exponential_simulator::step");
} // ExponentialSimulator::step


void ExponentialSimulator::computeIntegrationTerms(){
  i_active_ = 0; 
  for(unsigned int i=0; i<nc_; i++){
    if (!contacts_[i]->active) continue;
    Jc_.block(3*i_active_,0,3,model_->nv) = contacts_[i]->world_J_;
    dJv_.segment(3*i_active_,3) = contacts_[i]->dJv_; 
    p0_.segment(3*i_active_,3)=contacts_[i]->x_start; 
    p_.segment(3*i_active_,3)=contacts_[i]->x; 
    dp_.segment(3*i_active_,3)=contacts_[i]->v;  
    kp0_.segment(3*i_active_,3).noalias() = K.block(3*i_active_,3*i_active_,3,3)*p0_.segment(3*i_active_,3);
    i_active_ += 1;  
  }
  CONSIM_START_PROFILER("exponential_simulator::computeMinverse");
  Minv_ = pinocchio::computeMinverse(*model_, *data_, q_);
  CONSIM_STOP_PROFILER("exponential_simulator::computeMinverse");
  JcT_.noalias() = Jc_.transpose(); 
  JMinv_.noalias() = Jc_ * Minv_;
  MinvJcT_.noalias() = Minv_*JcT_; 
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



  void ExponentialSimulator::computeContactForces()
{
  data_->M.fill(0);
  
  CONSIM_START_PROFILER("pinocchio::computeAllTerms");
  pinocchio::computeAllTerms(*model_, *data_, q_, v_);
  CONSIM_STOP_PROFILER("pinocchio::computeAllTerms");
  pinocchio::updateFramePlacements(*model_, *data_);
  detectContacts();
  
  
  
  
  if (nactive_>0){
    if (f_.size()!=3*nactive_){
    CONSIM_START_PROFILER("exponential_simulator::resizeVectorsAndMatrices");
    resizeVectorsAndMatrices();
    CONSIM_STOP_PROFILER("exponential_simulator::resizeVectorsAndMatrices");
    }
    // pinocchio::forwardKinematics(*model_, *data_, q_, v_, dv_);   // needed for dJv_ 
    i_active_ = 0; 
    for(unsigned int i=0; i<nc_; i++){
      if (!contacts_[i]->active) continue;
      contacts_[i]->firstOrderContactKinematics(*data_);
      contacts_[i]->optr->computePenetration(*contacts_[i]);
      contacts_[i]->secondOrderContactKinematics(*data_);
      contacts_[i]->optr->contact_model_->computeForce(*contacts_[i]);
      f_.segment(3*i_active_,3) = contacts_[i]->f; 
      i_active_ += 1;  
    }
  }
} // ExponentialSimulator::computeContactForces



void ExponentialSimulator::checkFrictionCone(){
  temp03_.noalias() = D*intxt_;
  f_avg= kp0_ + temp03_/sub_dt; 
  i_active_ = 0; 
  cone_flag_ = false; 
  for(unsigned int i=0; i<nc_; i++){
    if (!contacts_[i]->active) continue;
    if (!contacts_[i]->unilateral) {
      fpr_.segment(3*i_active_,3) = f_avg.segment(3*i_active_,3); 
      i_active_ += 1; 
      continue;
    }
    
    fnor_ = contacts_[i]->contactNormal_.dot(f_avg);
    if (fnor_<0.){
      /*!< check for pulling force at contact i */  
      fpr_.segment(3*i_active_,3).fill(0); 
      cone_flag_ = true; 
    } else{
      /*!< check for friction bounds  */  
      normalFi_ = fnor_* contacts_[i]->contactNormal_; 
      tangentFi_ = fpr_ - normalFi_; 
      ftan_ = sqrt(tangentFi_.dot(tangentFi_));
      if(ftan_ > (contacts_[i]->optr->contact_model_->friction_coeff_ * fnor_)){
        /*!< cone violated */  
        fpr_.segment(3*i_active_,3) = f_avg.segment(3*i_active_,3);   
        cone_flag_ = true;
      } else {
        /*!< if not violated still fill out in case another contact violates the cone  */  
        fpr_.segment(3*i_active_,3) = f_avg.segment(3*i_active_,3); 
      }

    }

    i_active_ += 1; 
  }
} // ExponentialSimulator::checkFrictionCone




void ExponentialSimulator::computeSlipping(){
  /**
   * Populate the constraints then solve the qp 
   * update x_start 
   * compute the projected contact forces for integration 
   **/  
}



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
    invK.resize(3 * nactive_, 3 * nactive_); K.setZero();
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
      K(3*i_active_, 3*i_active_) = contacts_[i]->optr->contact_model_->stiffness_(0);
      K(3*i_active_+1, 3*i_active_+1) = contacts_[i]->optr->contact_model_->stiffness_(1);
      K(3*i_active_+2, 3*i_active_+2) = contacts_[i]->optr->contact_model_->stiffness_(2);
      B(3*i_active_, 3*i_active_) = contacts_[i]->optr->contact_model_->damping_(0);
      B(3*i_active_+1, 3*i_active_+1) = contacts_[i]->optr->contact_model_->damping_(1);
      B(3*i_active_+2, 3*i_active_+2) = contacts_[i]->optr->contact_model_->damping_(2);
      // Kinv is required for slippling 
      invK(3*i_active_, 3*i_active_) = 1/K(3*i_active_, 3*i_active_);
      invK(3*i_active_ + 1, 3*i_active_ + 1) = 1/K(3*i_active_ + 1, 3*i_active_ + 1);
      invK(3*i_active_ + 2, 3 * i_active_ + 2) = 1/K(3*i_active_ + 2, 3 * i_active_ + 2);
      i_active_ += 1; 
    }
    // fillout D 
    D.block(0,0, 3*nactive_, 3*nactive_).noalias() = -K;
    D.block(0,3*nactive_, 3*nactive_, 3*nactive_).noalias() = -B; 
    // qp resizing 
    // constraints should account for both directions of friction 
    // and positive normal force, this implies 5 constraints per active contact
    // will be arranged as follows [normal, +ve_basisA, -ve_BasisA, +ve_BasisB, -ve_BasisB]
    // qp.reset(3*nactive_, 0., 5 * nactive_); 
    // Q_cone.resize(3 * nactive_, 3 * nactive_); Q_cone.setZero(); 
    // Q_cone.noalias() = 2 * Eigen::MatrixXd::Identity(3*nactive_, 3*nactive_);
    // g0_cone.resize(3*nactive_); g0_cone.setZero();
    // Cineq_cone.resize(5 * nactive_,5 * nactive_); Cineq_cone.setZero();
    // cineq_cone.resize(5 * nactive_);  cineq_cone.setZero();
  } // nactive_ > 0
  


  Eigen::internal::set_is_malloc_allowed(false);
} // ExponentialSimulator::resizeVectorsAndMatrices


/*____________________________________________________________________________________________*/



}  // namespace consim 
