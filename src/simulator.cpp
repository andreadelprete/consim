#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
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
#include "consim/simulator.hpp"
// #include "consim/utils/stop-watch.hpp"

#include <iostream>

// TODO: sqr already defined in contact.cpp 
#define sqr(x) (x * x)

namespace consim {

/** 
 * AbstractSimulator Class 
*/

AbstractSimulator::AbstractSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD): 
model_(&model), data_(&data), dt_(dt), n_integration_steps_(n_integration_steps), sub_dt(dt / ((double)n_integration_steps)), whichFD_(whichFD) {
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


const ContactPoint &AbstractSimulator::addContactPoint(std::string name, int frame_id, bool unilateral)
{
  ContactPoint *cptr = new ContactPoint(*model_, name, frame_id, model_->nv, unilateral);
	contacts_.push_back(cptr);
  nc_ += 1; /*!< total number of defined contact points */ 
  resetflag_ = false; /*!< cannot call Simulator::step() if resetflag is false */ 
  return getContact(name);
}


const ContactPoint &AbstractSimulator::getContact(std::string name)
{
  for (auto &cptr : contacts_) {
    if (cptr->name_==name){
      return *cptr; 
    } 
  }
  throw std::runtime_error("Contact name not recongnized ");
}


void AbstractSimulator::resetContactAnchorPoint(std::string name, Eigen::Vector3d &p0, bool updateContactForces){
  for (auto &cptr : contacts_) {
    if (cptr->name_==name){
      if (cptr->active){
        cptr->resetAnchorPoint(p0); 
      }
      break; 
    }
  }
  if (updateContactForces){
    computeContactForces();
  }
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
  for (unsigned int i=0; i<nc_; ++i){
    contacts_[i]->predictedX_ = data_->oMf[contacts_[i]->frame_id].translation(); 
  }
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

EulerSimulator::EulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD):
AbstractSimulator(model, data, dt, n_integration_steps, whichFD) {}


void EulerSimulator::computeContactForces() 
{
  CONSIM_START_PROFILER("pinocchio::computeAllTerms");
  pinocchio::forwardKinematics(*model_, *data_, q_, v_, fkDv_);
  pinocchio::computeJointJacobians(*model_, *data_);
  pinocchio::updateFramePlacements(*model_, *data_);
  pinocchio::nonLinearEffects(*model_, *data_, q_, v_);
  /*!< loops over all contacts and objects to detect contacts and update contact positions*/ 
  detectContacts();
  CONSIM_START_PROFILER("compute_contact_forces");
  for (auto &cp : contacts_) {
    if (!cp->active) continue;
    cp->firstOrderContactKinematics(*data_); /*!<  must be called before computePenetration() it updates cp.v and jacobian*/   
    cp->optr->computePenetration(*cp); 
    cp->optr->contact_model_->computeForce(*cp);
    tau_ += cp->world_J_.transpose() * cp->f; 
  }
  CONSIM_STOP_PROFILER("compute_contact_forces");
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
      CONSIM_START_PROFILER("euler_simulator::substep");
      // \brief add input control 
      tau_ += tau;
      // \brief joint damping 
      if (joint_friction_flag_){
        tau_ -= joint_friction_.cwiseProduct(v_);
      }
      /**
       * Solving the Forward Dynamics  
       *  1: pinocchio::computeMinverse()
       *  2: pinocchio::aba()
       *  3: cholesky decompostion 
       **/  
      switch (whichFD_)
      {
      case 1:
        mDv_ = tau_ - data_->nle; 
        inverseM_ = pinocchio::computeMinverse(*model_, *data_, q_);
        dv_ = inverseM_*mDv_; 
        break;
      
      case 2:
        pinocchio::aba(*model_, *data_, q_, v_, tau_);
        dv_ = data_-> ddq; 
        break;
      
      case 3:
        pinocchio::crba(*model_, *data_, q_);
        mDv_ = tau_ - data_->nle; 
        lltM_.compute(data_->M);
        dv_ = lltM_.solve(mDv_);
        break;
      
      default:
        throw std::runtime_error("Forward Dynamics Method not recognized");
      }
      
      /*!< integrate twice */  
      vMean_ = v_ + .5 * sub_dt*dv_;
      pinocchio::integrate(*model_, q_, vMean_ * sub_dt, qnext_);
      q_ = qnext_;
      v_ += dv_ * sub_dt;
      
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

ExponentialSimulator::ExponentialSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, 
                                            int n_integration_steps, int whichFD,
                                            int slipping_method, bool compute_predicted_forces) : 
                                            AbstractSimulator(model, data, dt, n_integration_steps, whichFD), 
                                            slipping_method_(slipping_method),
                                            compute_predicted_forces_(compute_predicted_forces)
{
  dvMean_.resize(model_->nv);
  dvMean2_.resize(model_->nv);
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
      CONSIM_START_PROFILER("exponential_simulator::computeIntegrationTerms");
      computeIntegrationTerms();
      CONSIM_STOP_PROFILER("exponential_simulator::computeIntegrationTerms");
      
      CONSIM_START_PROFILER("exponential_simulator::computeIntegralsXt");
      utilDense_.ComputeIntegralXt(A, a_, x0_, sub_dt, intxt_);
      utilDense_.ComputeDoubleIntegralXt(A, a_, x0_, sub_dt, int2xt_); 
      CONSIM_STOP_PROFILER("exponential_simulator::computeIntegralsXt");

      CONSIM_START_PROFILER("exponential_simulator::checkFrictionCone");
      checkFrictionCone();
      CONSIM_STOP_PROFILER("exponential_simulator::checkFrictionCone");
      
      if (!cone_flag_ || slipping_method_==1){
        /*!< f projection is computed then anchor point is updated */ 
        CONSIM_START_PROFILER("exponential_simulator::subIntegration");
        temp01_.noalias() = JcT_*fpr_; 
        temp02_ = tau_ - data_->nle + temp01_;
        dvMean_.noalias() = Minv_*temp02_; 

        temp01_.noalias() = JcT_*fpr2_; 
        temp02_ = tau_ - data_->nle + temp01_;
        dvMean2_.noalias() = Minv_*temp02_; 
        vMean2_.noalias() = v_ + .5 * sub_dt * dvMean2_; 
        pinocchio::integrate(*model_, q_, vMean2_ * sub_dt, qnext_);
        CONSIM_STOP_PROFILER("exponential_simulator::subIntegration"); 
        /* PSEUDO-CODE
        dv_mean = dv_bar + JMinv.T @ f_pr
        v_mean = v + 0.5*dt*(dv_bar + JMinv.T @ f_pr2)
        */
      }
      else{
        /*!< anchor point is optimized, then one f projection is computed and itegrated */ 
        // computeSlipping(); // slipping qp not implemented 
        temp01_.noalias() = JcT_*fpr_; 
        temp02_ = tau_ - data_->nle + temp01_;
        dvMean_.noalias() = Minv_*temp02_; 
        vMean_ = v_ + .5 * sub_dt* dvMean_; 
        pinocchio::integrate(*model_, q_, vMean_ * sub_dt, qnext_);
      }
    } /*!< active contacts */
    else{
      pinocchio::nonLinearEffects(*model_, *data_, q_, v_);
      switch (whichFD_)
      {
      case 1:
        mDv_ = tau_ - data_->nle; 
        inverseM_ = pinocchio::computeMinverse(*model_, *data_, q_);
        dvMean_ = inverseM_*mDv_; 
        break;
      
      case 2:
        pinocchio::aba(*model_, *data_, q_, v_, tau_);
        dvMean_ = data_-> ddq; 
        break;
      
      case 3:
        pinocchio::crba(*model_, *data_, q_);
        mDv_ = tau_ - data_->nle; 
        lltM_.compute(data_->M);
        dvMean_ = lltM_.solve(mDv_);
        break;
      
      default:
        throw std::runtime_error("Forward Dynamics Method not recognized");
      }

      vMean_ = v_ + .5 * sub_dt*dvMean_;
      pinocchio::integrate(*model_, q_, vMean_ * sub_dt, qnext_);
    } /*!< no active contacts */

    q_ = qnext_;
    v_ += sub_dt*dvMean_;
    dv_ = dvMean_; 
    
    //
    computeContactForces();
    CONSIM_STOP_PROFILER("exponential_simulator::substep");
    Eigen::internal::set_is_malloc_allowed(true);
  }  // sub_dt loop
  CONSIM_STOP_PROFILER("exponential_simulator::step");
} // ExponentialSimulator::step


void ExponentialSimulator::computeIntegrationTerms(){
  /**
   * computes M, nle
   * fills J, dJv, p0, p, dp, Kp0, and x0 
   * computes A and b 
   **/   
  // No need to compute M because we just need M inverse
  //pinocchio::crba(*model_, *data_, q_);
  pinocchio::nonLinearEffects(*model_, *data_, q_, v_);
  i_active_ = 0; 
  for(unsigned int i=0; i<nc_; i++){
    if (!contacts_[i]->active) continue;
    Jc_.block(3*i_active_,0,3,model_->nv) = contacts_[i]->world_J_;
    dJv_.segment<3>(3*i_active_) = contacts_[i]->dJv_; 
    p0_.segment<3>(3*i_active_)  = contacts_[i]->x_anchor; 
    p_.segment<3>(3*i_active_)   = contacts_[i]->x; 
    dp_.segment<3>(3*i_active_)  = contacts_[i]->v;  
    kp0_.segment<3>(3*i_active_).noalias() = contacts_[i]->optr->contact_model_->stiffness_.cwiseProduct(p0_.segment<3>(3*i_active_));
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
  /**
   * computes the kinematics at the end of the integration step, 
   * runs contact detection 
   * resizes matrices to match the number of active contacts if needed 
   * compute the contact froces of the active contacts 
   **/  
  
  CONSIM_START_PROFILER("pinocchio::computeKinematics");
  pinocchio::forwardKinematics(*model_, *data_, q_, v_, fkDv_);
  pinocchio::computeJointJacobians(*model_, *data_);
  pinocchio::updateFramePlacements(*model_, *data_);
  CONSIM_STOP_PROFILER("pinocchio::computeKinematics");
  detectContacts(); /*!<inactive contacts get automatically filled with zero here */

  if (nactive_>0){
    if (f_.size()!=3*nactive_){
      CONSIM_START_PROFILER("exponential_simulator::resizeVectorsAndMatrices");
      resizeVectorsAndMatrices();
      CONSIM_STOP_PROFILER("exponential_simulator::resizeVectorsAndMatrices");
    }
    i_active_ = 0; 
    for(unsigned int i=0; i<nc_; i++){
      if (!contacts_[i]->active) continue;
      contacts_[i]->firstOrderContactKinematics(*data_);
      contacts_[i]->optr->computePenetration(*contacts_[i]);
      contacts_[i]->secondOrderContactKinematics(*data_, v_);
      /*!< computeForce updates the anchor point */ 
      contacts_[i]->optr->contact_model_->computeForce(*contacts_[i]);
      f_.segment<3>(3*i_active_) = contacts_[i]->f; 
      i_active_ += 1;  
    }
  }
} // ExponentialSimulator::computeContactForces



void ExponentialSimulator::computePredictedXandF(){
  /**
   * computes e^{dt*A}
   * computes \int{e^{dt*A}}
   * computes predictedXf = edtA x0 + int_edtA_ * b 
   **/  
  
  if(compute_predicted_forces_){
    util_eDtA.compute(sub_dt*A,expAdt_);   
    inteAdt_.fill(0);
    switch (nactive_)
    {
    case 1:
      util_int_eDtA_one.computeExpIntegral(A, inteAdt_, sub_dt);
      break;
    
    case 2:
      util_int_eDtA_two.computeExpIntegral(A, inteAdt_, sub_dt);
      break;
    
    case 3:
      util_int_eDtA_three.computeExpIntegral(A, inteAdt_, sub_dt);
      break;
    
    case 4:
      util_int_eDtA_four.computeExpIntegral(A, inteAdt_, sub_dt);
      break;
    
    default:
      break;
    }
    predictedXf_ = expAdt_*x0_ + inteAdt_*a_; 
    predictedForce_ = kp0_ + D*predictedXf_;
  }
  else{
    predictedXf_ = x0_;
    predictedForce_ = kp0_;
  }
  
}


void ExponentialSimulator::checkFrictionCone(){
  /**
   * computes the average force on the integration interval 
   * checks for pulling force constraints
   * checks for friction forces constraints 
   * sets a flag needed to complete the integration step 
   * predictedF should be F at end of integration step unless saturated 
   **/  
  CONSIM_START_PROFILER("exponential_simulator::computePredictedXandF");
  // also update contact position, velocity and force at the end of step 
  computePredictedXandF();
  CONSIM_STOP_PROFILER("exponential_simulator::computePredictedXandF");

  /**
      f_avg = D @ int_x / dt
      f_avg2 = D @ int2_x / (0.5*dt*dt)
  */
  temp03_.noalias() = D*intxt_;
  f_avg  = kp0_ + temp03_/sub_dt; 

  temp03_.noalias() = D*int2xt_;
  temp03_.noalias() = temp03_/(0.5*sub_dt*sub_dt);
  f_avg2 = kp0_ +  temp03_; 
  
  i_active_ = 0;
  cone_flag_ = false; 

  for(unsigned int i=0; i<nc_; i++){
    if (!contacts_[i]->active) continue;

    contacts_[i]->predictedX_ = predictedXf_.segment<3>(3*i_active_); 
    contacts_[i]->predictedV_ = predictedXf_.segment<3>(3*nactive_+3*i_active_);
    contacts_[i]->predictedF_ = predictedForce_.segment<3>(3*i_active_); 

    if (!contacts_[i]->unilateral) {
      fpr_.segment<3>(3*i_active_) = f_avg.segment<3>(3*i_active_); 
      fpr2_.segment<3>(3*i_active_) = f_avg2.segment<3>(3*i_active_); 
      contacts_[i]->predictedF_ = fpr_.segment<3>(3*i_active_);
      i_active_ += 1; 
      continue;
    }
    
    f_tmp = f_avg.segment<3>(3*i_active_);
    contacts_[i]->projectForceInCone(f_tmp);
    fpr_.segment<3>(3*i_active_) = f_tmp;

    f_tmp = f_avg2.segment<3>(3*i_active_);
    contacts_[i]->projectForceInCone(f_tmp);
    fpr2_.segment<3>(3*i_active_) = f_tmp;

    cone_flag_ = true; 

    // f_avg_i = f_avg.segment<3>(3*i_active_);
    // f_avg_i2 = f_avg2.segment<3>(3*i_active_);
    // fnor_ = contacts_[i]->contactNormal_.dot(f_avg_i);
    // fnor2_= contacts_[i]->contactNormal_.dot(f_avg_i2);
    // if (fnor_<0.){
    //   /*!< check for pulling force at contact i */  
    //   fpr_.segment<3>(3*i_active_).fill(0); 
    //   contacts_[i]->predictedF_.fill(0);
    //   cone_flag_ = true; 
    // } else{
    //   /*!< check for friction bounds on average force   */  
    //   normalFi_ = fnor_* contacts_[i]->contactNormal_; 
    //   tangentFi_ = f_avg_i - normalFi_; 
    //   ftan_ = sqrt(tangentFi_.dot(tangentFi_));
    //   double mu = contacts_[i]->optr->contact_model_->friction_coeff_;
    //   if(ftan_ > mu * fnor_){
    //     /*!< cone violated */  
    //     fpr_.segment<3>(3*i_active_) = normalFi_ + (mu*fnor_/ftan_)*tangentFi_; 
    //     contacts_[i]->predictedF_ = fpr_.segment<3>(3*i_active_);
    //     /*!< move predicted x0 if update method is 1  */
    //     if(slipping_method_==1){
    //       contacts_[i]->predictedX0_ = invK_.block<3,3>(3*i_active_,3*i_active_)*(fpr_.segment<3>(3*i_active_) + K.block<3,3>(3*i_active_,3*i_active_) * contacts_[i]->predictedX_ + B.block<3,3>(3*i_active_,3*i_active_) * contacts_[i]->predictedV_);
    //     }
    //     cone_flag_ = true;
    //     // break; 
    //   } 
    //   else {
    //     fpr_.segment<3>(3*i_active_) = f_avg_i;
    //   }
    // }

    // if (fnor2_<0.){
    //   fpr2_.segment<3>(3*i_active_).fill(0);
    //   cone_flag_ = true; 
    // } else{
    //   /*!< check for friction bounds on average of average force */ 
    //   normalFi_2 = fnor2_* contacts_[i]->contactNormal_; 
    //   tangentFi_2 = f_avg_i2 - normalFi_2; 
    //   ftan2_ = sqrt(tangentFi_2.dot(tangentFi_2));
    //   double mu = contacts_[i]->optr->contact_model_->friction_coeff_;
    //   if(ftan2_ > mu * fnor2_){
    //     fpr2_.segment<3>(3*i_active_) = normalFi_2 + (mu*fnor2_/ftan2_)*tangentFi_2; 
    //     cone_flag_ = true;
    //   } 
    //   else {
    //     fpr2_.segment<3>(3*i_active_) = f_avg_i2;
    //   }
    // }
    i_active_ += 1; 
  }
} // ExponentialSimulator::checkFrictionCone




// void ExponentialSimulator::computeSlipping(){
  
  
//   if(slipping_method_==1){
//     // throw std::runtime_error("Slipping update method not implemented yet ");
//   }
//   else if(slipping_method_==2){
//     /**
//      * Populate the constraints then solve the qp 
//      * update x_start 
//      * compute the projected contact forces for integration 
//      **/  

//     D_intExpA_integrator = D * inteAdt_ * contact_position_integrator_; 

//     // std::cout<<"A bar \n"<< D_intExpA_integrator << std::endl; 

//     Cineq_cone.setZero();
//     Cineq_cone = - cone_constraints_* D_intExpA_integrator; 
//     cineq_cone.setZero();
//     cineq_cone = cone_constraints_ * f_avg;
//     // try to ensure normal force stays the same 
//     // Ceq_cone.setZero(); 
//     // Ceq_cone = -eq_cone_constraints_ * D_intExpA_integrator;

//     // std::cout<<"A_ineq  \n"<< Cineq_cone << std::endl; 
//     // std::cout<<"b_ineq  \n"<< cineq_cone << std::endl; 
//     // std::cout<<"A_eq  \n"<< Ceq_cone << std::endl; 

//     optdP_cone.setZero();

//     status_qp = qp.solve_quadprog(Q_cone, q_cone, Ceq_cone, ceq_cone, Cineq_cone, cineq_cone, optdP_cone);

//     i_active_ = 0; 
//     for (unsigned int i = 0; i<nactive_; i++){
//       if (!contacts_[i]->active || !contacts_[i]->unilateral) continue;
//       contacts_[i]->predictedX0_ += .5 * sub_dt * optdP_cone.segment<3>(3*i_active_); 
//       contacts_[i]->predictedF_ = K.block<3,3>(3*i_active_, 3*i_active_)*(contacts_[i]->predictedX0_-contacts_[i]->predictedX_); 
//       contacts_[i]->predictedF_ -= B.block<3,3>(3*i_active_, 3*i_active_)*contacts_[i]->predictedV_; 
//       fpr_.segment<3>(3*i_active_) = contacts_[i]->predictedF_;
//       i_active_ += 1; 
//     }
//     // std::cout<<"optimizer status\n"<<status_qp<<std::endl; 
//     // if (status_qp == expected_qp){
//     //   i_active_ = 0; 
//     //   for (unsigned int i = 0; i<nactive_; i++){
//     //     if (!contacts_[i]->active || !contacts_[i]->unilateral) continue;
//     //     contacts_[i]->predictedX0_ += .5 * sub_dt * optdP_cone.segment<3>(3*i_active_); 
//     //     contacts_[i]->predictedF_ = K.block<3,3>(3*i_active_, 3*i_active_)*(contacts_[i]->predictedX0_-contacts_[i]->predictedX_); 
//     //     contacts_[i]->predictedF_ -= B.block<3,3>(3*i_active_, 3*i_active_)*contacts_[i]->predictedV_; 
//     //     fpr_.segment<3>(3*i_active_) = contacts_[i]->predictedF_;
//     //     i_active_ += 1; 
//     //   }
//     // } else{
//     //   throw std::runtime_error("solver did not converge ");
//     // }
//   } 
//   else{
//     throw std::runtime_error("Slipping update method not recongnized ");
//   }

// }



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
    predictedXf_.resize(6 * nactive_); predictedXf_.setZero();
    intxt_.resize(6 * nactive_); intxt_.setZero();
    int2xt_.resize(6 * nactive_); int2xt_.setZero();
    kp0_.resize(3 * nactive_); kp0_.setZero();
    K.resize(3 * nactive_, 3 * nactive_); K.setZero();
    invK_.resize(3 * nactive_, 3 * nactive_); invK_.setZero();
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
    f_avg2.resize(3 * nactive_); f_avg.setZero();
    fpr_.resize(3 * nactive_); fpr_.setZero();
    fpr2_.resize(3 * nactive_); fpr2_.setZero();
    tempStepMat_.resize(3 * nactive_, 3 * nactive_); tempStepMat_.setZero();
    temp03_.resize(3*nactive_); temp03_.setZero();
    temp04_.resize(3*nactive_); temp04_.setZero();


    // qp resizing 
    // constraints should account for both directions of friction 
    // and positive normal force, this implies 5 constraints per active contact
    // will be arranged as follows [+ve_basisA, -ve_BasisA, +ve_BasisB, -ve_BasisB]
    // cone_constraints_.resize(4*nactive_,3*nactive_); cone_constraints_.setZero(); 
    // eq_cone_constraints_.resize(nactive_,3*nactive_); eq_cone_constraints_.setZero(); 
    // contact_position_integrator_.resize(6*nactive_,3*nactive_); contact_position_integrator_.setZero(); 
    // // divide integrator matrix by sub_dt directly 
    // contact_position_integrator_.block(0,0, 3*nactive_, 3*nactive_) = .5 * Eigen::MatrixXd::Identity(3*nactive_, 3*nactive_);
    // contact_position_integrator_.block(3*nactive_,0, 3*nactive_, 3*nactive_) = Eigen::MatrixXd::Identity(3*nactive_, 3*nactive_)/sub_dt;
    // D_intExpA_integrator.resize(3*nactive_,3*nactive_); D_intExpA_integrator.setZero(); 

    // qp.reset(3*nactive_, 0., 4 * nactive_); 
    // Q_cone.resize(3 * nactive_, 3 * nactive_); Q_cone.setZero(); 
    // Q_cone.noalias() = 2 * Eigen::MatrixXd::Identity(3*nactive_, 3*nactive_);
    // q_cone.resize(3*nactive_); q_cone.setZero();
    // Cineq_cone.resize(4 * nactive_,3 * nactive_); Cineq_cone.setZero();
    // cineq_cone.resize(4 * nactive_);  cineq_cone.setZero();
    // // Ceq_cone.resize(nactive_,3 * nactive_); Ceq_cone.setZero();
    // // ceq_cone.resize(nactive_);  ceq_cone.setZero();
    // Ceq_cone.resize(0,3 * nactive_); Ceq_cone.setZero();
    // ceq_cone.resize(0);  ceq_cone.setZero();
    // optdP_cone.resize(3*nactive_), optdP_cone.setZero();


    //
    expAdt_.resize(6 * nactive_, 6 * nactive_); expAdt_.setZero();
    util_eDtA.resize(6 * nactive_);
    inteAdt_.resize(6 * nactive_, 6 * nactive_); inteAdt_.setZero();
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

      // fill up contact normals and tangents for constraints 
      // cone_constraints_.block<1,3>(4*i_active_, 3*i_active_) = (1/sqrt(2)) * contacts_[i]->optr->contact_model_->friction_coeff_*contacts_[i]->contactNormal_.transpose() - contacts_[i]->contactTangentA_.transpose();
      // cone_constraints_.block<1,3>(4*i_active_+1, 3*i_active_) = (1/sqrt(2)) * contacts_[i]->optr->contact_model_->friction_coeff_*contacts_[i]->contactNormal_.transpose() + contacts_[i]->contactTangentA_.transpose();
      // cone_constraints_.block<1,3>(4*i_active_+2, 3*i_active_) = (1/sqrt(2)) * contacts_[i]->optr->contact_model_->friction_coeff_*contacts_[i]->contactNormal_.transpose() - contacts_[i]->contactTangentB_.transpose();
      // cone_constraints_.block<1,3>(4*i_active_+3, 3*i_active_) = (1/sqrt(2)) * contacts_[i]->optr->contact_model_->friction_coeff_*contacts_[i]->contactNormal_.transpose() + contacts_[i]->contactTangentB_.transpose();
      // eq_cone_constraints_.block<1,3>(i_active_, 3*i_active_) = contacts_[i]->contactNormal_.transpose();

      i_active_ += 1; 
    }
    // fillout D 
    D.block(0,0, 3*nactive_, 3*nactive_).noalias() = -K;
    D.block(0,3*nactive_, 3*nactive_, 3*nactive_).noalias() = -B; 

    invK_ = K.inverse();

    // std::cout<<"resize vectors and matrices"<<std::endl;
    
  } // nactive_ > 0
  
  // std::cout<<"cone constraints \n"<<cone_constraints_<<std::endl; 
  // std::cout<<"contact velocity integrator \n"<<contact_position_integrator_<<std::endl; 


  Eigen::internal::set_is_malloc_allowed(false);
} // ExponentialSimulator::resizeVectorsAndMatrices


/*____________________________________________________________________________________________*/



}  // namespace consim 
