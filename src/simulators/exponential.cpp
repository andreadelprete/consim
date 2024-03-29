
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

#include "consim/simulators/exponential.hpp"

#include <iostream>

namespace consim {

/* ____________________________________________________________________________________________*/
/** 
 * ExponentialSimulator Class 
*/

ExponentialSimulator::ExponentialSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, 
                                            int n_integration_steps, int whichFD, EulerIntegrationType type,
                                            int slipping_method, bool compute_predicted_forces, 
                                            int exp_max_mat_mul, int lds_max_mat_mul) : 
                                            AbstractSimulator(model, data, dt, n_integration_steps, whichFD, type), 
                                            slipping_method_(slipping_method),
                                            compute_predicted_forces_(compute_predicted_forces), 
                                            expMaxMatMul_(exp_max_mat_mul),
                                            ldsMaxMatMul_(lds_max_mat_mul),
                                            assumeSlippageContinues_(true),
                                            use_diagonal_matrix_exp_(false),
                                            update_A_frequency_(1),
                                            update_A_counter_(0)
{
  dvMean_.resize(model_->nv);
  dvMean2_.resize(model_->nv);
  vMean2_.resize(model_->nv);
  dv_bar.resize(model_->nv); dv_bar.setZero();
  temp01_.resize(model_->nv); temp01_.setZero();
  temp02_.resize(model_->nv); temp02_.setZero();

  util_eDtA.setMaxMultiplications(expMaxMatMul_); 
  utilDense_.setMaxMultiplications(ldsMaxMatMul_); 
}


void ExponentialSimulator::step(const Eigen::VectorXd &tau){
  CONSIM_START_PROFILER("exponential_simulator::step");
  if(!resetflag_){
    throw std::runtime_error("resetState() must be called first !");
  }

  // \brief add input control 
  tau_ = tau;
  // \brief joint damping 
  if (joint_friction_flag_){
    tau_ -= joint_friction_.cwiseProduct(v_);
  }

  for (int i = 0; i < n_integration_steps_; i++){
    CONSIM_START_PROFILER("exponential_simulator::substep"); 
    if (nactive_> 0){
      Eigen::internal::set_is_malloc_allowed(false);
      
      CONSIM_START_PROFILER("exponential_simulator::computeExpLDS");
      bool update_A = false;
      update_A_counter_--;
      if(update_A_counter_ <= 0){
        update_A = true;
        update_A_counter_ = update_A_frequency_;
      }
      computeExpLDS(update_A);
      CONSIM_STOP_PROFILER("exponential_simulator::computeExpLDS");

      CONSIM_START_PROFILER("exponential_simulator::computeIntegralsXt");
      utilDense_.ComputeIntegrals(A, a_, x0_, sub_dt, intxt_, int2xt_);
      CONSIM_STOP_PROFILER("exponential_simulator::computeIntegralsXt");

      CONSIM_START_PROFILER("exponential_simulator::checkFrictionCone");
      checkFrictionCone();
      CONSIM_STOP_PROFILER("exponential_simulator::checkFrictionCone");

      CONSIM_START_PROFILER("exponential_simulator::integrateState");
      /*!< f projection is computed then anchor point is updated */ 
      dvMean_ = dv_bar + MinvJcT_*fpr_; 
      dvMean2_.noalias() =  dv_bar + MinvJcT_*fpr2_;  
      vMean_ =  v_ +  .5 * sub_dt * dvMean2_; 
    } /*!< active contacts */
    else{
      CONSIM_START_PROFILER("exponential_simulator::noContactsForwardDynamics");
      forwardDynamics(tau_, dvMean_); 
      CONSIM_STOP_PROFILER("exponential_simulator::noContactsForwardDynamics");
      CONSIM_START_PROFILER("exponential_simulator::integrateState");
      vMean_ = v_ + .5 * sub_dt*dvMean_;
    } /*!< no active contacts */
    
    v_ += sub_dt*dvMean_;
    if(integration_type_==SEMI_IMPLICIT && nactive_==0){
      pinocchio::integrate(*model_, q_, v_ * sub_dt, qnext_);
    }
    else{
      pinocchio::integrate(*model_, q_, vMean_ * sub_dt, qnext_);
    }
    q_ = qnext_;
    dv_ = dvMean_; 
    CONSIM_STOP_PROFILER("exponential_simulator::integrateState");
    
    CONSIM_START_PROFILER("exponential_simulator::computeContactForces");
    computeContactForces();
    Eigen::internal::set_is_malloc_allowed(true);
    elapsedTime_ += sub_dt; 
    CONSIM_STOP_PROFILER("exponential_simulator::computeContactForces");

    CONSIM_STOP_PROFILER("exponential_simulator::substep");
  }  // sub_dt loop
  CONSIM_STOP_PROFILER("exponential_simulator::step");
} // ExponentialSimulator::step


void ExponentialSimulator::computeExpLDS(bool update_A){
  /**
   * computes M, nle
   * fills J, dJv, p0, p, dp, Kp0, and x0 
   * computes A and b 
   **/   
  // Do we need to compute M before computing M inverse?
  B_copy = B;
  i_active_ = 0; 
  for(auto &cp : contacts_){
    if (!cp->active) continue;
    Jc_.block(3*i_active_,0,3,model_->nv) = cp->world_J_;
    dJv_.segment<3>(3*i_active_) = cp->dJv_; 
    p0_.segment<3>(3*i_active_)  = cp->x_anchor; 
    dp0_.segment<3>(3*i_active_)  = cp->v_anchor; 
    p_.segment<3>(3*i_active_)   = cp->x; 
    dp_.segment<3>(3*i_active_)  = cp->v;  
    if (cp->slipping)
        B_copy.diagonal().segment<3>(3*i_active_).setZero();
    i_active_ += 1;  
  }
  JcT_.noalias() = Jc_.transpose(); 

  CONSIM_START_PROFILER("exponential_simulator::forwardDynamics");
  forwardDynamics(tau_, dv_bar);  
  CONSIM_STOP_PROFILER("exponential_simulator::forwardDynamics");

  
  if(whichFD_==1 || whichFD_==2){
    if(whichFD_==2){
      inverseM_ = pinocchio::computeMinverse(*model_, *data_, q_);
      inverseM_.triangularView<Eigen::StrictlyLower>()
      = inverseM_.transpose().triangularView<Eigen::StrictlyLower>(); // need to fill the Lower part of the matrix
    }
    MinvJcT_.noalias() = inverseM_*JcT_; 
  }
  else if(whichFD_==3){
    // Sparse Cholesky factorization
    for(int i=0; i<JcT_.cols(); i++){
      dv_ = JcT_.col(i);  // use dv_ as temporary buffer
      // pinocchio::cholesky::decompose(*model_, *data_); // decomposition already computed
      pinocchio::cholesky::solve(*model_,*data_,dv_);
      MinvJcT_.col(i) = dv_;
    }
  }

  if(update_A)
  {
    Upsilon_.noalias() =  Jc_*MinvJcT_;
    if(use_diagonal_matrix_exp_)
      tempStepMat_.diagonal().noalias() =  Upsilon_.diagonal().cwiseProduct(K.diagonal());
    else
      tempStepMat_.noalias() =  Upsilon_ * K;
    A.bottomLeftCorner(3*nactive_, 3*nactive_).noalias() = -tempStepMat_;  

    DiagonalMatrixXd* B_to_use;
    if(assumeSlippageContinues_)
      B_to_use = &B_copy; 
    else
      B_to_use = &B; 
    if(use_diagonal_matrix_exp_)
      tempStepMat_.diagonal().noalias() =  Upsilon_.diagonal().cwiseProduct(B_to_use->diagonal());
    else
      tempStepMat_.noalias() = Upsilon_ * (*B_to_use); 
    A.bottomRightCorner(3*nactive_, 3*nactive_).noalias() = -tempStepMat_; 
  }
  temp04_.noalias() = Jc_* dv_bar;  
  b_.noalias() = temp04_ + dJv_; 
  a_.tail(3*nactive_) = b_;
  x0_.head(3*nactive_) = p_-p0_; 
  x0_.tail(3*nactive_) = dp_;
}


void ExponentialSimulator::computeContactForces()
{
  /**
   * computes the kinematics at the end of the integration step, 
   * runs contact detection 
   * resizes matrices to match the number of active contacts if needed 
   * compute the contact forces of the active contacts 
   **/  
  
  CONSIM_START_PROFILER("exponential_simulator::kinematics");
  pinocchio::forwardKinematics(*model_, *data_, q_, v_, fkDv_);
  pinocchio::computeJointJacobians(*model_, *data_);
  pinocchio::updateFramePlacements(*model_, *data_);
  CONSIM_STOP_PROFILER("exponential_simulator::kinematics");

  CONSIM_START_PROFILER("exponential_simulator::contactDetection");
  detectContacts(contacts_); /*!<inactive contacts get automatically filled with zero here */
  CONSIM_STOP_PROFILER("exponential_simulator::contactDetection");

  if (nactive_>0){
    bool size_changed = false;
    if (f_.size()!=3*nactive_){
      size_changed = true;
      CONSIM_START_PROFILER("exponential_simulator::resizeVectorsAndMatrices");
      resizeVectorsAndMatrices();
      CONSIM_STOP_PROFILER("exponential_simulator::resizeVectorsAndMatrices");
    }
    
    CONSIM_START_PROFILER("exponential_simulator::contactKinematics");
    i_active_ = 0; 
    for(auto &cp : contacts_){
      if (!cp->active) continue;
      cp->firstOrderContactKinematics(*data_);
      cp->optr->computePenetration(*cp);
      cp->secondOrderContactKinematics(*data_);
      /*!< computeForce updates the anchor point */ 
      cp->optr->contact_model_->computeForce(*cp);
      f_.segment<3>(3*i_active_) = cp->f; 
      i_active_ += 1;  
      // if (contactChange_){
      //   std::cout<<cp->name_<<" p ["<< cp->x.transpose() << "] v ["<< cp->v.transpose() << "] f ["<<  cp->f.transpose() <<"]"<<std::endl; 
      // }
    }
    CONSIM_STOP_PROFILER("exponential_simulator::contactKinematics");

    if(size_changed){
      // force the update of A
      update_A_counter_ = 1;
    }
  }
} // ExponentialSimulator::computeContactForces



void ExponentialSimulator::computePredictedXandF(){
  /**
   * computes e^{dt*A}
   * computes \int{e^{dt*A}}
   * computes predictedXf = edtA x0 + int_edtA_ * b 
   **/  
  Eigen::internal::set_is_malloc_allowed(true);
  if(compute_predicted_forces_){
    util_eDtA.compute(sub_dt*A,expAdt_);   // TODO: there is memory allocation here 
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
    //  prediction terms also have memory allocation 
    predictedXf_ = expAdt_*x0_ + inteAdt_*a_; 
    predictedForce_ =  D*predictedXf_;
  }
  else{
    predictedXf_ = x0_;
    predictedForce_ = p0_;
    // predictedForce_ = kp0_; // this doesnt seem correct ?
  }
  Eigen::internal::set_is_malloc_allowed(false);
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
  f_avg  = temp03_/sub_dt;

  temp03_.noalias() = D*int2xt_;
  f_avg2.noalias() = temp03_/(0.5*sub_dt*sub_dt);
  
  i_active_ = 0;
  for(unsigned int i=0; i<nc_; i++){
    if (!contacts_[i]->active) continue;

    contacts_[i]->predictedX_ = predictedXf_.segment<3>(3*i_active_); 
    contacts_[i]->predictedV_ = predictedXf_.segment<3>(3*nactive_+3*i_active_);
    contacts_[i]->predictedF_ = predictedForce_.segment<3>(3*i_active_); 

    contacts_[i]->f_avg  = f_avg.segment<3>(3*i_active_);
    contacts_[i]->f_avg2 = f_avg2.segment<3>(3*i_active_);

    if (!contacts_[i]->unilateral) {
      fpr_.segment<3>(3*i_active_) = f_avg.segment<3>(3*i_active_); 
      fpr2_.segment<3>(3*i_active_) = f_avg2.segment<3>(3*i_active_); 
      contacts_[i]->f_prj  = fpr_.segment<3>(3*i_active_);
      contacts_[i]->f_prj2 = fpr2_.segment<3>(3*i_active_);
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

    contacts_[i]->f_prj  = fpr_.segment<3>(3*i_active_);
    contacts_[i]->f_prj2 = fpr2_.segment<3>(3*i_active_);

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
    dp0_.resize(3 * nactive_); dp0_.setZero();
    p_.resize(3 * nactive_); p_.setZero();
    dp_.resize(3 * nactive_); dp_.setZero();
    a_.resize(6 * nactive_); a_.setZero();
    b_.resize(3 * nactive_); b_.setZero();
    x0_.resize(6 * nactive_); x0_.setZero();
    predictedXf_.resize(6 * nactive_); predictedXf_.setZero();
    intxt_.resize(6 * nactive_); intxt_.setZero();
    int2xt_.resize(6 * nactive_); int2xt_.setZero();
    K.resize(3 * nactive_); K.setZero();
    B.resize(3 * nactive_); B.setZero();
    D.resize(3 * nactive_, 6 * nactive_); D.setZero();
    A.resize(6 * nactive_, 6 * nactive_); A.setZero();
    A.block(0, 3*nactive_, 3*nactive_, 3*nactive_) = Eigen::MatrixXd::Identity(3*nactive_, 3*nactive_); 
    Jc_.resize(3 * nactive_, model_->nv); Jc_.setZero();
    JcT_.resize(model_->nv, 3 * nactive_); JcT_.setZero();
    Upsilon_.resize(3 * nactive_, 3 * nactive_); Upsilon_.setZero();
    MinvJcT_.resize(model_->nv, 3*nactive_); MinvJcT_.setZero();
    dJv_.resize(3 * nactive_); dJv_.setZero();
    utilDense_.resize(6 * nactive_);
    f_avg.resize(3 * nactive_); f_avg.setZero();
    f_avg2.resize(3 * nactive_); f_avg2.setZero();
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
    // expAdt_.resize(6 * nactive_, 6 * nactive_); expAdt_.setZero();
    // util_eDtA.resize(6 * nactive_);
    // inteAdt_.resize(6 * nactive_, 6 * nactive_); inteAdt_.setZero();
    // fillout K & B only needed whenever number of active contacts changes 
    i_active_ = 0; 
    for(unsigned int i=0; i<nc_; i++){
      if (!contacts_[i]->active) continue;
      K.diagonal().segment<3>(3*i_active_) = contacts_[i]->optr->contact_model_->stiffness_;
      B.diagonal().segment<3>(3*i_active_) = contacts_[i]->optr->contact_model_->damping_;

      // fill up contact normals and tangents for constraints 
      // cone_constraints_.block<1,3>(4*i_active_, 3*i_active_) = (1/sqrt(2)) * contacts_[i]->optr->contact_model_->friction_coeff_*contacts_[i]->contactNormal_.transpose() - contacts_[i]->contactTangentA_.transpose();
      // cone_constraints_.block<1,3>(4*i_active_+1, 3*i_active_) = (1/sqrt(2)) * contacts_[i]->optr->contact_model_->friction_coeff_*contacts_[i]->contactNormal_.transpose() + contacts_[i]->contactTangentA_.transpose();
      // cone_constraints_.block<1,3>(4*i_active_+2, 3*i_active_) = (1/sqrt(2)) * contacts_[i]->optr->contact_model_->friction_coeff_*contacts_[i]->contactNormal_.transpose() - contacts_[i]->contactTangentB_.transpose();
      // cone_constraints_.block<1,3>(4*i_active_+3, 3*i_active_) = (1/sqrt(2)) * contacts_[i]->optr->contact_model_->friction_coeff_*contacts_[i]->contactNormal_.transpose() + contacts_[i]->contactTangentB_.transpose();
      // eq_cone_constraints_.block<1,3>(i_active_, 3*i_active_) = contacts_[i]->contactNormal_.transpose();

      i_active_ += 1; 
    }
    // fillout D 
    D.leftCols(3*nactive_) = -1*K;
    D.rightCols(3*nactive_) = -1*B; 

    // std::cout<<"resize vectors and matrices"<<std::endl;
    
  } // nactive_ > 0
  
  // std::cout<<"cone constraints \n"<<cone_constraints_<<std::endl; 
  // std::cout<<"contact velocity integrator \n"<<contact_position_integrator_<<std::endl; 


  Eigen::internal::set_is_malloc_allowed(false);
} // ExponentialSimulator::resizeVectorsAndMatrices


/*____________________________________________________________________________________________*/



}  // namespace consim 
