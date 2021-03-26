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

#pragma once

#include <Eigen/Eigen>
#include <Eigen/Cholesky>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/motion.hpp>

#include <MatrixExponential.hpp>
#include <LDSUtility.hpp>
#include <MatExpIntegral.hpp>


#include "consim/object.hpp"
#include "consim/contact.hpp"
#include "eiquadprog/eiquadprog-fast.hpp"
#include "consim/simulators/base.hpp"

namespace consim 
{

  class ExponentialSimulator : public AbstractSimulator
  {

    public:
      /**
       * slipping metho selects anchor point update method during slipping 
       * 1: compute average force over the integration step, project on the cone boundary then update p0 
       * 2: a QP method to update the anchor point velocity, then average force is computed 
       **/  
      ExponentialSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps,
              int whichFD, EulerIntegrationType type, int slipping_method=1, bool compute_predicted_forces=false, 
              int exp_max_mat_mul=100, int lds_max_mat_mul=100); 

      ~ExponentialSimulator(){};
      void step(const Eigen::VectorXd &tau) override;

      int getMatrixMultiplications(){ return utilDense_.getMatrixMultiplications(); }
      // Return the L1 norm of the last matrix used for computing the matrix exponential
      double getMatrixExpL1Norm(){ return utilDense_.getL1Norm(); }
      void useMatrixBalancing(bool flag){ utilDense_.useBalancing(flag); }
      void assumeSlippageContinues(bool flag){ assumeSlippageContinues_=flag; }
      void setUseDiagonalMatrixExp(bool flag){ use_diagonal_matrix_exp_=flag; }

    protected:
      /**
       * AbstractSimulator::computeContactState() must be called before  
       * calling ExponentialSimulator::computeContactForces()
       */
      void computeContactForces() override; 
      /**
       * computes average contact force during one integration step 
       * loops over the average force to compute tangential and normal force per contact 
       * projects any violations of the cone onto its boundaries 
       * sets a flag to to switch integration mode to include saturated forces 
       */
      void checkFrictionCone(); 

      void resizeVectorsAndMatrices();
      // convenience method to compute terms needed in integration  
      void computeExpLDS();

      int slipping_method_; 
      bool compute_predicted_forces_;
      // expokit 
      expokit::LDSUtility<double, Dynamic> utilDense_;
      const int expMaxMatMul_;
      const int ldsMaxMatMul_; 

      bool assumeSlippageContinues_; // flag deciding whether dp0 is used in force computation 
      bool use_diagonal_matrix_exp_; // flag deciding whether a diagonal approximation of the matrix exponential is used
      
      Eigen::VectorXd f_;  // contact forces
      Eigen::MatrixXd Jc_; // contact Jacobian for all contacts 
      Eigen::VectorXd p0_;  // anchor point positions
      Eigen::VectorXd dp0_; // anchor point velocities
      Eigen::VectorXd p_;   // contact point positions
      Eigen::VectorXd dp_;  // contact point velocities
      Eigen::VectorXd x0_;
      Eigen::VectorXd a_;
      Eigen::VectorXd b_;
      Eigen::VectorXd intxt_;
      Eigen::VectorXd int2xt_;
      Eigen::VectorXd dv_bar; 
      // contact acceleration components 
      Eigen::VectorXd dJv_;  
      DiagonalMatrixXd K;
      DiagonalMatrixXd B;
      DiagonalMatrixXd B_copy;
      Eigen::MatrixXd D;
      Eigen::MatrixXd A; 
      Eigen::MatrixXd MinvJcT_;
      Eigen::MatrixXd Upsilon_;
      Eigen::MatrixXd JcT_; 
      
      // 
      void computePredictedXandF();  // predicts xf at end of integration step 
      expokit::MatrixExponential<double, Dynamic> util_eDtA;
      expokit::MatExpIntegral<double>  util_int_eDtA_one = expokit::MatExpIntegral<double>(6);   // current implementation static 
      expokit::MatExpIntegral<double>  util_int_eDtA_two = expokit::MatExpIntegral<double>(12);   // current implementation static 
      expokit::MatExpIntegral<double>  util_int_eDtA_three = expokit::MatExpIntegral<double>(18);   // current implementation static 
      expokit::MatExpIntegral<double>  util_int_eDtA_four = expokit::MatExpIntegral<double>(24);   // current implementation static 
      /*!< terms to approximate integral of e^{\tau A} */ 

      Eigen::MatrixXd expAdt_; 
      Eigen::MatrixXd inteAdt_;

      
      Eigen::VectorXd predictedForce_;
      Eigen::VectorXd predictedX0_;  
      Eigen::VectorXd predictedXf_; 
      Eigen::VectorXd dvMean_;
      Eigen::VectorXd temp01_;
      Eigen::VectorXd temp02_;
      Eigen::VectorXd temp03_;
      Eigen::VectorXd temp04_;
      Eigen::MatrixXd tempStepMat_; 
      // friction cone 
      Eigen::VectorXd f_avg;  // average force for cone 
      Eigen::VectorXd f_avg2;  // average of average force for cone 
      Eigen::VectorXd fpr_;   // projected force on cone boundaries 
      Eigen::VectorXd fpr2_;   // projected force on cone boundaries
      double cone_direction_; // angle of tangential(to contact surface) force 

      // Eigen::Vector3d f_avg_i; 
      // Eigen::Vector3d normalFi_; // normal component of contact force Fi at contact Ci  
      // Eigen::Vector3d tangentFi_; // normal component of contact force Fi at contact Ci  
      // Eigen::Vector3d f_avg_i2; 
      // Eigen::Vector3d normalFi_2; // normal component of contact force Fi at contact Ci  
      // Eigen::Vector3d tangentFi_2; // normal component of contact force Fi at contact Ci  

      Eigen::VectorXd dvMean2_; // used in method 1 of contact slipping 
      Eigen::VectorXd vMean2_; // used in method 1 of contact slipping 

      

      // double fnor_;   // norm of normalFi_  
      // double ftan_;   // norm of tangentFi_ 
      // double fnor2_;   // norm of normalFi_  
      // double ftan2_;   // norm of tangentFi_ 
      unsigned int i_active_; // index of the active contact      

      /**
       * solves a QP to update anchor points of sliding contacts
       * min || dp0_avg || ^ 2 
       * st. Fc \in Firction Cone
      //  **/  
      // void computeSlipping(); 
      // Eigen::MatrixXd Q_cone; 
      // Eigen::VectorXd q_cone; 
      // Eigen::MatrixXd Cineq_cone; 
      // Eigen::VectorXd cineq_cone; 
      // Eigen::MatrixXd Ceq_cone; 
      // Eigen::VectorXd ceq_cone; 
      // Eigen::VectorXd optdP_cone; 

      

      // Eigen::MatrixXd cone_constraints_;
      // Eigen::MatrixXd eq_cone_constraints_;
      // Eigen::MatrixXd contact_position_integrator_; 
      // Eigen::MatrixXd D_intExpA_integrator; 

      // eiquadprog::solvers::EiquadprogFast_status expected_qp = eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL;

      // eiquadprog::solvers::EiquadprogFast_status status_qp;

  }; // class ExponentialSimulator

} // namespace consim 
