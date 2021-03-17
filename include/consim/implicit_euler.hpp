#pragma once

#include <Eigen/Eigen>
#include <Eigen/Cholesky>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/motion.hpp>

#include "consim/simulator.hpp"

namespace consim 
{
/*_______________________________________________________________________________*/

  class ImplicitEulerSimulator : public EulerSimulator
  {
    public: 
      ImplicitEulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps); 
      ~ImplicitEulerSimulator(){};

      /**
       * Implicit Euler first oder step 
      */
      void step(const Eigen::VectorXd &tau) override;

      void set_use_finite_differences_dynamics(bool value);
      bool get_use_finite_differences_dynamics() const;

      void set_use_finite_differences_nle(bool value);
      bool get_use_finite_differences_nle() const;

      void set_use_current_state_as_initial_guess(bool value);
      bool get_use_current_state_as_initial_guess() const;

    protected:      
      int computeDynamics(const Eigen::VectorXd &tau, const Eigen::VectorXd &x, Eigen::VectorXd &f);
      void computeDynamicsAndJacobian(const Eigen::VectorXd &tau, const Eigen::VectorXd &x, Eigen::VectorXd &f, Eigen::MatrixXd &Fx);
      void computeNonlinearEquations(const Eigen::VectorXd &tau, const Eigen::VectorXd &x, const Eigen::VectorXd &xNext, Eigen::VectorXd &out);

      // Eigen::VectorXd vnext_;   // guess used for the iterative search
      Eigen::VectorXd f_;       // evaluation of dynamics function
      Eigen::VectorXd g_;       // residual of ackward Euler integration
      Eigen::VectorXd x_, z_;   // current and next state (including q and v)
      Eigen::VectorXd zNext_;   // temporary variable
      Eigen::VectorXd xIntegrated_; // used in Newton solver, integration of current state x_
      Eigen::VectorXd dz_;      // Newton step expressed in tangent space
      
      Eigen::MatrixXd Fx_;      // dynamics Jacobian matrix
      Eigen::MatrixXd G_;       // gradient in Newton solver
      Eigen::MatrixXd Dintegrate_Ddx_;
      Eigen::MatrixXd Ddifference_Dx0_;
      Eigen::MatrixXd Ddifference_Dx1_;

      Eigen::MatrixXd MinvJcT_;
      Eigen::MatrixXd Jc_;
      DiagonalMatrixXd K_;
      DiagonalMatrixXd B_;
      Eigen::VectorXd lambda_;       // contact forces

      std::vector<ContactPoint *> contactsCopy_;
      bool use_finite_differences_dynamics_;
      bool use_finite_differences_nle_;
      bool use_current_state_as_initial_guess_;
  }; // class ImplicitEulerSimulator

} // namespace consim 
