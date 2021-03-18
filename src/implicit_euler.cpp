
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
#include "consim/implicit_euler.hpp"
// #include "consim/utils/stop-watch.hpp"

#include <iostream>

using namespace Eigen;
using namespace std;

// TODO: sqr already defined in contact.cpp 
#define sqr(x) (x * x)

namespace consim {

/* ____________________________________________________________________________________________*/
/** 
 * ImplicitEulerSimulator Class 
*/

ImplicitEulerSimulator::ImplicitEulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps):
EulerSimulator(model, data, dt, n_integration_steps, 3, EXPLICIT),
use_finite_differences_dynamics_(true),
use_finite_differences_nle_(true),
use_current_state_as_initial_guess_(true),
convergence_threshold_(1e-8),
regularization_(1e-10)
{
  const int nv = model.nv, nq=model.nq;
  int nx = nq+nv;
  int ndx = 2*nv;

  Fx_.resize(ndx, ndx); Fx_.setZero();
  G_.resize(ndx, ndx); G_.setZero();
  Dintegrate_Ddx_.resize(ndx, ndx); Dintegrate_Ddx_.setZero();
  Ddifference_Dx0_.resize(ndx, ndx); Ddifference_Dx0_.setZero();
  Ddifference_Dx1_.resize(ndx, ndx); Ddifference_Dx1_.setZero();
  Dintegrate_Ddx_Fx_.resize(ndx, ndx);
  Ddifference_Dx0_Dintegrate_Ddx_Fx_.resize(ndx, ndx);

  Fx_.topLeftCorner(nv, nv).setZero();
  Fx_.topRightCorner(nv, nv).setIdentity(nv,nv);

  x_.resize(nx);x_.setZero();
  z_.resize(nx);z_.setZero();
  zNext_.resize(nx); zNext_.setZero();
  xIntegrated_.resize(nx);xIntegrated_.setZero();
  f_.resize(ndx);f_.setZero();
  g_.resize(ndx);g_.setZero();
  dz_.resize(ndx);dz_.setZero();

  tau_f_.resize(nv); tau_f_.setZero();
  tau_plus_JT_f_.resize(nv); tau_plus_JT_f_.setZero();

  const int nactive=0;
  lambda_.resize(3 * nactive); lambda_.setZero();
  K_.resize(3 * nactive); K_.setZero();
  B_.resize(3 * nactive); B_.setZero();
  Jc_.resize(3 * nactive, nv); Jc_.setZero();
  MinvJcT_.resize(nv, 3*nactive); MinvJcT_.setZero();
  G_LU_ = PartialPivLU<MatrixXd>(ndx);
}

void ImplicitEulerSimulator::set_use_finite_differences_dynamics(bool value) { use_finite_differences_dynamics_ = value; }
bool ImplicitEulerSimulator::get_use_finite_differences_dynamics() const{ return use_finite_differences_dynamics_; }

void ImplicitEulerSimulator::set_use_finite_differences_nle(bool value) { use_finite_differences_nle_ = value; }
bool ImplicitEulerSimulator::get_use_finite_differences_nle() const{ return use_finite_differences_nle_; }

void ImplicitEulerSimulator::set_use_current_state_as_initial_guess(bool value){ use_current_state_as_initial_guess_ = value; }
bool ImplicitEulerSimulator::get_use_current_state_as_initial_guess() const { return use_current_state_as_initial_guess_; }

void ImplicitEulerSimulator::set_convergence_threshold(double value){ convergence_threshold_ = value; }
double ImplicitEulerSimulator::get_convergence_threshold() const{ return convergence_threshold_; }

double ImplicitEulerSimulator::get_avg_iteration_number() const { return avg_iteration_number_; }

int ImplicitEulerSimulator::computeDynamics(const Eigen::VectorXd &tau, const Eigen::VectorXd &x, Eigen::VectorXd &f)
{
  // create a copy of the current contacts
  CONSIM_START_PROFILER("imp_euler_simulator::copyContacts");
  for(auto &cp: contactsCopy_){
    delete cp;
  }
  contactsCopy_.clear();
  for(auto &cp: contacts_){
    contactsCopy_.push_back(new ContactPoint(*cp));
  }
  CONSIM_STOP_PROFILER("imp_euler_simulator::copyContacts");

  const int nq = model_->nq, nv = model_->nv;
  int nactive = computeContactForces_imp(*model_, *data_, x.head(nq), x.tail(nv), tau_f_, contactsCopy_, objects_);
  // cout<<"tau_f: "<<tau_f_.transpose()<<endl;
  tau_plus_JT_f_ = tau + tau_f_;
  pinocchio::aba(*model_, *data_, x.head(nq), x.tail(nv), tau_plus_JT_f_);
  f.head(nv) = x.tail(nv);
  f.tail(nv) = data_-> ddq;
  
  return nactive;
}

void ImplicitEulerSimulator::computeDynamicsJacobian(const Eigen::VectorXd &tau, const Eigen::VectorXd &x, 
                                                     const Eigen::VectorXd &f, Eigen::MatrixXd &Fx)
{
  CONSIM_START_PROFILER("imp_euler_simulator::computeDynamicsJacobian");
  const int nq = model_->nq, nv = model_->nv;

  if(use_finite_differences_dynamics_)
  {  
    const double epsilon = 1e-8;
    VectorXd x_eps(nq+nv), dx_eps(2*nv), f_eps(2*nv);
    for(int i=0; i<2*nv; ++i)
    {
      // cout<<"Fx "<<i<<endl;
      dx_eps.setZero();
      dx_eps[i] = epsilon;
      integrateState(*model_, x, dx_eps, 1.0, x_eps);
      computeDynamics(tau, x_eps, f_eps);
      // cout<<"xEps: "<<x_eps.transpose()<<endl;
      // cout<<"fEps: "<<f_eps.transpose()<<endl;
      // cout<<"Fx[i]: "<<((f_eps.tail(nv) - f.tail(nv))/epsilon).transpose()<<endl;
      Fx.block(nv, i, nv, 1) = (f_eps.tail(nv) - f.tail(nv))/epsilon;
    }
  }
  else
  {  
    // compute contact Jacobian and contact forces
    if (nactive_>0){
      CONSIM_START_PROFILER("imp_euler_simulator::update_J_K_B_f");
      lambda_.resize(3 * nactive_); lambda_.setZero();
      K_.resize(3 * nactive_); K_.setZero();
      B_.resize(3 * nactive_); B_.setZero();
      Jc_.resize(3 * nactive_, model_->nv); Jc_.setZero();
      MinvJcT_.resize(model_->nv, 3*nactive_); MinvJcT_.setZero();
  
      int i_active_ = 0; 
      for(unsigned int i=0; i<nc_; i++){
        ContactPoint *cp = contactsCopy_[i];
        if (!cp->active) continue;
        Jc_.block(3*i_active_,0,3,model_->nv) = cp->world_J_;
        K_.diagonal().segment<3>(3*i_active_) = cp->optr->contact_model_->stiffness_;
        B_.diagonal().segment<3>(3*i_active_) = cp->optr->contact_model_->damping_;
        lambda_.segment<3>(3*i_active_) = cp->f;
        i_active_ += 1; 
      }
      CONSIM_STOP_PROFILER("imp_euler_simulator::update_J_K_B_f");
    } 

    CONSIM_START_PROFILER("imp_euler_simulator::computeABADerivatives");
    pinocchio::computeABADerivatives(*model_, *data_, x.head(nq), x.tail(nv), tau);
    Fx.bottomLeftCorner(nv, nv) = data_->ddq_dq;
    Fx.bottomRightCorner(nv, nv) = data_->ddq_dv;
    CONSIM_STOP_PROFILER("imp_euler_simulator::computeABADerivatives");

    if (nactive_>0){
      CONSIM_START_PROFILER("imp_euler_simulator::Minv_JT_K_J");
      data_->Minv.triangularView<Eigen::StrictlyLower>()
      = data_->Minv.transpose().triangularView<Eigen::StrictlyLower>(); // need to fill the Lower part of the matrix
      MinvJcT_.noalias() = data_->Minv * Jc_.transpose(); 
      Fx.bottomLeftCorner(nv, nv)  -= MinvJcT_ * K_ * Jc_;
      Fx.bottomRightCorner(nv, nv) -= MinvJcT_ * B_ * Jc_;
      CONSIM_STOP_PROFILER("imp_euler_simulator::Minv_JT_K_J");
    }
    // cout<<"Fx:\n"<<Fx<<endl;
  }
  CONSIM_STOP_PROFILER("imp_euler_simulator::computeDynamicsJacobian");
}

void ImplicitEulerSimulator::computeNonlinearEquations(const Eigen::VectorXd &tau, const Eigen::VectorXd &x, 
                                                       const Eigen::VectorXd &xNext, Eigen::VectorXd &out)
{
  CONSIM_START_PROFILER("imp_euler_simulator::computeNonlinearEquations");
  // out = g = xNext - (x + h*f)
  // cout<<"**** compute NLE\n";
  nactive_ = computeDynamics(tau, xNext, f_);
  // cout<<"xNext="<<xNext.transpose()<<"\nf(xNext)="<<f.transpose()<<endl;
  integrateState(*model_, x, f_, sub_dt, xIntegrated_);
  // cout<<"xIntegrated="<<xIntegrated.transpose()<<endl;
  differenceState(*model_, xIntegrated_, xNext, out);
  CONSIM_STOP_PROFILER("imp_euler_simulator::computeNonlinearEquations");
}

void ImplicitEulerSimulator::step(const Eigen::VectorXd &tau) 
{
  if(!resetflag_){
    throw std::runtime_error("resetState() must be called first !");
  }
  CONSIM_START_PROFILER("imp_euler_simulator::step");
  assert(tau.size() == model_->nv);

  avg_iteration_number_ = 0.0;
  for (int i = 0; i < n_integration_steps_; i++)
  {
    // Eigen::internal::set_is_malloc_allowed(false);
    CONSIM_START_PROFILER("imp_euler_simulator::substep");
    // add input control to contact forces J^T*f that are already in tau_
    tau_ += tau;
    // \brief joint damping 
    if (joint_friction_flag_){
      tau_ -= joint_friction_.cwiseProduct(v_);
    }
    
    x_.head(model_->nq) = q_;
    x_.tail(model_->nv) = v_;
    
    if(use_current_state_as_initial_guess_)
    {
      // use current state as initial guess
      z_ = x_;
    }
    else
    {    
      /*!< integrate twice with explicit Euler to compute initial guess */ 
      // z = x[i,:] + h*ode.f(x[i,:], U[ii,:], t[i])
      forwardDynamics(tau_, dv_);
      pinocchio::integrate(*model_, q_, v_ * sub_dt, qnext_);
      z_.head(model_->nq) = qnext_;
      z_.tail(model_->nv) = v_ + sub_dt*dv_;
    }

    //   Solve the following system of equations for z:
    //       g(z) = z - x - h*f(z) = 0
    //   Start by computing the Newton step:
    //       g(z) = g(z_i) + G(z_i)*dz = 0 => dz = -G(z_i)^-1 * g(z_i)
    //   where G is the Jacobian of g and z_i is our current guess of z
    //       G(z) = I - h*F(z)
    //   where F(z) is the Jacobian of f wrt z.

    CONSIM_START_PROFILER("imp_euler_simulator::computeResidual");
    tau_ = tau;
    computeNonlinearEquations(tau_, x_, z_, g_);
    // g_ = z_ - xIntegrated_;
    double residual = g_.norm();
    CONSIM_STOP_PROFILER("imp_euler_simulator::computeResidual");
    
    bool converged = false;
    int j=0;
    
    for(; j<30; ++j)
    {
      if(residual < convergence_threshold_){
        converged = true;
        break;
      }
      
      if(use_finite_differences_nle_)
      {
        const int ndx = 2*model_->nv;
        VectorXd delta_z_eps(ndx), zEps(z_.size()), gEps(ndx);
        const double eps = 1e-8;
        for(int k=0; k<ndx; ++k)
        {
          // perturb z in direction k
          // cout<<"k="<<k<<endl;
          delta_z_eps.setZero();
          delta_z_eps(k) = eps;
          integrateState(*model_, z_, delta_z_eps, 1.0, zEps);
          // recompute nonlinear equations with perturbed z
          computeNonlinearEquations(tau_, x_, zEps, gEps);
          G_.col(k) = (gEps - g_)/eps;
        }
      }
      else
      {
        CONSIM_START_PROFILER("imp_euler_simulator::computeNewtonSystem");
        // Compute gradient G = I - h*Fx
        computeDynamicsJacobian(tau_, z_, f_, Fx_);
        // g = diff(int(x, h*f(z)), z)
        // G = Dg/Dz = Ddiff_Dx1 + h * Ddiff_Dx0 * Dint * Fx
        DintegrateState(    *model_, x_, f_, sub_dt,   Dintegrate_Ddx_);
        // cout<<"Dintegrate_Ddx_ = \n"<<Dintegrate_Ddx_<<endl;
        DdifferenceState_x0(*model_, xIntegrated_, z_, Ddifference_Dx0_);
        // cout<<"Ddifference_Dx0_ = \n"<<Ddifference_Dx0_<<endl;
        DdifferenceState_x1(*model_, xIntegrated_, z_, Ddifference_Dx1_);
        // cout<<"Ddifference_Dx1_ = \n"<<Ddifference_Dx1_<<endl;
        Dintegrate_Ddx_Fx_.noalias() = Dintegrate_Ddx_ * Fx_;
        Ddifference_Dx0_Dintegrate_Ddx_Fx_.noalias() = Ddifference_Dx0_ * Dintegrate_Ddx_Fx_;
        G_.noalias() = sub_dt * Ddifference_Dx0_Dintegrate_Ddx_Fx_;
        G_ += Ddifference_Dx1_;
        // G_.setIdentity();
        // G_ -= sub_dt * Fx_;
        CONSIM_STOP_PROFILER("imp_euler_simulator::computeNewtonSystem");
      }
      // cout<<"G\n"<<G_<<endl;

      CONSIM_START_PROFILER("imp_euler_simulator::solveNewtonSystem");
      // Update with Newton step: z += solve(G, -g)
      g_ *= -1;
      G_ += regularization_ * MatrixXd::Identity(2*model_->nv, 2*model_->nv);
      G_LU_.compute(G_);
      dz_ = G_LU_.solve(g_);
      // dz_ = G_.colPivHouseholderQr().solve(g_); // slower than LU
      // dz_ = G_.partialPivLu().solve(g_);
      // dz_ = G_.ldlt().solve(g_); // cannot use LDLT decomposition because G is not PD in general
      // cout<<"dz = "<<dz_.transpose()<<endl;
      CONSIM_STOP_PROFILER("imp_euler_simulator::solveNewtonSystem");

      CONSIM_START_PROFILER("imp_euler_simulator::lineSearch");
      double alpha = 1.0, new_residual;
      bool line_search_converged = false;
      for(int k=0; k<20 && !line_search_converged; ++k)
      {
        integrateState(*model_, z_, dz_, alpha, zNext_);
        computeNonlinearEquations(tau_, x_, zNext_, g_);
        // // g = z - x[i,:] - h*f
        // cout<<"   line search "<<k<<" g="<<g_.transpose()<<endl;
        // cout<<"   line search "<<k<<" |g|="<<g_.norm()<<endl;
        new_residual = g_.norm();
        if(new_residual >= residual){
          alpha *= 0.5;
        }
        else{
          line_search_converged = true;
          residual = new_residual;
          z_ = zNext_;
          if(k==0){
            regularization_ *= 0.1;
            if(regularization_<1e-10)
              regularization_ = 1e-10;
          }
            
        }
      } // end of line search
      CONSIM_STOP_PROFILER("imp_euler_simulator::lineSearch");

      if(!line_search_converged)
      {
        regularization_ *= 10;
        if(regularization_ > 1e-3){
          regularization_ = 1e-3;
          break;
        }
        // cout<<"t "<<elapsedTime_<<" substep "<<i<<" iter "<<j << " increase reg to "<<regularization_<<" |g|="<<residual<<endl;
        // cout<<"Iter "<<j<<". Line search did not converge. new residual: "<<new_residual<<" old residual: "<<residual<<endl;
        // recompute residual, just for error print
        // computeNonlinearEquations(tau_, x_, z_, g_);
        // break;
      }
      // cout<<"z="<<z_.transpose()<<endl;
    }
    avg_iteration_number_ += j;

    // if(!converged && residual>=convergence_threshold_)
      // cout<<"Substep "<<i<<" iter "<<j<<" Implicit Euler did not converge!!!! |g|="<<residual<<endl;
    
    q_ = z_.head(model_->nq);
    v_ = z_.tail(model_->nv);
    
    tau_.setZero();
    // \brief adds contact forces to tau_
    computeContactForces(); 
    // Eigen::internal::set_is_malloc_allowed(true);
    CONSIM_STOP_PROFILER("imp_euler_simulator::substep");
    elapsedTime_ += sub_dt; 
  }
  avg_iteration_number_ /= n_integration_steps_;
  CONSIM_STOP_PROFILER("imp_euler_simulator::step");
}

}  // namespace consim 
