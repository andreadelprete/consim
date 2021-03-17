
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
use_current_state_as_initial_guess_(true)
{
  int nx = model.nq+model.nv;
  int ndx = 2*model.nv;

  Fx_.resize(ndx, ndx); Fx_.setZero();
  G_.resize(ndx, ndx); G_.setZero();
  Dintegrate_Ddx_.resize(ndx, ndx); Dintegrate_Ddx_.setZero();
  Ddifference_Dx0_.resize(ndx, ndx); Ddifference_Dx0_.setZero();
  Ddifference_Dx1_.resize(ndx, ndx); Ddifference_Dx1_.setZero();

  x_.resize(nx);x_.setZero();
  z_.resize(nx);z_.setZero();
  zNext_.resize(nx); zNext_.setZero();
  xIntegrated_.resize(nx);xIntegrated_.setZero();
  f_.resize(ndx);f_.setZero();
  g_.resize(ndx);g_.setZero();
  dz_.resize(ndx);dz_.setZero();

  tau_f_.resize(model.nv); tau_f_.setZero();

  const int nactive=0;
  lambda_.resize(3 * nactive); lambda_.setZero();
  K_.resize(3 * nactive); K_.setZero();
  B_.resize(3 * nactive); B_.setZero();
  Jc_.resize(3 * nactive, model_->nv); Jc_.setZero();
  MinvJcT_.resize(model_->nv, 3*nactive); MinvJcT_.setZero();
}

void ImplicitEulerSimulator::set_use_finite_differences_dynamics(bool value) { use_finite_differences_dynamics_ = value; }
bool ImplicitEulerSimulator::get_use_finite_differences_dynamics() const{ return use_finite_differences_dynamics_; }

void ImplicitEulerSimulator::set_use_finite_differences_nle(bool value) { use_finite_differences_nle_ = value; }
bool ImplicitEulerSimulator::get_use_finite_differences_nle() const{ return use_finite_differences_nle_; }

void ImplicitEulerSimulator::set_use_current_state_as_initial_guess(bool value){ use_current_state_as_initial_guess_ = value; }
bool ImplicitEulerSimulator::get_use_current_state_as_initial_guess() const { return use_current_state_as_initial_guess_; }

int ImplicitEulerSimulator::computeDynamics(const Eigen::VectorXd &tau, const Eigen::VectorXd &x, Eigen::VectorXd &f)
{
  // create a copy of the current contacts
  for(auto &cp: contactsCopy_){
    delete cp;
  }
  contactsCopy_.clear();
  for(auto &cp: contacts_){
    contactsCopy_.push_back(new ContactPoint(*cp));
  }

  const int nq = model_->nq, nv = model_->nv;
  int nactive = computeContactForces_imp(*model_, *data_, x.head(nq), x.tail(nv), tau_f_, contactsCopy_, objects_);
  // cout<<"tau_f: "<<tau_f_.transpose()<<endl;
  VectorXd tau_plus_f = tau + tau_f_;
  pinocchio::aba(*model_, *data_, x.head(nq), x.tail(nv), tau_plus_f);
  f.head(nv) = x.tail(nv);
  f.tail(nv) = data_-> ddq;
  
  return nactive;
}

void ImplicitEulerSimulator::computeDynamicsAndJacobian(const Eigen::VectorXd &tau, const Eigen::VectorXd &x, 
                                                        Eigen::VectorXd &f, Eigen::MatrixXd &Fx)
{
  const int nq = model_->nq, nv = model_->nv;
  const double epsilon = 1e-8;

  int nactive = computeDynamics(tau, x, f);

  Fx.topLeftCorner(nv, nv).setZero();
  Fx.topRightCorner(nv, nv).setIdentity(nv,nv);

  if(use_finite_differences_dynamics_)
  {  
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
    // cout<<"nactive "<<nactive<<endl;
    if (nactive>0){
      lambda_.resize(3 * nactive); lambda_.setZero();
      K_.resize(3 * nactive); K_.setZero();
      B_.resize(3 * nactive); B_.setZero();
      Jc_.resize(3 * nactive, model_->nv); Jc_.setZero();
      MinvJcT_.resize(model_->nv, 3*nactive); MinvJcT_.setZero();
  
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
    } 

    pinocchio::computeABADerivatives(*model_, *data_, x.head(nq), x.tail(nv), tau);
    Fx.bottomLeftCorner(nv, nv) = data_->ddq_dq;
    Fx.bottomRightCorner(nv, nv) = data_->ddq_dv;

    if (nactive>0){
      data_->Minv.triangularView<Eigen::StrictlyLower>()
      = data_->Minv.transpose().triangularView<Eigen::StrictlyLower>(); // need to fill the Lower part of the matrix
      MinvJcT_.noalias() = data_->Minv * Jc_.transpose(); 
      Fx.bottomLeftCorner(nv, nv)  -= MinvJcT_ * K_ * Jc_;
      Fx.bottomRightCorner(nv, nv) -= MinvJcT_ * B_ * Jc_;
    }
    // cout<<"Fx:\n"<<Fx<<endl;
  }
}

void ImplicitEulerSimulator::computeNonlinearEquations(const Eigen::VectorXd &tau, const Eigen::VectorXd &x, 
                                                       const Eigen::VectorXd &xNext, Eigen::VectorXd &out)
{
  // out = g = xNext - (x + h*f)
  // cout<<"**** compute NLE\n";
  VectorXd f(model_->nv*2), xIntegrated(x.size());
  computeDynamics(tau, xNext, f);
  // cout<<"xNext="<<xNext.transpose()<<"\nf(xNext)="<<f.transpose()<<endl;
  integrateState(*model_, x, f, sub_dt, xIntegrated);
  // cout<<"xIntegrated="<<xIntegrated.transpose()<<endl;
  differenceState(*model_, xIntegrated, xNext, out);
}

void ImplicitEulerSimulator::step(const Eigen::VectorXd &tau) 
{
  if(!resetflag_){
    throw std::runtime_error("resetState() must be called first !");
  }
  CONSIM_START_PROFILER("imp_euler_simulator::step");
  assert(tau.size() == model_->nv);

  for (int i = 0; i < n_integration_steps_; i++)
  {
    // Eigen::internal::set_is_malloc_allowed(false);
    CONSIM_START_PROFILER("imp_euler_simulator::substep");
    // \brief add input control 
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
      // cout<<"Initial guess"<<endl;
      // cout<<"q="<<q_.transpose()<<endl;
      // cout<<"v="<<v_.transpose()<<endl;
      // cout<<"dv="<<dv_.transpose()<<endl;
      // cout<<"z="<<z_.transpose()<<endl ;
    }

    //   Solve the following system of equations for z:
    //       g(z) = z - x - h*f(z) = 0
    //   Start by computing the Newton step:
    //       g(z) = g(z_i) + G(z_i)*dz = 0 => dz = -G(z_i)^-1 * g(z_i)
    //   where G is the Jacobian of g and z_i is our current guess of z
    //       G(z) = I - h*F(z)
    //   where F(z) is the Jacobian of f wrt z.
    bool converged = false;
    double residual;
    for(int j=0; j<5; ++j)
    {
      // cout<<"j = "<<j<<endl;
      // cout<<"  z = "<<z_.transpose()<<endl;
      tau_ = tau;
      if(use_finite_differences_nle_)
      {
        computeNonlinearEquations(tau_, x_, z_, g_);
        residual = g_.norm();
        // cout<<"g="<<g_.transpose()<<endl;
        // cout<<"  |g|="<<g_.norm()<<endl;
        if(residual < 1e-10){
          converged = true;
          break;
        }

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
          // cout<<"zEps="<<zEps.transpose()<<endl;
          // recompute nonlinear equations with perturbed z
          computeNonlinearEquations(tau_, x_, zEps, gEps);
          G_.col(k) = (gEps - g_)/eps;
          // cout<<"gEps="<<gEps.transpose()<<endl;
        }
      }
      else
      {
        computeDynamicsAndJacobian(tau_, z_, f_, Fx_);
        // g = z - x[i,:] - h*f
        // cout<<"f = "<<f_.transpose()<<endl;
        // cout<<"Fx = \n"<<Fx_<<endl;
        integrateState(*model_, x_, f_, sub_dt, xIntegrated_);
        // cout<<"xIntegrated: "<<xIntegrated_.transpose()<<endl;
        differenceState(*model_, xIntegrated_, z_, g_);
        // g_ = z_ - xIntegrated_;
        // cout<<"g="<<g_.transpose()<<endl;
        // cout<<j<<" |g|="<<g_.norm()<<endl;
        residual = g_.norm();
        if(residual < 1e-10){
          converged = true;
          break;
        }
        // Compute gradient G = I - h*Fx
        // g = diff(int(x, h*f(z)), z)
        // G = Dg/Dz = Ddiff_Dx1 + h * Ddiff_Dx0 * Dint * Fx
        DintegrateState(    *model_, x_, f_, sub_dt,   Dintegrate_Ddx_);
        // cout<<"Dintegrate_Ddx_ = \n"<<Dintegrate_Ddx_<<endl;
        DdifferenceState_x0(*model_, xIntegrated_, z_, Ddifference_Dx0_);
        // cout<<"Ddifference_Dx0_ = \n"<<Ddifference_Dx0_<<endl;
        DdifferenceState_x1(*model_, xIntegrated_, z_, Ddifference_Dx1_);
        // cout<<"Ddifference_Dx1_ = \n"<<Ddifference_Dx1_<<endl;
        G_ = sub_dt * Ddifference_Dx0_ * Dintegrate_Ddx_ * Fx_;
        G_ += Ddifference_Dx1_;
        // G_ = sub_dt * Ddifference_Dx1_ * Dintegrate_Ddx_ * Fx_;
        // G_ += Ddifference_Dx0_;
      }
      // cout<<"G\n"<<G_<<endl;

      // Update with Newton step: z += solve(G, -g)
      g_ *= -1;
      dz_ = G_.colPivHouseholderQr().solve(g_);
      // dz_ = G_.ldlt().solve(g_); // cannot use LDLT decomposition because G is not PD in general
      // cout<<"dz = "<<dz_.transpose()<<endl;
      double alpha = 1.0, new_residual;
      bool line_search_converged = false;
      for(int k=0; k<20 && !line_search_converged; ++k)
      {
        integrateState(*model_, z_, dz_, alpha, zNext_);
        tau_ = tau;   // reset tau here because it is modified in computeDynamics
        // cout<<"j = "<<j<<endl;
        computeNonlinearEquations(tau_, x_, zNext_, g_);
        // computeDynamics(tau_, zNext_, f_);
        // // g = z - x[i,:] - h*f
        // integrateState(*model_, x_, f_, sub_dt, xIntegrated_);
        // differenceState(*model_, xIntegrated_, zNext_, g_);
        // g_ = z_ - xIntegrated_;
        // cout<<"   line search "<<k<<" g="<<g_.transpose()<<endl;
        // cout<<"   line search "<<k<<" |g|="<<g_.norm()<<endl;
        new_residual = g_.norm();
        if(new_residual > residual){
          alpha *= 0.5;
        }
        else{
          line_search_converged = true;
        }
      } // end of line search

      if(line_search_converged)
      {
        z_ = zNext_;
      }
      else
      {
        cout<<j<<" Line search did not converge. new residual: "<<new_residual<<" old residual: "<<residual<<endl;
        if(new_residual > 10*residual)
          cout<<"Newton step: "<<dz_.transpose()<<endl;
        tau_ = tau;   // reset tau here because it is modified in computeDynamics
        computeNonlinearEquations(tau_, x_, z_, g_);
        // computeDynamics(tau_, z_, f_);
        // integrateState(*model_, x_, f_, sub_dt, xIntegrated_);
        // differenceState(*model_, xIntegrated_, z_, g_);
        break;
      }
      // cout<<"z="<<z_.transpose()<<endl;
    }


    if(!converged)
      cout<<i<<" Implicit Euler did not converge!!!! |g|="<<g_.norm()<<endl;
    
    q_ = z_.head(model_->nq);
    v_ = z_.tail(model_->nv);
    
    tau_.setZero();
    // \brief adds contact forces to tau_
    computeContactForces(); 
    // Eigen::internal::set_is_malloc_allowed(true);
    CONSIM_STOP_PROFILER("imp_euler_simulator::substep");
    elapsedTime_ += sub_dt; 
  }
  CONSIM_STOP_PROFILER("imp_euler_simulator::step");
}

}  // namespace consim 
