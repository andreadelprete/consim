
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
// #include "consim/utils/stop-watch.hpp"

#include <iostream>

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
  J.bottomRightCorner(model.nv, model.nv) = MatrixXd::Identity(model.nv, model.nv) * dt;
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

AbstractSimulator::AbstractSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD, EulerIntegrationType type): 
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
  elapsedTime_ = 0.;  
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
 * ImplicitEulerSimulator Class 
*/

ImplicitEulerSimulator::ImplicitEulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps):
EulerSimulator(model, data, dt, n_integration_steps, 3, EXPLICIT) 
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

void ImplicitEulerSimulator::computeDynamicsAndJacobian(Eigen::VectorXd &tau, Eigen::VectorXd &q , Eigen::VectorXd &v, 
                                                        Eigen::VectorXd &f, Eigen::MatrixXd &Fx)
{
  // create a copy of the current contacts
  for(auto &cp: contactsCopy_){
    delete cp;
  }
  contactsCopy_.clear();
  for(auto &cp: contacts_){
    contactsCopy_.push_back(new ContactPoint(*cp));
  }

  int nactive = computeContactForces_imp(*model_, *data_, q, v, tau_f_, contactsCopy_, objects_);
  tau += tau_f_;
  pinocchio::aba(*model_, *data_, q, v, tau);
  int nv = model_->nv;
  f.head(nv) = v;
  f.tail(nv) = data_-> ddq;

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

  pinocchio::computeABADerivatives(*model_, *data_, q, v, tau);
  Fx.topLeftCorner(nv, nv).setZero();
  Fx.topRightCorner(nv, nv).setIdentity(nv,nv);
  Fx.bottomLeftCorner(nv, nv) = data_->ddq_dq;
  Fx.bottomRightCorner(nv, nv) = data_->ddq_dv;

  if (nactive>0){
    data_->Minv.triangularView<Eigen::StrictlyLower>()
    = data_->Minv.transpose().triangularView<Eigen::StrictlyLower>(); // need to fill the Lower part of the matrix
    MinvJcT_.noalias() = data_->Minv * Jc_.transpose(); 
    Fx.bottomLeftCorner(nv, nv) -= MinvJcT_ * K_ * Jc_;
  }
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
    
    /*!< integrate twice with explicit Euler to compute initial guess */ 
    // z = x[i,:] + h*ode.f(x[i,:], U[ii,:], t[i])
    forwardDynamics(tau_, dv_);
    pinocchio::integrate(*model_, q_, v_ * sub_dt, qnext_);
    vnext_ = v_ + sub_dt*dv_;

    //   Solve the following system of equations for z:
    //       g(z) = z - x - h*f(z) = 0
    //   Start by computing the Newton step:
    //       g(z) = g(z_i) + G(z_i)*dz = 0 => dz = -G(z_i)^-1 * g(z_i)
    //   where G is the Jacobian of g and z_i is our current guess of z
    //       G(z) = I - h*F(z)
    //   where F(z) is the Jacobian of f wrt z.
    bool converged = false;
    x_.head(model_->nq) = q_;
    x_.tail(model_->nv) = v_;
    z_.head(model_->nq) = qnext_;
    z_.tail(model_->nv) = vnext_;

    for(int j=0; j<50; ++j)
    {
      tau_ = tau;
      // cout<<"j = "<<j<<endl;
      computeDynamicsAndJacobian(tau_, qnext_, vnext_, f_, Fx_);
      // g = z - x[i,:] - h*f
      // cout<<"f = "<<f_.transpose()<<endl;
      // cout<<"Fx = \n"<<Fx_<<endl;
      integrateState(*model_, x_, f_, sub_dt, xIntegrated_);
      // cout<<"xIntegrated: "<<xIntegrated_.transpose()<<endl;
      differenceState(*model_, xIntegrated_, z_, g_);
      // g_ = z_ - xIntegrated_;
      // cout<<"g="<<g_.transpose()<<endl;
      // cout<<"|g|="<<g_.norm()<<endl;
      double residual = g_.norm();
      if(residual < 1e-10){
        converged = true;
        break;
      }
      // Compute gradient G = I - h*Fx
      DintegrateState(    *model_, x_, f_, sub_dt,   Dintegrate_Ddx_);
      // cout<<"Dintegrate_Ddx_ = \n"<<Dintegrate_Ddx_<<endl;
      DdifferenceState_x0(*model_, xIntegrated_, z_, Ddifference_Dx0_);
      // cout<<"Ddifference_Dx0_ = \n"<<Ddifference_Dx0_<<endl;
      DdifferenceState_x1(*model_, xIntegrated_, z_, Ddifference_Dx1_);
      // cout<<"Ddifference_Dx1_ = \n"<<Ddifference_Dx1_<<endl;
      G_ = sub_dt * Ddifference_Dx1_ * Dintegrate_Ddx_ * Fx_;
      G_ += Ddifference_Dx0_;
      // cout<<"G\n"<<G_<<endl;

      // Update with Newton step: z += solve(G, -g)
      dz_ = G_.ldlt().solve(g_);
      // cout<<"dz = "<<dz_.transpose()<<endl;
      double alpha = 1.0;
      bool line_search_converged = false;
      for(int k=0; k<50 && !line_search_converged; ++k)
      {
        integrateState(*model_, z_, dz_, alpha, zNext_);
        tau_ = tau;
        qnext_ = zNext_.head(model_->nq);
        vnext_ = zNext_.tail(model_->nv);
        // cout<<"j = "<<j<<endl;
        computeDynamicsAndJacobian(tau_, qnext_, vnext_, f_, Fx_);
        // g = z - x[i,:] - h*f
        // cout<<"f = "<<f_.transpose()<<endl;
        // cout<<"Fx = \n"<<Fx_<<endl;
        integrateState(*model_, x_, f_, sub_dt, xIntegrated_);
        // cout<<"xIntegrated: "<<xIntegrated_.transpose()<<endl;
        differenceState(*model_, xIntegrated_, zNext_, g_);
        // g_ = z_ - xIntegrated_;
        // cout<<"g="<<g_.transpose()<<endl;
        // cout<<"|g|="<<g_.norm()<<endl;
        double new_residual = g_.norm();
        if(new_residual > residual){
          alpha *= 0.5;
        }
        else{
          line_search_converged = true;
        }
      }
      
      z_ = zNext_;
      qnext_ = z_.head(model_->nq);
      vnext_ = z_.tail(model_->nv);
      // cout<<"z="<<z_.transpose()<<endl;
    }


    if(!converged)
      cout<<"Implicit Euler did not converge!!!! |g|="<<g_.norm()<<endl;
    
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
                                            assumeSlippageContinues_(true)
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

  for (unsigned int i = 0; i < n_integration_steps_; i++){
    CONSIM_START_PROFILER("exponential_simulator::substep"); 
    if (nactive_> 0){
      Eigen::internal::set_is_malloc_allowed(false);
      CONSIM_START_PROFILER("exponential_simulator::computeExpLDS");
      computeExpLDS();
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


void ExponentialSimulator::computeExpLDS(){
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

  Upsilon_.noalias() =  Jc_*MinvJcT_;
  tempStepMat_.noalias() =  Upsilon_ * K;
  A.bottomLeftCorner(3*nactive_, 3*nactive_).noalias() = -tempStepMat_;  
  if(assumeSlippageContinues_)
    tempStepMat_.noalias() = Upsilon_ * B_copy; 
  else
    tempStepMat_.noalias() = Upsilon_ * B; 
  A.bottomRightCorner(3*nactive_, 3*nactive_).noalias() = -tempStepMat_; 
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
    if (f_.size()!=3*nactive_){
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
