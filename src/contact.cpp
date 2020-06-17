#include "consim/contact.hpp"
#include "consim/object.hpp"
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <math.h>

namespace consim {

#define macro_sign(x) ((x > 0) - (x < 0))
#define sqr(x) (x * x)


ContactPoint::ContactPoint(const pinocchio::Model &model, const std::string & name, unsigned int frameId, unsigned int nv, bool isUnilateral):
        model_(&model), name_(name), frame_id(frameId), unilateral(isUnilateral) {
          active = false; 
          slipping = false;
          f.fill(0);
          predictedF_.fill(0);
          predictedX_.fill(0);
          world_J_.resize(3, nv); world_J_.setZero();
          full_J_.resize(6, nv); full_J_.setZero();
        }

void ContactPoint::updatePosition(pinocchio::Data &data){
  x = data.oMf[frame_id].translation(); 
}

void ContactPoint::firstOrderContactKinematics(pinocchio::Data &data){
  vlocal_ = pinocchio::getFrameVelocity(*model_, data, frame_id); 
  frameSE3_.rotation() = data.oMf[frame_id].rotation();
  v.noalias() = frameSE3_.rotation()*vlocal_.linear();
  pinocchio::getFrameJacobian(*model_, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, full_J_);
  world_J_ = full_J_.topRows<3>(); 
}


void ContactPoint::secondOrderContactKinematics(pinocchio::Data &data){
  dJvlocal_ = pinocchio::getFrameAcceleration(*model_, data, frame_id); 
  dJvlocal_.linear() += vlocal_.angular().cross(vlocal_.linear());
  dJv_ = frameSE3_.act(dJvlocal_).linear();
}


void ContactPoint::resetAnchorPoint(const Eigen::Vector3d &p0, bool slipping){
  x_anchor = p0; 
  v_anchor.setZero();
  this->slipping = slipping;
  predictedX0_ = p0;  
}

void ContactPoint::projectForceInCone(Eigen::Vector3d &f){
  optr->contact_model_->projectForceInCone(f, *this);
}


// --------------------------------------------------------------------------------------------------------// 

LinearPenaltyContactModel::LinearPenaltyContactModel(Eigen::Vector3d &stiffness, Eigen::Vector3d &damping, double frictionCoeff){
  stiffness_=stiffness;
  damping_=damping; 
  friction_coeff_=frictionCoeff;
  for(int i=0; i<3; i++)
    stiffnessInverse_(i) = 1.0/stiffness_(i);
 }


void LinearPenaltyContactModel::computeForce(ContactPoint& cp)
{
  
  if(cp.slipping){
    // assume that if you were slipping at previous iteration you're still slipping
    // TODO: could be better to check whether velocity has changed direction
    cp.v_anchor = cp.v - (cp.v.dot(cp.contactNormal_))*cp.contactNormal_;
  }

  cp.f = stiffness_.cwiseProduct(cp.delta_x) + damping_.cwiseProduct(cp.v_anchor - cp.v); 
  /*!< force along normal to contact object */ 
  normalNorm_ = cp.f.dot(cp.contactNormal_);
  /*!< unilateral force, no pulling into contact object */ 
  if (cp.unilateral && normalNorm_<0){
    cp.f.fill(0);
    cp.slipping = false;
    return;
  } 


  normalF_ = normalNorm_ * cp.contactNormal_; 
  tangentF_ = cp.f - normalF_;
  tangentNorm_ = tangentF_.norm();
  
  if (cp.unilateral && (tangentNorm_ > friction_coeff_*normalNorm_)){
    cp.slipping = true;
    tangentDir_ = tangentF_/tangentNorm_; 
    cp.f = normalF_;
    cp.f += friction_coeff_*normalNorm_*tangentDir_; 
    
    // assume anchor point tangent vel is equal to contact point tangent vel
    cp.v_anchor = cp.v - (cp.v.dot(cp.contactNormal_))*cp.contactNormal_;
    // f = K@(p0-p) + B@(v0-v) => p0 = p + (f - B@(v0-v))/K
    cp.x_anchor = cp.x + stiffnessInverse_.cwiseProduct(cp.f - damping_.cwiseProduct(cp.v_anchor-cp.v));
    return;
  } 
    
  cp.slipping = false;
}

void LinearPenaltyContactModel::projectForceInCone(Eigen::Vector3d &f, ContactPoint& cp)
{
  /*!< force along normal to contact object */ 
  normalNorm_ = f.dot(cp.contactNormal_); 

  /*!< unilateral force, no pulling into contact object */ 
  if (cp.unilateral && normalNorm_<0.0){
    f.setZero();
    return;
  } 
  tangentF_ = f - normalNorm_*cp.contactNormal_;
  tangentNorm_ = tangentF_.norm();
  if (cp.unilateral && (tangentNorm_ > friction_coeff_*normalNorm_)){
    tangentDir_ = tangentF_/tangentNorm_; 
    f = normalNorm_*cp.contactNormal_ + (friction_coeff_*normalNorm_)*tangentDir_; 
  }
}

// --------------------------------------------------------------------------------------------------------// 




} // namespace consim
