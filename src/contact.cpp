#include "consim/contact.hpp"
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <math.h>

namespace consim {

#define macro_sign(x) ((x > 0) - (x < 0))
#define sqr(x) (x * x)


ContactPoint::ContactPoint(std::string name, unsigned int frameId, unsigned int nv, bool isUnilateral): 
        name_(name), frame_id(frameId), unilateral(isUnilateral) {
          active = false; 
          friction_flag = false; 
          f.fill(0);
          dJdt_.resize(3, nv); dJdt_.setZero();
          world_J_.resize(3, nv); world_J_.setZero();
          full_J_.resize(6, nv); full_J_.setZero();
        }

void ContactPoint::updatePosition(const pinocchio::Model &model, pinocchio::Data &data){
  x = data.oMf[frame_id].translation(); 
}

void ContactPoint::firstOrderContactKinematics(const pinocchio::Model &model, pinocchio::Data &data){
  vlocal_ = pinocchio::getFrameVelocity(model, data, frame_id); 
  frameSE3_.rotation() = data.oMf[frame_id].rotation();
  v.noalias() = frameSE3_.rotation()*vlocal_.linear();
  pinocchio::getFrameJacobian(model, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, full_J_);
  world_J_ = full_J_.topRows<3>();
  /** compute displacement relative to contact object
   * dx: relative penetration 
   * normal: penetration along normal to contact object
   * tangent: penetration along tangent to contact object
   * normalvel: velocity along normal to contact object
   * tanvel: velocity along tangent to contact object
   * */ 
  dx = x_start - x; 
  normal = dx.dot(optr->contact_normal) * optr->contact_normal; 
  tangent = dx - normal; 
  normvel = (v).dot(optr->contact_normal) * optr->contact_normal; 
  tanvel = v - normvel; 
}


void ContactPoint::secondOrderContactKinematics(const pinocchio::Model &model, pinocchio::Data &data){
  dJvlocal_ = pinocchio::getFrameAcceleration(model, data, frame_id); 
  dJvlocal_.linear() += vlocal_.angular().cross(vlocal_.linear());
  dJv_ = frameSE3_.act(dJvlocal_).linear();
}

void ContactPoint::computeContactForce(){
  optr->contact_model_->computeForce(*this);
}

// --------------------------------------------------------------------------------------------------------// 

LinearPenaltyContactModel::LinearPenaltyContactModel(Eigen::Matrix3d stiffness, Eigen::Matrix3d damping, double frictionCoeff){
  stiffness_=stiffness;
  damping_=damping; 
  friction_coeff_=frictionCoeff;
 }


void LinearPenaltyContactModel::computeForce(ContactPoint& cp)
{
  cp.friction_flag = false; 
  /*!< force along normal to contact object */ 
  normalF_ = stiffness_ * cp.normal - damping_*cp.normvel; 
  if(cp.unilateral){
    for (unsigned int i = 0; i < 3; ++i) {
      // make sure the damping part does not attract a contact force with wrong sign
      if (normalF_(i) * macro_sign(cp.normal(i) < 0)) {
        normalF_(i) = 0.0;
      }
    }
  } 
  normalNorm_ = sqrt(normalF_.transpose()*normalF_);
  tangentF_ = stiffness_ * cp.tangent - damping_*cp.tanvel;
  tangentNorm_ = sqrt(tangentF_.transpose()*tangentF_);
  cp.f = normalF_; 
  //
  if (cp.unilateral && (tangentNorm_ > friction_coeff_*normalNorm_)){
    tangentDir_ = tangentF_/tangentNorm_; 
    cp.f += friction_coeff_*normalNorm_*tangentDir_; 
    // TODO: different friction coefficient along x,y will have to do 
    //       the update below in vector form  
    delAnchor_ = (tangentNorm_ - friction_coeff_*normalNorm_)/stiffness_(0,0);  
    cp.x_start -= delAnchor_ * tangentDir_; 
    cp.friction_flag = true;  
  } // friction cone violation
  else{
    cp.f += tangentF_;
  }
}

// --------------------------------------------------------------------------------------------------------// 

ContactObject::ContactObject(std::string name, ContactModel& contact_model):
    name_(name), contact_model_(&contact_model) { }


} // namespace consim
