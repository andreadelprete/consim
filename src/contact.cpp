#include "consim/contact.hpp"
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <math.h>

namespace consim {

#define macro_sign(x) ((x > 0) - (x < 0))
#define sqr(x) (x * x)


ContactPoint::ContactPoint(const pinocchio::Model &model, std::string name, unsigned int frameId, unsigned int nv, bool isUnilateral): 
        model_(&model), name_(name), frame_id(frameId), unilateral(isUnilateral) {
          active = false; 
          f.fill(0);
          predictedF_.fill(0);
          predictedX_.fill(0);
          world_J_.resize(3, nv); world_J_.setZero();
          full_J_.resize(6, nv); full_J_.setZero();
          dJdt_.resize(6, nv); dJdt_.setZero();
           
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


void ContactPoint::secondOrderContactKinematics(pinocchio::Data &data, Eigen::VectorXd &v){
  // pinocchio::getFrameJacobianTimeVariation(*model_, data, frame_id, pinocchio::LOCAL, dJdt_);
  dJvlocal_ = pinocchio::getFrameAcceleration(*model_, data, frame_id); 
  // // std::cout<<"dJv acceleration component "<< dJvlocal_.linear() << std::endl;
  dJvlocal_.linear() += vlocal_.angular().cross(vlocal_.linear());
  // dJvlocal_.linear() = dJdt_.topRows<3>() * v; 
  // dJvlocal_.angular() = dJdt_.bottomRows<3>() * v; 
  dJv_ = frameSE3_.act(dJvlocal_).linear();

}


void ContactPoint::resestAnchorPoint(Eigen::VectorXd &x0){
  x_start = x0; 
  predictedX0_ = x0;  
}


// --------------------------------------------------------------------------------------------------------// 

LinearPenaltyContactModel::LinearPenaltyContactModel(Eigen::Vector3d &stiffness, Eigen::Vector3d &damping, double frictionCoeff){
  stiffness_=stiffness;
  damping_=damping; 
  friction_coeff_=frictionCoeff;
 }


void LinearPenaltyContactModel::computeForce(ContactPoint& cp)
{
  /*!< force along normal to contact object */ 
  normalF_ = stiffness_.cwiseProduct(cp.normal) - damping_.cwiseProduct(cp.normvel); 

  /*!< unilateral force, no pulling into contact object */ 
  if (cp.unilateral && normalF_.dot(cp.contactNormal_)<0){
    cp.f.fill(0);
  } 
  else{
    normalNorm_ = sqrt(normalF_.transpose()*normalF_);
    tangentF_ = stiffness_.cwiseProduct(cp.tangent) - damping_.cwiseProduct(cp.tanvel);
    tangentNorm_ = sqrt(tangentF_.transpose()*tangentF_);
    cp.f = normalF_; 
    //
    if (cp.unilateral && (tangentNorm_ > friction_coeff_*normalNorm_)){
      tangentDir_ = tangentF_/tangentNorm_; 
      cp.f += friction_coeff_*normalNorm_*tangentDir_; 
      // TODO: different friction coefficient along x,y will have to do 
      //       the update below in vector form  
      delAnchor_ = (tangentNorm_ - friction_coeff_*normalNorm_)/stiffness_(0);  
      cp.x_start -= delAnchor_ * tangentDir_;  
    } // friction cone violation
    else{
      cp.f += tangentF_;
    }
  }
}

// --------------------------------------------------------------------------------------------------------// 




} // namespace consim
