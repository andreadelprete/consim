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
          dJdt_.resize(3, nv); dJdt_.setZero();
          world_J_.resize(3, nv); world_J_.setZero();
          full_J_.resize(6, nv); full_J_.setZero();
        }

void ContactPoint::firstOrderContactKinematics(pinocchio::Model &model, pinocchio::Data &data){
  x = data.oMf[frame_id].translation(); 

  vlocal_ = pinocchio::getFrameVelocity(model, data, frame_id); 
  frameSE3_.rotation() = data.oMf[frame_id].rotation();
  v.noalias() = frameSE3_.rotation()*vlocal_.linear();

  pinocchio::getFrameJacobian(model, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, full_J_);
  world_J_ = full_J_.topRows<3>();
}


void ContactPoint::secondOrderContactKinematics(pinocchio::Model &model, pinocchio::Data &data){

  dJvlocal_ = pinocchio::getFrameAcceleration(model, data, frame_id); 
  dJvlocal_.linear() += vlocal_.angular().cross(vlocal_.linear());
  frameSE3_.rotation() = data.oMf[frame_id].rotation();
  dJv_ = frameSE3_.act(dJvlocal_).linear();

}

void ContactPoint::computeContactForce(){

}

// --------------------------------------------------------------------------------------------------------// 

LinearPenaltyContactModel::LinearPenaltyContactModel(Eigen::Matrix3d stiffness, Eigen::Matrix3d damping, double frictionCoeff):
stiffness_(stiffness), damping_(damping), friction_coeff_(frictionCoeff) {}



// void LinearPenaltyContactModel::computeForce(ContactPoint& cptr)
// {
//   unsigned int i;
//   double viscvel;
//   Eigen::Vector3d temp;

//   double tangent_force = 0.;
//   double normal_force = 0.;
//   for (i = 0; i < 3; ++i) {
//     // TODO: Write as Vector Operation
//     cptr.f(i) = normal_spring_const_ * cptr.normal(i) + normal_damping_coeff_ * cptr.normvel(i);

//     // make sure the damping part does not attract a contact force with wrong sign
//     if (cptr.unilateral && (cptr.f(i)) * macro_sign(cptr.normal(i)) < 0) {
//       cptr.f(i) = 0.0;
//     }

//     normal_force += sqr(cptr.f(i));
//   }
//   normal_force = sqrt(normal_force);

//   // project the spring force according to the information in the contact and object structures
//   // TODO: Need to support force projection from SL. Doing no projection is the
//   //       same as having "F_FULL" contact point.
//   //
//   // projectForce(cptr,optr);

//   // the force due to static friction, modeled as horizontal damper, again
//   // in object centered coordinates
//   tangent_force = 0;
//   viscvel = 1.e-10;
//   for (i = 0; i < 3; ++i) {
//     // TODO: Write as Vector Operation
//     temp(i) = -static_friction_spring_coeff_ * cptr.tangent(i) -
//         static_friction_damping_spring_coeff_ * cptr.tanvel(i);
//     tangent_force += sqr(temp(i));
//     viscvel += sqr(cptr.viscvel(i));
//   }
//   tangent_force = sqrt(tangent_force);
//   viscvel = sqrt(viscvel);

//   if(cptr.unilateral && (tangent_force > static_friction_coeff_ * normal_force || cptr.friction_flag)) 
//   {
//     /* If static friction too large -> spring breaks -> dynamic friction in
//       the direction of the viscvel vector; we also reset the x_start
//       vector such that static friction would be triggered appropriately,
//       i.e., when the viscvel becomes zero */
//     cptr.friction_flag = true;
//     for (i = 0; i < 3; ++i) {
//       // TODO: Write as Vector Operation
//       // TODO: Take x_start reset out of the loop ? anytime friction cone is violated, it is called 3 times
//       cptr.f(i) += -dynamic_friction_coeff_ * normal_force * cptr.viscvel(i)/viscvel;
//       if (viscvel < 0.01) {
//         cptr.friction_flag = false;
//         cptr.x_start = cptr.x;
//       }
//     }
//   } 
//   else {
//       // TODO: Write as Vector Operation
//       for (i = 0; i < 3; ++i) {
//         cptr.f(i) += temp(i);
//       }
//   }
// }

void LinearPenaltyContactModel::computeForce(ContactPoint& cptr)
{

}


} // namespace consim
