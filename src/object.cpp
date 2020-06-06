
#include "consim/object.hpp"

namespace consim {

ContactObject::ContactObject(const std::string & name, ContactModel& contact_model):
    name_(name), contact_model_(&contact_model) { }

// -------------------------------------------------------------------------------

bool FloorObject::checkCollision(ContactPoint &cp)
{
  // checks for penetration into the floor 
  if (cp.x(2) > 0.) {
    return false;
  }

  if (!cp.active) {
    cp.x_anchor = cp.x;
    cp.v_anchor.setZero();
    cp.predictedX0_ = cp.x;
    cp.contactNormal_ << 0.,0.,1.; 
    cp.contactTangentA_ << 0.,1.,0.;
    cp.contactTangentB_ << 1.,0.,0.;
  }
  //
  return true;
}

void FloorObject::computePenetration(ContactPoint &cp){
  /** compute displacement relative to contact object
   * delta_x: relative penetration 
   * normal: penetration along normal to contact object
   * tangent: penetration along tangent to contact object
   * normalvel: velocity along normal to contact object
   * tanvel: velocity along tangent to contact object
   * */ 
  cp.delta_x = cp.x_anchor - cp.x; 
  cp.normal = cp.delta_x.dot(cp.contactNormal_) * cp.contactNormal_; 
  cp.tangent = cp.delta_x - cp.normal; 
  cp.normvel = (cp.v).dot(cp.contactNormal_) * cp.contactNormal_; 
  cp.tanvel = cp.v - cp.normvel; 
}

}
