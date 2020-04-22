
#include "consim/object.hpp"

namespace consim {

Object::Object(std::string name, ContactModel& contact_model):
    name_(name), contact_model_(&contact_model) { }

bool FloorObject::checkCollision(ContactPoint &cp)
{
  // if(!cp.unilateral){
  //   return true;
  // }

  // checks for penetration into the floor 
  if (cp.x(2) > 0.) {
    return false;
  }

  if (!cp.active) {
    cp.x_start = cp.x;
    // update basis only at switch  
    cp.contact_surface_normal = normal;
    cp.contact_surface_basisA = tangentA;
    cp.contact_surface_basisB = tangentB;
  }

  // Compute the normal displacement and velocities.
  cp.normal = (cp.x_start - cp.x).dot(cp.contact_surface_normal) * cp.contact_surface_normal;
  cp.normvel = (-cp.v).dot(cp.contact_surface_normal) * cp.contact_surface_normal;

  // Compute the tangential offset and velocity by removing the normal component.
  // NOTE: The normal component has a different signs than the tangential one.
  cp.tangent = (cp.x - cp.x_start) + cp.normal;
  cp.tanvel = cp.v + cp.normvel;

  return true;
}

void FloorObject::contactModel(ContactPoint &cp)
{
  // Compute the normal displacement and velocities.
  // TODO: Move this into a separate method (of the contact object)?
  cp.normal = (cp.x_start - cp.x).dot(cp.contact_surface_normal) * cp.contact_surface_normal;
  cp.normvel = (-cp.v).dot(cp.contact_surface_normal) * cp.contact_surface_normal;

  // Compute the tangential offset and velocity by removing the normal component.
  // NOTE: The normal component has a different signs than the tangential one.
  cp.tangent = (cp.x - cp.x_start) + cp.normal;
  cp.tanvel = cp.v + cp.normvel;

  // TODO: Figure out if the viscvel is always the same as the tanvel for each
  //       shape.
  cp.viscvel = cp.tanvel;

  contact_model_->computeForce(cp); // call using this in the contact class 
}

}
