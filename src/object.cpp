
#include "consim/object.hpp"

namespace consim {

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
  }

  // Compute the normal displacement and velocities.
  cp.normal = (cp.x_start - cp.x).dot(normal) * normal;
  cp.normvel = (-cp.v).dot(normal) * normal;

  // Compute the tangential offset and velocity by removing the normal component.
  // NOTE: The normal component has a different signs than the tangential one.
  cp.tangent = (cp.x - cp.x_start) + cp.normal;
  cp.tanvel = cp.v + cp.normvel;

  return true;
}

}
