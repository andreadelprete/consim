
#include "consim/object.hpp"

namespace consim {

bool FloorObject::checkCollision(ContactPoint &cp)
{
  // checks for penetration into the floor 
  if (cp.x(2) > 0.) {
    return false;
  }

  if (!cp.active) {
    cp.x_start = cp.x;
  }
  //
  return true;
}

}
