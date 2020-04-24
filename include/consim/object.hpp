#pragma once

#include "consim/contact.hpp"

namespace consim {

class FloorObject: public ContactObject
{
  FloorObject(std::string name, ContactModel& contact_model): ContactObject(name, contact_model) {
    normal << 0., 0., 1.;
    tangentA << 1., 0., 0.;
    tangentB << 0., 1., 0.;
   };

  bool checkCollision(ContactPoint &cp); 
};

}

