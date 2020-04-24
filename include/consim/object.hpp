#pragma once

#include "consim/contact.hpp"

namespace consim {

class FloorObject: public ContactObject
{
  public:
  FloorObject(std::string name, ContactModel& contact_model): ContactObject(name, contact_model) {
    contact_tangentA << 1., 0., 0.;
    contact_tangentB << 0., 1., 0.;
    contact_normal << 0., 0., 1.;
   };
   ~FloorObject(){};

  bool checkCollision(ContactPoint &cp); 
};

}

