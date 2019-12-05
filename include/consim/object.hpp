#pragma once

#include "consim/contact.hpp"

namespace consim {

class Object {
public:
  Object(std::string name, ContactModel& contact_model);

  virtual bool check_contact(Contact& cp) = 0;
  virtual void contact_model(Contact& cp) = 0;

protected:
  std::string name_;
  ContactModel* contact_model_;
};

class FloorObject: public Object
{
public:
  FloorObject(std::string name, ContactModel& contact_model): Object(name, contact_model) { };

  bool check_contact(Contact& cp);
  void contact_model(Contact& cp);
};

}
