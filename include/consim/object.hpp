#pragma once

#include "consim/contact.hpp"

namespace consim {

class Object {
public:
  Object(std::string name, ContactModel& contact_model);

  virtual bool checkContact(ContactPoint &cp) = 0;
  virtual void contactModel(ContactPoint &cp) = 0;

protected:
  std::string name_;
  ContactModel* contact_model_;
};

class FloorObject: public Object
{
public:
  FloorObject(std::string name, ContactModel& contact_model): Object(name, contact_model) { };

  bool checkContact(ContactPoint &cp);
  void contactModel(ContactPoint &cp);
};

}
