#pragma once

#include "consim/contact.hpp"

namespace consim {

class Object {
public:
  Object(std::string name, ContactModel& contact_model);

  virtual bool checkContact(ContactPoint &cp) = 0;
  virtual void contactModel(ContactPoint &cp) = 0;

  double getNormalStiffness() const { return contact_model_->getNormalStiffness(); };
  double getNormalDamping() const { return contact_model_->getNormalDamping(); };
  double getTangentialStiffness() const { return contact_model_->getTangentialStiffness(); };
  double getTangentialDamping() const { return contact_model_->getTangentialDamping(); };
  double getFrictionCoefficient() const { return contact_model_->getFrictionCoefficient(); };
  const std::string & getName() const { return name_; }

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

