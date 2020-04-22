#pragma once

#include "consim/contact.hpp"

namespace consim {

// class Object {
// public:
//   Object(std::string name, ContactModel& contact_model);

//   virtual bool checkContact(ContactPoint &cp) = 0;
//   virtual void contactModel(ContactPoint &cp) = 0;

//   double getNormalStiffness() const { return contact_model_->getNormalStiffness(); };
//   double getNormalDamping() const { return contact_model_->getNormalDamping(); };
//   double getTangentialStiffness() const { return contact_model_->getTangentialStiffness(); };
//   double getTangentialDamping() const { return contact_model_->getTangentialDamping(); };
//   double getFrictionCoefficient() const { return contact_model_->getFrictionCoefficient(); };
//   const std::string & getName() const { return name_; }

// protected:
//   std::string name_;
//   ContactModel* contact_model_;
// };

// class FloorObject: public Object
// {
// public:
//   FloorObject(std::string name, ContactModel& contact_model): Object(name, contact_model) { };

//   bool checkContact(ContactPoint &cp);
//   void contactModel(ContactPoint &cp);
// };

// -----------------------------------------------------------------------------

class ContactObject {
  public:
  // initialize with a specific contact model, could be viscoElastic, rigid .. etc 
  // all model specific parameters will be stored in the model itself 
    ContactObject(std::string name, ContactModel& contact_model);
    ~ContactObject(){};
    
    /** CheckCollision()
     * Checks if a given contact point is in collision with the object
     * Computes contact point kinematics relative to contact object 
     * and updates it in the contact point class 
     * A contact point can be in contact with one object only
     **/  
    virtual bool checkCollision(ContactPoint &cp) = 0;

    virtual void contactModel(ContactPoint &cp) = 0;
    const std::string & getName() const { return name_; }

  protected:
    // \brief The normal of the contact surface in the world coordinate frame.
    Eigen::Vector3d normal;
    // \brief basis of the tangent to thhe contact surface in thhe world coordinate frame 
    Eigen::Vector3d tangentA;
    Eigen::Vector3d tangentB;
  
    std::string name_;
    ContactModel* contact_model_;
}

class FloorObject: public ContactObject
{
  FloorObject(std::string name, ContactModel& contact_model): Object(name, contact_model) {
    normal << 0., 0., 1.;
    tangentA << 1., 0., 0.;
    tangentB << 0., 1., 0.;
   };

  bool checkCollision(ContactPoint &cp);
  void contactModel(ContactPoint &cp);
}

}

