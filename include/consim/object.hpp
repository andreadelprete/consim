#pragma once

// #include "consim/object.fwd.hpp"
// #include "consim/contact.fwd.hpp"
#include "consim/contact.hpp"

namespace consim {

/**
 * base class for ContactObject is defined here 
 * object can modify contact point and vice-versa
 * one had to be included before the other 
 * **/  

class ContactObject {
  public:
  // initialize with a specific contact model, could be viscoElastic, rigid .. etc 
  // all model specific parameters will be stored in the model itself 
    ContactObject(const std::string & name, ContactModel& contact_model);
    ~ContactObject(){};
    
    /** CheckCollision()
     * Checks if a given contact point is in collision with the object
     * A contact point can be in contact with one object only
     **/  
    virtual bool checkCollision(ContactPoint &cp) = 0;
    virtual void computePenetration(ContactPoint &cp) = 0;
    const std::string & getName() const { return name_; }
    ContactModel* contact_model_;
     
    std::string name_;
    
};

// -----------------------------------------------------------------------------

class FloorObject: public ContactObject
{
  public:
  FloorObject(const std::string & name, ContactModel& contact_model)
  : ContactObject(name, contact_model)
  { };
  ~FloorObject(){};

  bool checkCollision(ContactPoint &cp) override; 
  void computePenetration(ContactPoint &cp) override;
};

}

