#pragma once
#include <Eigen/Geometry>
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


// -----------------------------------------------------------------------------

class HalfPlaneObject: public ContactObject
{
  public:
  HalfPlaneObject(const std::string & name, ContactModel& contact_model, double alpha);
  ~HalfPlaneObject(){};

  bool checkCollision(ContactPoint &cp) override; 
  void computePenetration(ContactPoint &cp) override;

  private:
  const double angle_; 
  Eigen::Vector3d planeNormal_; 
  Eigen::Vector3d planeTangentA_; 
  Eigen::Vector3d planeTangentB_; 
  Eigen::Affine3d t_; 

  double plane_offset_; 
  double distance_; 

  void computeDistance(ContactPoint &cp);

};


}

