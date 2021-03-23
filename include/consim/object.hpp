//
//  Copyright (c) 2020-2021 UNITN, NYU
//
//  This file is part of consim
//  consim is free software: you can redistribute it
//  and/or modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation, either version
//  3 of the License, or (at your option) any later version.
//  consim is distributed in the hope that it will be
//  useful, but WITHOUT ANY WARRANTY; without even the implied warranty
//  of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
//  General Lesser Public License for more details. You should have
//  received a copy of the GNU Lesser General Public License along with
//  consim If not, see
//  <http://www.gnu.org/licenses/>.

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

