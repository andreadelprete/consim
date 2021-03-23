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

#include <Eigen/Eigen>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/motion.hpp>

// #include "consim/contact.fwd.hpp"
// #include "consim/object.fwd.hpp"



namespace consim {

class ContactObject; 

class ContactPoint {

  // \brief for now will keep everything public instead of get/set methods 

  public:
    ContactPoint(const pinocchio::Model &model, const std::string & name, 
    unsigned int frameId, unsigned int nv, bool isUnilateral=true); 
    ~ContactPoint() {};

    void updatePosition(pinocchio::Data &data);  /*!< updates cartesian position */ 
    void firstOrderContactKinematics(pinocchio::Data &data);  /*!< computes c, world_J_ and relative penetration */ 
    void secondOrderContactKinematics(pinocchio::Data &data); /*!< computes dJv_ */
    void resetAnchorPoint(const Eigen::Vector3d &p0, bool slipping);  /*!< resets the anchor point position and velocity */
    void projectForceInCone(Eigen::Vector3d &f);
    
    const pinocchio::Model *model_;
    std::string name_;
    unsigned int frame_id;

    bool active;            /*!< true if the point is in collision with the environment, false otherwise */
    bool slipping;          /*!< true if the contact is slipping, false otherwise */
    bool unilateral;      /*!< true if the contact is unilateral, false if bilateral */

    ContactObject* optr;         /*!< pointer to current contact object, changes with each new contact switch */  

    Eigen::Vector3d     x_anchor;               /*!< anchor point for visco-elastic contact models  */
    Eigen::Vector3d     v_anchor;               /*!< anchor point velocity for visco-elastic contact models  */
    Eigen::Vector3d     x;                      /*!< contact point position in world frame */
    Eigen::Vector3d     v;                      /*!< contact point translation velocity in world frame */
    Eigen::Vector3d     dJv_; 

    Eigen::MatrixXd     world_J_; 
    Eigen::MatrixXd     full_J_;  

    // velocity transformation from local to world 
    pinocchio::Motion vlocal_ = pinocchio::Motion::Zero();
    pinocchio::Motion dJvlocal_ = pinocchio::Motion::Zero(); 
    pinocchio::SE3 frameSE3_ = pinocchio::SE3::Identity(); 

    // relative to contact object 
    Eigen::Vector3d     delta_x;                      /*!< penetration into the object */
    Eigen::Vector3d     normal;                 /*!< normal displacement vector */
    Eigen::Vector3d     normvel;                /*!< normal velocity vector */
    Eigen::Vector3d     tangent;                /*!< tangential displacement vector */
    Eigen::Vector3d     tanvel;                 /*!< tangential velocity vector */
    Eigen::Vector3d     f;                      /*!< contact force in world coordinates */
    Eigen::Vector3d     f_avg;                  /*!< average contact force during time step */
    Eigen::Vector3d     f_avg2;                 /*!< average of average contact force during time step */
    Eigen::Vector3d     f_prj;                  /*!< projection of f_avg in friction cone */
    Eigen::Vector3d     f_prj2;                 /*!< projection of f_avg2 in friction cone */
    Eigen::Vector3d     predictedF_;            /*!< contact forces predicted through exponential integration */
    Eigen::Vector3d     predictedX_;
    Eigen::Vector3d     predictedV_;
    Eigen::Vector3d     predictedX0_;

    Eigen::Vector3d    contactNormal_;
    Eigen::Vector3d    contactTangentA_;
    Eigen::Vector3d    contactTangentB_; 

}; 


// -----------------------------------------------------------------------------


class ContactModel {
public:
  ContactModel(){};
  ~ContactModel(){};
  
  // COmpute contact force and updates the contact point state
  virtual void computeForce(ContactPoint &cp) = 0;
  
  // Compute contact force without updating the contact point state
  virtual void computeForceNoUpdate(const ContactPoint &cp, Eigen::Vector3d& f) = 0;
  
  virtual void projectForceInCone(Eigen::Vector3d &f, ContactPoint& cp) = 0;
  
  Eigen::Vector3d  stiffness_; 
  Eigen::Vector3d  stiffnessInverse_; 
  Eigen::Vector3d  damping_; 
  double friction_coeff_;
};

class LinearPenaltyContactModel: public ContactModel {
public:
  LinearPenaltyContactModel(Eigen::Vector3d &stiffness, Eigen::Vector3d &damping, double frictionCoeff);    
  
  void computeForce(ContactPoint& cp) override;
  void computeForceNoUpdate(const ContactPoint &cp, Eigen::Vector3d& f) override;
  void projectForceInCone(Eigen::Vector3d &f, ContactPoint& cp) override;

  Eigen::Vector3d normalF_;
  Eigen::Vector3d tangentF_; 
  Eigen::Vector3d tangentDir_; 
  double delAnchor_; 
  double normalNorm_; 
  double tangentNorm_; 
  
};

}
