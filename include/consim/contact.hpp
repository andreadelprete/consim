#pragma once

#include <Eigen/Eigen>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/motion.hpp>

// #include "consim/object.hpp"



namespace consim {

class ContactObject; /*!< ContactObject and ContactPoint depend on each other, one has to be declared */

class ContactPoint {

  // \brief for now will keep everything public instead of get/set methods 

  public:
    ContactPoint(std::string name, unsigned int frameId, unsigned int nv, bool isUnilateral=true); 
    ~ContactPoint() {};

    void updatePosition(const pinocchio::Model &model, pinocchio::Data &data);  /*!< updates cartesian position */ 
    void firstOrderContactKinematics(const pinocchio::Model &model, pinocchio::Data &data);  /*!< computes c, world_J_ and relative penetration */ 
    void secondOrderContactKinematics(const pinocchio::Model &model, pinocchio::Data &data); /*!< computes dJv_ */
    void computeContactForce();  /*!< calls contact model from object pointer to compute the force  */
    
    std::string name_;
    unsigned int frame_id;

    bool active;
    bool unilateral;      /*!< true if the contact is unilateral, false if bilateral */
    bool friction_flag;   /*!< true if force is outside of friction cone  */

    ContactObject* optr;         /*!< pointer to current contact object, changes with each new contact switch */  

    Eigen::Vector3d     x_start;                /*!< anchor point for visco-elastic contact models  */
    Eigen::Vector3d     x;                      /*!< contact point position in world frame */
    Eigen::Vector3d     v;                      /*!< contact point translation velocity in world frame */
    Eigen::Vector3d     dJv_; 

    Eigen::MatrixXd     dJdt_; 
    Eigen::MatrixXd     world_J_; 
    Eigen::MatrixXd     full_J_;  

    // velocity transformation from local to world 
    pinocchio::Motion vlocal_ = pinocchio::Motion::Zero();
    pinocchio::Motion dJvlocal_ = pinocchio::Motion::Zero(); 
    pinocchio::SE3 frameSE3_ = pinocchio::SE3::Identity(); 

    // relative to contact object 
    Eigen::Vector3d     dx;                      /*!< penetration into the object */
    Eigen::Vector3d     normal;                 /*!< normal displacement vector */
    Eigen::Vector3d     normvel;                /*!< normal velocity vector */
    Eigen::Vector3d     tangent;                /*!< tangential displacement vector */
    Eigen::Vector3d     tanvel;                 /*!< tangential velocity vector */
    Eigen::Vector3d     f;                      /*!< contact forces in world coordinates */

}; 



// -----------------------------------------------------------------------------


class ContactModel {
public:
  virtual void computeForce(ContactPoint &cp) = 0;
};

class LinearPenaltyContactModel: public ContactModel {
public:
  LinearPenaltyContactModel(Eigen::Matrix3d stiffness, Eigen::Matrix3d damping, double frictionCoeff);    
  
  void computeForce(ContactPoint& cp) override;

  Eigen::Matrix3d stiffness_; 
  Eigen::Matrix3d invStiffness_;  
  Eigen::Matrix3d damping_; 
  const double friction_coeff_;

  Eigen::Vector3d normalF_;
  Eigen::Vector3d tangentF_; 
  Eigen::Vector3d tangentDir_; 
  double delAnchor_; 
  double normalNorm_; 
  double tangentNorm_; 
  
};

// -----------------------------------------------------------------------------

/**
 * base class for ContactObject is defined here 
 * object can modify contact point and vice-versa
 * one had to be included before the other 
 * **/  

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
    const std::string & getName() const { return name_; }
    ContactModel* contact_model_;
    
    // \brief The normal of the contact surface in the world coordinate frame.
    Eigen::Vector3d contact_normal;
    // \brief basis of the tangent to thhe contact surface in thhe world coordinate frame 
    Eigen::Vector3d contact_tangentA;
    Eigen::Vector3d contact_tangentB;
     
    std::string name_;
    
};


}
