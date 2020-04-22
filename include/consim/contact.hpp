#pragma once

#include <Eigen/Eigen>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/motion.hpp>

// #include "consim/object.hpp"



namespace consim {

class Object;

class ContactPoint {

  // \brief for now will keep everything public instead of get/set methods 

  public:
    ContactPoint(std::string name, unsigned int frameId, unsigned int nv, bool isUnilateral=true); 
    ~ContactPoint() {};
   
    void firstOrderContactKinematics(pinocchio::Model &model, pinocchio::Data &data);  /*!< computes world_J_ */ 
    void secondOrderContactKinematics(pinocchio::Model &model, pinocchio::Data &data); /*!< computes dJv_ */
    void computeContactForce();  /*!< calls contact model from object pointer to compute the force  */
    
    std::string name_;
    unsigned int frame_id;

    bool active;
    bool unilateral;      /*!< true if the contact is unilateral, false if bilateral */
    bool friction_flag;   /*!< true if force is outside of friction cone  */

    Object* optr;         /*!< pointer to current contact object, changes with each new contact switch */  

    Eigen::Vector3d     x_start;                /*!< anchor point for visco-elastic contact models  */
    Eigen::Vector3d     x;                      /*!< contact point position in world frame */
    Eigen::Vector3d     v;                      /*!< contact point translation velocity in world frame */
    Eigen::Vector3d     normal;                 /*!< normal displacement vector */
    Eigen::Vector3d     normvel;                /*!< normal velocity vector */
    Eigen::Vector3d     tangent;                /*!< tangential displacement vector */
    Eigen::Vector3d     tanvel;                 /*!< tangential velocity vector */
    Eigen::Vector3d     f;                      /*!< contact forces in world coordinates */
    Eigen::Vector3d     dJv_; 
    
    Eigen::MatrixXd     dJdt_; 
    Eigen::MatrixXd     world_J_; 
    Eigen::MatrixXd     full_J_;  

    // velocity transformation from local to world 
    pinocchio::Motion vlocal_ = pinocchio::Motion::Zero();
    pinocchio::Motion dJvlocal_ = pinocchio::Motion::Zero(); 
    pinocchio::SE3 frameSE3_ = pinocchio::SE3::Identity(); 



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

  const Eigen::Matrix3d stiffness_;  
  const Eigen::Matrix3d damping_; 
  const double friction_coeff_;

};

}
