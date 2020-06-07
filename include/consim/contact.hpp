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
    
    std::string name_;
    const pinocchio::Model *model_;
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
    Eigen::MatrixXd     dJdt_;  

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
    Eigen::Vector3d     f;                      /*!< contact forces in world coordinates */
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
  virtual void computeForce(ContactPoint &cp) = 0;
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
  void projectForceInCone(Eigen::Vector3d &f, ContactPoint& cp);

  Eigen::Vector3d normalF_;
  Eigen::Vector3d tangentF_; 
  Eigen::Vector3d tangentDir_; 
  double delAnchor_; 
  double normalNorm_; 
  double tangentNorm_; 
  
};

}
