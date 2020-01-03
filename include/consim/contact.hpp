#pragma once

#include <Eigen/Eigen>
#include <pinocchio/spatial/motion-dense.hpp>


namespace consim {

#define NO_CONTACT                       0
#define DAMPED_SPRING_STATIC_FRICTION    1
#define DAMPED_SPRING_VISCOUS_FRICTION   2
#define DAMPED_SPRING_LIMITED_REBOUND    3

// QUESTIONS:
// - Why is contact computation done in the frame of object? Any issue of doing
//   it in the world frame directly?
// - What's the purpose of special contact for behavior like F_ORTH etc?
//   SEE: https://git-amd.tuebingen.mpg.de/amd-clmc/locomotion-sl/blob/master/src/SL_objects.c#L1035

class Object;

struct ContactPoint {
  int frame_id;

  bool active;

  Object* optr;

  double     fraction_start;                   /*!< fraction relative to start point */
  double     fraction_end;                     /*!< fraction relative to end point */

  // \brief Position of contact point in world coordinate frame.
  Eigen::Vector3d x;

  // \brief Velocity of contact point in world coordinate frame.
  Eigen::Vector3d v;

  // \brief Position where the contact point touched the object the first time
  // in world coordinate frame.
  Eigen::Vector3d x_start;

  // \brief The normal of the contact surface in the world coordinate frame.
  Eigen::Vector3d contact_surface_normal;

  bool friction_flag;

  Eigen::Vector3d     normal;                 /*!< normal displacement vector */
  Eigen::Vector3d     normvel;                /*!< normal velocity vector */
  Eigen::Vector3d     tangent;                /*!< tangential displacement vector */
  Eigen::Vector3d     tanvel;                 /*!< tangential velocity vector */
  Eigen::Vector3d     viscvel;                /*!< velocity vector for viscous friction */
  Eigen::Vector3d     f;                      /*!< contact forces in world coordinates */
  
  //TODO: store local and world v, and dJv also in both frames
  Eigen::Vector6d v_local; 
  Eigen::Vector3d v_world;
  Eigen::Vector6d dJv_local;
  Eigen::Vector3d dJv_world;
};


// -----------------------------------------------------------------------------


class ContactModel {
public:
  virtual void computeForce(ContactPoint &cp) = 0;
};

class LinearPenaltyContactModel: public ContactModel {
public:
  /**
   * contact_parms[1] = normal spring coefficient
   * contact_parms[2] = normal damping coefficient
   * contact_parms[3] = static friction spring coefficient
   * contact_parms[4] = static friction damping spring coefficient
   * contact_parms[5] = static friction coefficient (friction cone)
   * contact_parms[6] = dynamic friction coefficient (proportional to normal force)
   */
  LinearPenaltyContactModel(
      double normal_spring_const, double normal_damping_coeff,
      double static_friction_spring_coeff, double static_friction_damping_spring_coeff,
      double static_friction_coeff, double dynamic_friction_coeff);

  void computeForce(ContactPoint& cp);


private:
  double normal_spring_const_;
  double normal_damping_coeff_;
  double static_friction_spring_coeff_;
  double static_friction_damping_spring_coeff_;
  double static_friction_coeff_;
  double dynamic_friction_coeff_;
  // add option for exponential integrator, resize and fill in constructor  
  Eigen::MatrixXd matK_;
  Eigen::MatrixXd matB_;
};


}
