#pragma once

#include <Eigen/Eigen>

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

struct Contact {
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

};


// -----------------------------------------------------------------------------


class ContactModel {
public:
  virtual void compute_force(Contact& cp) = 0;

};

class DampedSpringStaticFrictionContactModel: public ContactModel {
public:
  /**
   * contact_parms[1] = normal spring coefficient
   * contact_parms[2] = normal damping coefficient
   * contact_parms[3] = static friction spring coefficient
   * contact_parms[4] = static friction damping spring coefficient
   * contact_parms[5] = static friction coefficient (friction cone)
   * contact_parms[6] = dynamic friction coefficient (proportional to normal force)
   */
  DampedSpringStaticFrictionContactModel(
      double normal_spring_const, double normal_damping_coeff,
      double static_friction_spring_coeff, double static_friction_damping_spring_coeff,
      double static_friction_coeff, double dynamic_friction_coeff);


  void compute_force(Contact& cp);

private:
  double normal_spring_const_;
  double normal_damping_coeff_;
  double static_friction_spring_coeff_;
  double static_friction_damping_spring_coeff_;
  double static_friction_coeff_;
  double dynamic_friction_coeff_;
};

class NonlinearSpringDamperContactModel: public ContactModel{
public:
  /*
  */
  NonlinearSpringDamperContactModel(
    double spring_stiffness_coeff, double spring_damping_coeff,
    double static_friction_coeff, double dynamic_friction_coeff,
    double maximum_penetration, bool enable_friction_cone);

  void compute_force(Contact& cp);


private:
  double spring_stiffness_coeff_;
  double spring_damping_coeff_;
  double static_friction_coeff_;
  double dynamic_friction_coeff_;
  double maximum_penetration_;
  bool enable_friction_cone_;
  const double pi_ = std::atan(1.0)*4;

};

}
