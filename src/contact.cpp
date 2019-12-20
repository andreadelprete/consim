#include "consim/contact.hpp"
#include <math.h>

namespace consim {

LinearPenaltyContactModel::LinearPenaltyContactModel(
    double normal_spring_const, double normal_damping_coeff,
    double static_friction_spring_coeff, double static_friction_damping_spring_coeff,
    double static_friction_coeff, double dynamic_friction_coeff)
{
  normal_spring_const_ = normal_spring_const;
  normal_damping_coeff_ = normal_damping_coeff;
  static_friction_spring_coeff_ = static_friction_spring_coeff;
  static_friction_damping_spring_coeff_ = static_friction_damping_spring_coeff;
  static_friction_coeff_ = static_friction_coeff;
  dynamic_friction_coeff_ = dynamic_friction_coeff;
}

#define macro_sign(x) ((x > 0) - (x < 0))
#define sqr(x) (x * x)

/**
 * Based on the DAMPED_SPRING_STATIC_FRICTION contact model in SL.
 *
 * note: the contact_parms arrays has the following elements:
 *       contact_parms[1] = normal spring coefficient
 *       contact_parms[2] = normal damping coefficient
 *       contact_parms[3] = static friction spring coefficient
 *       contact_parms[4] = static friction damping spring coefficient
 *       contact_parms[5] = static friction coefficient (friction cone)
 *       contact_parms[6] = dynamic friction coefficient (proportional to normal force)
 *
 * https://git-amd.tuebingen.mpg.de/amd-clmc/locomotion-sl/blob/master/src/SL_objects.c#L1372-1430
 */
void LinearPenaltyContactModel::compute_force(Contact &cptr)
{
  int i;
  double viscvel;
  Eigen::Vector3d temp;

  double tangent_force = 0.;
  double normal_force = 0.;
  for (int i = 0; i < 3; ++i) {
    cptr.f(i) = normal_spring_const_ * cptr.normal(i) + normal_damping_coeff_ * cptr.normvel(i);

    // make sure the damping part does not attract a contact force with wrong sign
    if (macro_sign(cptr.f(i)) * macro_sign(cptr.normal(i)) < 0) {
      cptr.f(i) = 0.0;
    }

    normal_force += sqr(cptr.f(i));
  }
  normal_force = sqrt(normal_force);

  // project the spring force according to the information in the contact and object structures
  // TODO: Need to support force projection from SL. Doing no projection is the
  //       same as having "F_FULL" contact point.
  //
  // projectForce(cptr,optr);

  // the force due to static friction, modeled as horizontal damper, again
  // in object centered coordinates
  tangent_force = 0;
  viscvel = 1.e-10;
  for (i = 0; i < 3; ++i) {
    temp(i) = -static_friction_spring_coeff_ * cptr.tangent(i) -
        static_friction_damping_spring_coeff_ * cptr.tanvel(i);
    tangent_force += sqr(temp(i));
    viscvel += sqr(cptr.viscvel(i));
  }
  tangent_force = sqrt(tangent_force);
  viscvel = sqrt(viscvel);

  /* If static friction too large -> spring breaks -> dynamic friction in
     the direction of the viscvel vector; we also reset the x_start
     vector such that static friction would be triggered appropriately,
     i.e., when the viscvel becomes zero */
  if (tangent_force > static_friction_coeff_ * normal_force || cptr.friction_flag) {
    cptr.friction_flag = true;
    for (i = 0; i < 3; ++i) {
      cptr.f(i) += -dynamic_friction_coeff_ * normal_force * cptr.viscvel(i)/viscvel;
      if (viscvel < 0.01) {
        cptr.friction_flag = false;
        cptr.x_start = cptr.x;
      }
    }
  } else {
    for (i = 0; i < 3; ++i) {
      cptr.f(i) += temp(i);
    }
  }
}


}
