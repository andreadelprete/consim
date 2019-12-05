#include <Eigen/Eigen>

#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include "consim/object.hpp"
#include "consim/contact.hpp"

namespace consim {

class Simulator {
public:
  Simulator(float dt, int n_integration_steps, const pinocchio::Model& model, pinocchio::Data& data);

  const Contact& add_contact_point(int frame_id);

  const Contact& get_contact(int index);

  /**
   * Adds an object to the simulator for contact interaction checking.
   */
  void add_object(Object& obj);

  /**
   * Convenience method to perform a single dt timestep of the simulation. The
   * q and dq after the step are available form the sim.q and sim.dq property.
   * The acceloration during the last step is available from data.ddq;
   *
   * The jacobian, frames etc. of data is update after the final q/dq values
   * are computed. This allows to use the data object after calling step
   * without the need to re-run the computeXXX methods etc.
   */
  void step(const Eigen::VectorXd& tau);

  /**
   * Use this function to change the q and dq the next integration step()
   * starts from.
   *
   * If reset_contact_state is true, all contacts are set to inactive. Otherwise
   * the contact state is left untouched.
   */
  void reset_state(const Eigen::VectorXd& q, const Eigen::VectorXd& dq, bool reset_contact_state);

  void set_joint_friction(const Eigen::VectorXd& joint_friction);

  /**
   * TODO: The following fields should be private and there should be only
   * getter functions. However, jviereck didn't manage to get boost-python
   * to use use the get_q etc. functions for property lookups.
   */

  Eigen::VectorXd q_;
  Eigen::VectorXd dq_;
  Eigen::VectorXd dqMean_;
  Eigen::VectorXd tau_;



private:
  inline void contact_linear_jacobian_(int frame_id);

  /**
   * Checks the contact points for contacts with the objects. If there is a
   * contact, some contact properties like contact normal, velocity etc.
   * are computed.
   *
   * \returns Returns true if a contact was detected.
   */
  void check_contact_();

  /**
   * Usually called after detect_contact();
   *
   * Assumes the contact points are assigned to objects already etc.
   */
  void compute_contact_forces_and_torques_(const Eigen::VectorXd& dq);

  void compute_terms_and_contact_state_();

  const pinocchio::Model* model_;
  pinocchio::Data* data_;

  double dt_;
  int n_integration_steps_;
  std::vector<Contact*> contacts_;
  std::vector<Object*> objects_;
  // joint friction flag and values
  Eigen::VectorXd joint_friction_;
  bool joint_friction_flag_=0;

  Eigen::MatrixXd frame_Jc_;
  pinocchio::Data::Matrix6x J_;
};

}
