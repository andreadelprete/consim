#include <Eigen/Eigen>

#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include "consim/object.hpp"
#include "consim/contact.hpp"
#include "consim/exponential_integrator.hpp"

namespace consim {

  class AbstractSimulator {
    public:
      AbstractSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps); 
      ~AbstractSimulator(){};

      /**
        * Defines a pinocchio frame as a contact point for contact interaction checking.
        * A contact is a struct containing all the contact information 
      */
      const ContactPoint &addContactPoint(int frame_id);

      /**
        * Returns the contact points reference 
      */

      const ContactPoint &getContact(int index);

      /**
        * Adds an object to the simulator for contact interaction checking.
      */
      void addObject(Object &obj);

      /**
       * Convenience method to perform a single dt timestep of the simulation. The
       * q and dq after the step are available form the sim.q and sim.dq property.
       * The acceloration during the last step is available from data.ddq;
       * 
       * The jacobian, frames etc. of data is update after the final q/dq values
       * are computed. This allows to use the data object after calling step
       * without the need to re-run the computeXXX methods etc.
       */

      void resetState(const Eigen::VectorXd &q, const Eigen::VectorXd &dq, bool reset_contact_state);

      void setJointFriction(const Eigen::VectorXd &joint_friction);

      Eigen::VectorXd get_q() const {return q_};
      Eigen::VectorXd get_dq() const {return dq_};
      Eigen::VectorXd get_ddq() const {return ddq_};
      // Eigen::VectorXd get_f() const {return f_};

      /**
       * Convenience method to perform a single dt timestep of the simulation. 
       * Computes q, dq, ddq, and contact forces for a single time step 
       * results are stored in sim.q, sim.dq, sim.ddq, sim.f 
       */

      virtual void step(const Eigen::VectorXd &tau)=0;
      

    
    protected:
      /**
        * loops over contact points, checks active contacts and sets reference contact positions 
      */
      void checkContact();
      /**
      * computes all relative dynamic and kinematic terms, then checks for the contacts  
      */
      void computeContactState();

      /** 
       * 
      */
      virtual void computeContactForces()=0;

      const pinocchio::Model *model_;
      pinocchio::Data *data_;

      double dt_;
      int n_integration_steps_;
  
      Eigen::VectorXd q_;  
      Eigen::VectorXd dq_;
      Eigen::VectorXd ddq_;
      // Eigen::VectorXd f_;
      Eigen::VectorXd dqMean_;
      Eigen::VectorXd tau_;

      std::vector<ContactPoint *> contacts_;
      std::vector<Object *> objects_;

      Eigen::VectorXd joint_friction_;
      bool joint_friction_flag_ = 0;

  } // class AbstractSimulator

  class EulerSimulator : public AbstractSimulator
  {
    public: 
      EulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps); 
      ~AbstractSimulator(){};

      void step(const Eigen::VectorXd &tau);
    

    protected:
      inline void contactLinearJacobian(int frame_id);
      void computeContactForces(const Eigen::VectorXd &dq);
      Eigen::MatrixXd frame_Jc_;
      pinocchio::Data::Matrix6x J_;


  } // class EulerSimulator

  class ExponentialSimulator : public AbstractSimulator
  {

  } // class ExponentialSimulator

//   class Simulator
//   {
//   public:
//     Simulator(float dt, int n_integration_steps, const pinocchio::Model &model, pinocchio::Data &data, bool expo_integrator = false,
//               bool sparse_solver = false, bool invertibleA = false);

//     const ContactPoint &addContactPoint(int frame_id);

//     const ContactPoint &getContact(int index);

//     /**
//    * Adds an object to the simulator for contact interaction checking.
//    */
//     void addObject(Object &obj);

//     /**
//    * Convenience method to perform a single dt timestep of the simulation. The
//    * q and dq after the step are available form the sim.q and sim.dq property.
//    * The acceloration during the last step is available from data.ddq;
//    * 
//    * The jacobian, frames etc. of data is update after the final q/dq values
//    * are computed. This allows to use the data object after calling step
//    * without the need to re-run the computeXXX methods etc.
//    */
//     void step(const Eigen::VectorXd &tau);

    /**
   * Use this function to change the q and dq the next integration step()
   * starts from.
   *
   * If reset_contact_state is true, all contacts are set to inactive. Otherwise
   * the contact state is left untouched.
   */
//     void resetState(const Eigen::VectorXd &q, const Eigen::VectorXd &dq, bool reset_contact_state);

//     void setJointFriction(const Eigen::VectorXd &joint_friction);

//     /**
//    * TODO: The following fields should be private and there should be only
//    * getter functions. However, jviereck didn't manage to get boost-python
//    * to use use the get_q etc. functions for property lookups.
//    */

//     Eigen::VectorXd q_;
//     Eigen::VectorXd dq_;
//     Eigen::VectorXd dqMean_;
//     Eigen::VectorXd tau_;
//     // enable exponential integration

//   private:
//     inline void contactLinearJacobian_(int frame_id);

//     /**
//    * Checks the contact points for contacts with the objects. If there is a
//    * contact, some contact properties like contact normal, velocity etc.
//    * are computed.
//    *
//    * \returns Returns true if a contact was detected.
//    */
//     void checkContact_();

//     /**
//    * Usually called after detect_contact();
//    *
//    * Assumes the contact points are assigned to objects already etc.
//    */
//     void computeContactForces(const Eigen::VectorXd &dq);

//     void computeContactState_();

//     const pinocchio::Model *model_;
//     pinocchio::Data *data_;

//     double dt_;
//     int n_integration_steps_;
//     std::vector<ContactPoint *> contacts_;
//     std::vector<Object *> objects_;
//     // joint friction flag and values
//     Eigen::VectorXd joint_friction_;
//     bool joint_friction_flag_ = 0;

//     Eigen::MatrixXd frame_Jc_;
//     pinocchio::Data::Matrix6x J_;
// };

} // namespace consim 
