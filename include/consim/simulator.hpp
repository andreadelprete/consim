#include <Eigen/Eigen>

#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include "consim/object.hpp"
#include "consim/contact.hpp"
#include  "consim/dynamic_algebra.hpp"

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

      

      
      /**
       * Convenience method to perform a single dt timestep of the simulation. 
       * Computes q, dq, ddq, and contact forces for a single time step 
       * results are stored in sim.q, sim.dq, sim.ddq, sim.f 
       */

      virtual void step(const Eigen::VectorXd &tau)=0;

      Eigen::VectorXd get_q() const {return q_;};
      Eigen::VectorXd get_dq() const {return dq_;};
      Eigen::VectorXd get_ddq() const {return ddq_;};
      // Eigen::VectorXd get_f() const {return f_};

    
    protected:
      Eigen::VectorXd q_;  
      Eigen::VectorXd dq_;
      Eigen::VectorXd ddq_;
      // Eigen::VectorXd f_;
      Eigen::VectorXd dqMean_;
      Eigen::VectorXd tau_;
      int nc_=0;
      int nk_=0;
      int nactive_; // number of active contacts
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
     // TODO: might change interface depending on parameters required for the exponential simulator version 
      virtual void computeContactForces(const Eigen::VectorXd &dq)=0;

      inline void contactLinearJacobian(int frame_id);
 

      const pinocchio::Model *model_;
      pinocchio::Data *data_;

      double dt_;
      int n_integration_steps_;
  
      std::vector<ContactPoint *> contacts_;
      std::vector<Object *> objects_;

      Eigen::MatrixXd frame_Jc_;
      pinocchio::Data::Matrix6x J_;

      Eigen::VectorXd joint_friction_;
      bool joint_friction_flag_ = 0;

  }; // class AbstractSimulator

  class EulerSimulator : public AbstractSimulator
  {
    public: 
      EulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps); 
      ~EulerSimulator(){};

    /**
     * Explicit Euler first oder step 
    */

      void step(const Eigen::VectorXd &tau) override;

      Eigen::VectorXd getQ() const {return q_;};
      Eigen::VectorXd getDq() const {return dq_;}; 
    

    protected:
      const double sub_dt;
      void computeContactForces(const Eigen::VectorXd &dq) override;
      
      


  }; // class EulerSimulator

  class ExponentialSimulator : public AbstractSimulator
  {
    public:
      ExponentialSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, 
                          bool sparse=false, bool invertibleA=false); 
      ~ExponentialSimulator(){};
      void step(const Eigen::VectorXd &tau) override;

    protected:
      /**
       * call after adding all contact points 
       * 
       */
      void allocateData(); 
      /**
       * AbstractSimulator::computeContactState() must be called before  
       * calling ExponentialSimulator::computeContactForces()
       */
      void computeContactForces(const Eigen::VectorXd &dq) override; 
      void solveDenseExpSystem(); 
      void solveSparseExpSystem(); 

      bool sparse_; 
      bool invertibleA_;
      const double sub_dt;
      

      Eigen::VectorXd f_;  // total force 
      Eigen::MatrixXd Jc_; // contact jacobian for all contacts 
      Eigen::VectorXd p0_; // reference position for contact 
      Eigen::VectorXd p_; // current contact position 
      Eigen::VectorXd dp_; // contact velocity 
      Eigen::VectorXd xt_;  // containts p and dp for all active contact points 
      Eigen::VectorXd intxt_;
      Eigen::VectorXd int2xt_;

      void computeFrameAcceleration(int frame_id); 

      // contact acceleration components 
      Eigen::VectorXd dJv_; 
      Eigen::VectorXd a_;

      Eigen::VectorXd dJvilocal_; // per frame
      Eigen::VectorXd ailocal_;   // per frame
      Eigen::VectorXd dJvi_; // per frame 
      Eigen::VectorXd ai_; // per frame 
      // keep the stiffness/damping matrices fixed to the total size of contact points
      // worry about tracking index of Active sub-blocks later
      Eigen::MatrixXd K;
      Eigen::MatrixXd B;
      Eigen::MatrixXd D;
      Eigen::MatrixXd A; 
      


  }; // class ExponentialSimulator



} // namespace consim 
