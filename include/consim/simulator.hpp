#pragma once

#include <Eigen/Eigen>
#include <Eigen/Cholesky>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/motion.hpp>

#define PROFILERYES
#include <MatrixExponential.hpp>
#include <LDSUtility.hpp>
#include <MatExpIntegral.hpp>

#include "consim/object.hpp"
#include "consim/contact.hpp"
#include "eiquadprog/eiquadprog-fast.hpp"

#define CONSIM_PROFILER
#ifndef CONSIM_PROFILER
#define CONSIM_START_PROFILER(name)
#define CONSIM_STOP_PROFILER(name)
#else
#define CONSIM_START_PROFILER(name) getProfiler().start(name)
#define CONSIM_STOP_PROFILER(name) getProfiler().stop(name)
#endif




namespace consim {
  enum EulerIntegrationType{ EXPLICIT=0, SEMI_IMPLICIT=1, CLASSIC_EXPLICIT=2};

  class AbstractSimulator {
    public:
      AbstractSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD, EulerIntegrationType type); 
      ~AbstractSimulator(){};

      /**
        * Defines a pinocchio frame as a contact point for contact interaction checking.
        * A contact is a struct containing all the contact information 
      */
      const ContactPoint &addContactPoint(const std::string & name, int frame_id, bool unilateral);

      /**
        * Returns the contact points reference 
      */

      const ContactPoint &getContact(const std::string & name);

      /**
       * checks if contact is active 
       * if active it updates p0 for the specified contact and returns true
       * can update all contact forces
       * can be called after sim.step to update contact forces for next simulation step 
       **/ 
      bool resetContactAnchorPoint(const std::string & name, const Eigen::Vector3d &p0, bool updateContactForces, bool slipping);

      /**
        * Adds an object to the simulator for contact interaction checking.
      */
      void addObject(ContactObject &obj);


      /**
       * Convenience method to perform a single dt timestep of the simulation. The
       * q and dq after the step are available form the sim.q and sim.v property.
       * The acceloration during the last step is available from data.dv;
       * 
       * The jacobian, frames etc. of data is update after the final q/v values
       * are computed. This allows to use the data object after calling step
       * without the need to re-run the computeXXX methods etc.
       */

      void resetState(const Eigen::VectorXd &q, const Eigen::VectorXd &dq, bool reset_contact_state);

      void setJointFriction(const Eigen::VectorXd &joint_friction);

      /**
       * Convenience method to perform a single dt timestep of the simulation. 
       * Computes q, dq, ddq, and contact forces for a single time step 
       * results are stored in sim.q, sim.v, sim.dv, sim.f 
       */

      virtual void step(const Eigen::VectorXd &tau)=0;

      const Eigen::VectorXd& get_q() const {return q_;};
      const Eigen::VectorXd& get_v() const {return v_;};
      const Eigen::VectorXd& get_dv() const {return dv_;};

    protected:
      Eigen::Vector3d  f_tmp;
      const double sub_dt;
      Eigen::VectorXd q_;  
      Eigen::VectorXd qnext_;
      Eigen::VectorXd v_;
      Eigen::VectorXd dv_;
      Eigen::VectorXd vMean_;
      Eigen::VectorXd tau_;
      unsigned int nc_=0;
      int nactive_; 
      int newActive_;
      double elapsedTime_;  
      bool resetflag_ = false;
      bool contactChange_; 
      EulerIntegrationType integration_type_;   
      // bool semi_implicit_; // flag to activate semi-implicit Euler integration scheme

      /**
        * loops over contact points, checks active contacts and sets reference contact positions 
      */
      void detectContacts();
      void forwardDynamics(Eigen::VectorXd &tau, Eigen::VectorXd &dv, const Eigen::VectorXd *q=NULL, const Eigen::VectorXd *v=NULL); 
      virtual void computeContactForces()=0;

      const pinocchio::Model *model_;
      pinocchio::Data *data_;

      double dt_;
      int n_integration_steps_;
  
      std::vector<ContactPoint *> contacts_;
      std::vector<ContactObject *> objects_;

      Eigen::VectorXd joint_friction_;
      bool joint_friction_flag_ = 0;
  

      /** which forward dynamics to use 
       *  1: pinocchio::computeMinverse()
       *  2: pinocchio::aba()
       *  3: cholesky decompostion 
       **/ 
      const int whichFD_; 
      Eigen::LLT<MatrixXd> lltM_; /*!< used for Cholesky FD */ 
      Eigen::MatrixXd inverseM_;  /*!< used for pinocchio::computeMinverse() */ 
      Eigen::VectorXd mDv_; 
      Eigen::VectorXd fkDv_; // filled with zeros for second order kinematics  

  }; // class AbstractSimulator

/*_______________________________________________________________________________*/

  class EulerSimulator : public AbstractSimulator
  {
    public: 
      EulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD, EulerIntegrationType type); 
      ~EulerSimulator(){};

    /**
     * Explicit Euler first oder step 
    */
      void step(const Eigen::VectorXd &tau) override;

    protected:
      void computeContactForces() override;
      

  }; // class EulerSimulator

/*_______________________________________________________________________________*/

  class ImplicitEulerSimulator : public EulerSimulator
  {
    public: 
      ImplicitEulerSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD); 
      ~ImplicitEulerSimulator(){};

    /**
     * Implicit Euler first oder step 
    */
      void step(const Eigen::VectorXd &tau) override;

    protected:      
      void computeDynamicsAndJacobian(Eigen::VectorXd &tau, Eigen::VectorXd &q , Eigen::VectorXd &v, Eigen::VectorXd &f, Eigen::MatrixXd &Fx);

      Eigen::VectorXd vnext_;   // guess used for the iterative search
      Eigen::MatrixXd Fx_;      // dynamics Jacobian matrix
  }; // class ImplicitEulerSimulator


  /*_______________________________________________________________________________*/


  class RK4Simulator : public EulerSimulator
  {
    public: 
      RK4Simulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD);  
      ~RK4Simulator(){};

    /**
     * Runge Kutta 4th order, only applied for integrating acceleration to velocity 
    */

      void step(const Eigen::VectorXd &tau) override;

    protected:
      void computeContactForces(bool updateContactStates);

    private: 
      //\brief : vectors for the RK4 integration will be allocated in the constructor, depends on state dimension
      std::vector<Eigen::VectorXd> qi_;
      std::vector<Eigen::VectorXd> vi_;
      std::vector<Eigen::VectorXd> dvi_;
      std::vector<double> rk_factors_;

      // std::vector<Eigen::VectorXd> dyi_;

      

  }; // class RK4Simulator

/*_______________________________________________________________________________*/

  class ExponentialSimulator : public AbstractSimulator
  {
    typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagonalMatrixXd;

    public:
      /**
       * slipping metho selects anchor point update method during slipping 
       * 1: compute average force over the integration step, project on the cone boundary then update p0 
       * 2: a QP method to update the anchor point velocity, then average force is computed 
       **/  
      ExponentialSimulator(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps,
              int whichFD, EulerIntegrationType type, int slipping_method=1, bool compute_predicted_forces=false, 
              int exp_max_mat_mul=100, int lds_max_mat_mul=100); 

      ~ExponentialSimulator(){};
      void step(const Eigen::VectorXd &tau) override;

      int getMatrixMultiplications(){ return utilDense_.getMatrixMultiplications(); }
      // Return the L1 norm of the last matrix used for computing the matrix exponential
      double getMatrixExpL1Norm(){ return utilDense_.getL1Norm(); }
      void useMatrixBalancing(bool flag){ utilDense_.useBalancing(flag); }
      void assumeSlippageContinues(bool flag){ assumeSlippageContinues_=flag; }

    protected:
      /**
       * AbstractSimulator::computeContactState() must be called before  
       * calling ExponentialSimulator::computeContactForces()
       */
      void computeContactForces() override; 
      /**
       * computes average contact force during one integration step 
       * loops over the average force to compute tangential and normal force per contact 
       * projects any violations of the cone onto its boundaries 
       * sets a flag to to switch integration mode to include saturated forces 
       */
      void checkFrictionCone(); 

      void resizeVectorsAndMatrices();
      // convenience method to compute terms needed in integration  
      void computeExpLDS();

      int slipping_method_; 
      bool compute_predicted_forces_;
      bool assumeSlippageContinues_; // flag deciding whether dp0 is used in force computation 
      
      Eigen::VectorXd f_;  // contact forces
      Eigen::MatrixXd Jc_; // contact Jacobian for all contacts 
      Eigen::VectorXd p0_;  // anchor point positions
      Eigen::VectorXd dp0_; // anchor point velocities
      Eigen::VectorXd p_;   // contact point positions
      Eigen::VectorXd dp_;  // contact point velocities
      Eigen::VectorXd x0_;
      Eigen::VectorXd a_;
      Eigen::VectorXd b_;
      Eigen::VectorXd intxt_;
      Eigen::VectorXd int2xt_;
      Eigen::VectorXd dv_bar; 
      // contact acceleration components 
      Eigen::VectorXd dJv_;  
      DiagonalMatrixXd K;
      DiagonalMatrixXd B;
      DiagonalMatrixXd B_copy;
      Eigen::MatrixXd D;
      Eigen::MatrixXd A; 
      Eigen::MatrixXd MinvJcT_;
      Eigen::MatrixXd Upsilon_;
      Eigen::MatrixXd JcT_; 
      // expokit 
      expokit::LDSUtility<double, Dynamic> utilDense_;
      const int expMaxMatMul_;
      const int ldsMaxMatMul_; 
      // 
      void computePredictedXandF();  // predicts xf at end of integration step 
      expokit::MatrixExponential<double, Dynamic> util_eDtA;
      expokit::MatExpIntegral<double>  util_int_eDtA_one = expokit::MatExpIntegral<double>(6);   // current implementation static 
      expokit::MatExpIntegral<double>  util_int_eDtA_two = expokit::MatExpIntegral<double>(12);   // current implementation static 
      expokit::MatExpIntegral<double>  util_int_eDtA_three = expokit::MatExpIntegral<double>(18);   // current implementation static 
      expokit::MatExpIntegral<double>  util_int_eDtA_four = expokit::MatExpIntegral<double>(24);   // current implementation static 
      /*!< terms to approximate integral of e^{\tau A} */ 

      Eigen::MatrixXd expAdt_; 
      Eigen::MatrixXd inteAdt_;

      
      Eigen::VectorXd predictedForce_;
      Eigen::VectorXd predictedX0_;  
      Eigen::VectorXd predictedXf_; 
      Eigen::VectorXd dvMean_;
      Eigen::VectorXd temp01_;
      Eigen::VectorXd temp02_;
      Eigen::VectorXd temp03_;
      Eigen::VectorXd temp04_;
      Eigen::MatrixXd tempStepMat_; 
      // friction cone 
      Eigen::VectorXd f_avg;  // average force for cone 
      Eigen::VectorXd f_avg2;  // average of average force for cone 
      Eigen::VectorXd fpr_;   // projected force on cone boundaries 
      Eigen::VectorXd fpr2_;   // projected force on cone boundaries
      double cone_direction_; // angle of tangential(to contact surface) force 

      // Eigen::Vector3d f_avg_i; 
      // Eigen::Vector3d normalFi_; // normal component of contact force Fi at contact Ci  
      // Eigen::Vector3d tangentFi_; // normal component of contact force Fi at contact Ci  
      // Eigen::Vector3d f_avg_i2; 
      // Eigen::Vector3d normalFi_2; // normal component of contact force Fi at contact Ci  
      // Eigen::Vector3d tangentFi_2; // normal component of contact force Fi at contact Ci  

      Eigen::VectorXd dvMean2_; // used in method 1 of contact slipping 
      Eigen::VectorXd vMean2_; // used in method 1 of contact slipping 

      

      // double fnor_;   // norm of normalFi_  
      // double ftan_;   // norm of tangentFi_ 
      // double fnor2_;   // norm of normalFi_  
      // double ftan2_;   // norm of tangentFi_ 
      unsigned int i_active_; // index of the active contact      

      /**
       * solves a QP to update anchor points of sliding contacts
       * min || dp0_avg || ^ 2 
       * st. Fc \in Firction Cone
      //  **/  
      // void computeSlipping(); 
      // Eigen::MatrixXd Q_cone; 
      // Eigen::VectorXd q_cone; 
      // Eigen::MatrixXd Cineq_cone; 
      // Eigen::VectorXd cineq_cone; 
      // Eigen::MatrixXd Ceq_cone; 
      // Eigen::VectorXd ceq_cone; 
      // Eigen::VectorXd optdP_cone; 

      

      // Eigen::MatrixXd cone_constraints_;
      // Eigen::MatrixXd eq_cone_constraints_;
      // Eigen::MatrixXd contact_position_integrator_; 
      // Eigen::MatrixXd D_intExpA_integrator; 

      // eiquadprog::solvers::EiquadprogFast_status expected_qp = eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL;

      // eiquadprog::solvers::EiquadprogFast_status status_qp;

  }; // class ExponentialSimulator

} // namespace consim 
