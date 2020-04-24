#include <pinocchio/bindings/python/multibody/data.hpp>
#include <pinocchio/bindings/python/multibody/model.hpp>
#include <pinocchio/fwd.hpp>
#include <boost/python.hpp>
#include <iostream>
// #include <Eigen/Dense>
#include <eigenpy/eigenpy.hpp>

#include "consim/simulator.hpp"
// #include "consim/utils/stop-watch.hpp"


//eigenpy::switchToNumpyMatrix();


namespace consim {
// abstract simulator wrapper for bindings 
class AbstractSimulatorWrapper : public AbstractSimulator, public boost::python::wrapper<AbstractSimulator>
{
  public: 
    AbstractSimulatorWrapper(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps) : 
              AbstractSimulator(model, data, dt, n_integration_steps), boost::python::wrapper<AbstractSimulator>() {}

    void step(const Eigen::VectorXd &tau){
      this->get_override("step")(tau);
    }

  void computeContactForces(const Eigen::VectorXd &dq){
    this->get_override("computeContactForces")(dq);
  }

};

EulerSimulator* build_euler_simulator(
    float dt, int n_integration_steps, const pinocchio::Model& model, pinocchio::Data& data,
    Eigen::Matrix3d stifness, Eigen::Matrix3d damping, double frictionCoefficient)
{
  LinearPenaltyContactModel *contact_model = new LinearPenaltyContactModel(
      stifness, damping, frictionCoefficient);

  ContactObject* obj = new FloorObject("Floor", *contact_model);

  if(!model.check(data))
  {
    std::cout<<"[build_euler_simulator] Data is not consistent with specified model\n";
    data = pinocchio::Data(model);
  }
  EulerSimulator* sim = new EulerSimulator(model, data, dt, n_integration_steps);
  sim->addObject(*obj);

  return sim;
}

// ExponentialSimulator* build_exponential_simulator(
//     float dt, int n_integration_steps, const pinocchio::Model& model, pinocchio::Data& data,
//     double normal_spring_const, double normal_damping_coeff,
//     double static_friction_spring_coeff, double static_friction_damping_spring_coeff,
//     double static_friction_coeff, double dynamic_friction_coeff,bool sparse, bool invertibleA)
// {
//   LinearPenaltyContactModel *contact_model = new LinearPenaltyContactModel(
//       normal_spring_const, normal_damping_coeff, static_friction_spring_coeff,
//       static_friction_damping_spring_coeff, static_friction_coeff, dynamic_friction_coeff);

//   ContactObject* obj = new FloorObject("Floor", *contact_model);

//   if(!model.check(data))
//   {
//     std::cout<<"[build_exponential_simulator] Data is not consistent with specified model\n";
//     data = pinocchio::Data(model);
//   }
//   ExponentialSimulator* sim = new ExponentialSimulator(model, data, dt, n_integration_steps, sparse, invertibleA);
//   sim->addObject(*obj);

//   return sim;
// }

void stop_watch_report(int precision)
{
  getProfiler().report_all(precision);
}


namespace bp = boost::python;


#define ADD_PROPERTY_RETURN_BY_VALUE(name, ref) add_property(name, \
    make_getter(ref, bp::return_value_policy<bp::return_by_value>()), \
    make_setter(ref, bp::return_value_policy<bp::return_by_value>()))

#define ADD_PROPERTY_READONLY_RETURN_BY_VALUE(name, ref) add_property(name, \
    make_getter(ref, bp::return_value_policy<bp::return_by_value>()))

BOOST_PYTHON_MODULE(libconsim_pywrap)
{
    using namespace boost::python;
    eigenpy::enableEigenPy();

    bp::def("build_euler_simulator", build_euler_simulator,
            "A simple way to create a simulator using explicit euler integration with floor object and LinearPenaltyContactModel.",
            bp::return_value_policy<bp::manage_new_object>());

    // bp::def("build_exponential_simulator", build_exponential_simulator,
    //         "A simple way to create a simulator using exponential integration with floor object and LinearPenaltyContactModel.",
    //         bp::return_value_policy<bp::manage_new_object>());

    bp::def("stop_watch_report", stop_watch_report,
            "Report all the times measured by the shared stop-watch.",
            bp::return_value_policy<bp::manage_new_object>());

    bp::class_<ContactPoint>("Contact",
                             "Contact struct",
                          bp::init<std::string, unsigned int, unsigned int, bool >())
        .def("updatePosition", &ContactPoint::updatePosition, return_internal_reference<>())
        .def("firstOrderContactKinematics", &ContactPoint::firstOrderContactKinematics, return_internal_reference<>())
        .def("secondOrderContactKinematics", &ContactPoint::secondOrderContactKinematics, return_internal_reference<>())
        .def("computeContactForce", &ContactPoint::computeContactForce, return_internal_reference<>())
        .ADD_PROPERTY_RETURN_BY_VALUE("frame_id", &ContactPoint::frame_id)
        .ADD_PROPERTY_RETURN_BY_VALUE("x", &ContactPoint::x)
        .ADD_PROPERTY_RETURN_BY_VALUE("v", &ContactPoint::v)
        .ADD_PROPERTY_RETURN_BY_VALUE("x_start", &ContactPoint::x_start)
        .ADD_PROPERTY_RETURN_BY_VALUE("normal", &ContactPoint::normal)
        .ADD_PROPERTY_RETURN_BY_VALUE("normvel", &ContactPoint::normvel)
        .ADD_PROPERTY_RETURN_BY_VALUE("tangent", &ContactPoint::tangent)
        .ADD_PROPERTY_RETURN_BY_VALUE("tanvel", &ContactPoint::tanvel)
        .ADD_PROPERTY_RETURN_BY_VALUE("f", &ContactPoint::f);


    bp::class_<AbstractSimulatorWrapper, boost::noncopyable>("AbstractSimulator", "Abstract Simulator Class", 
                         bp::init<pinocchio::Model &, pinocchio::Data &, float, int>())
        .def("add_contact_point", &AbstractSimulatorWrapper::addContactPoint, return_internal_reference<>())
        .def("get_contact", &AbstractSimulatorWrapper::getContact, return_internal_reference<>())
        .def("add_object", &AbstractSimulatorWrapper::addObject)
        .def("reset_state", &AbstractSimulatorWrapper::resetState)
        .def("set_joint_friction", &AbstractSimulatorWrapper::setJointFriction)
        .def("step", bp::pure_virtual(&AbstractSimulatorWrapper::step))
        .def("get_q", &AbstractSimulatorWrapper::get_q,bp::return_value_policy<bp::copy_const_reference>(), "configuration state vector")
        .def("get_v", &AbstractSimulatorWrapper::get_v,bp::return_value_policy<bp::copy_const_reference>(), "tangent vector to configuration")
        .def("get_dv", &AbstractSimulatorWrapper::get_dv,bp::return_value_policy<bp::copy_const_reference>(), "time derivative of tangent vector to configuration");



    bp::class_<EulerSimulator, bases<AbstractSimulatorWrapper>>("EulerSimulator",
                          "Euler Simulator class",
                          bp::init<pinocchio::Model &, pinocchio::Data &, float, int>())
        .def("add_contact_point", &EulerSimulator::addContactPoint, return_internal_reference<>())
        .def("get_contact", &EulerSimulator::getContact, return_internal_reference<>())
        .def("add_object", &EulerSimulator::addObject)
        .def("reset_state", &EulerSimulator::resetState)
        .def("set_joint_friction", &EulerSimulator::setJointFriction)
        .def("step", &EulerSimulator::step)
        .def("get_q", &EulerSimulator::get_q,bp::return_value_policy<bp::copy_const_reference>(), "configuration state vector")
        .def("get_v", &EulerSimulator::get_v,bp::return_value_policy<bp::copy_const_reference>(), "tangent vector to configuration")
        .def("get_dv", &EulerSimulator::get_dv,bp::return_value_policy<bp::copy_const_reference>(), "time derivative of tangent vector to configuration");


    // bp::class_<ExponentialSimulator, bases<AbstractSimulatorWrapper>>("ExponentialSimulator",
    //                       "Exponential Simulator class",
    //                       bp::init<pinocchio::Model &, pinocchio::Data &, float, int, bool, bool>())
    //     .def("add_contact_point", &ExponentialSimulator::addContactPoint, return_internal_reference<>())
    //     .def("get_contact", &ExponentialSimulator::getContact, return_internal_reference<>())
    //     .def("add_object", &ExponentialSimulator::addObject)
    //     .def("reset_state", &ExponentialSimulator::resetState)
    //     .def("set_joint_friction", &ExponentialSimulator::setJointFriction)
    //     .def("step", &ExponentialSimulator::step)
    //     .def("get_q", &ExponentialSimulator::get_q,bp::return_value_policy<bp::copy_const_reference>(), "configuration state vector")
    //     .def("get_v", &ExponentialSimulator::get_v,bp::return_value_policy<bp::copy_const_reference>(), "tangent vector to configuration")
    //     .def("get_dv", &ExponentialSimulator::get_dv,bp::return_value_policy<bp::copy_const_reference>(), "time derivative of tangent vector to configuration");

}
//
//#include <pinocchio/fwd.hpp>
//#include <eigenpy/eigenpy.hpp>
//#include <eigenpy/geometry.hpp>

//#include "tsid/bindings/python/robots/expose-robots.hpp"
//#include "tsid/bindings/python/constraint/expose-constraints.hpp"
//#include "tsid/bindings/python/contacts/expose-contact.hpp"
//#include "tsid/bindings/python/trajectories/expose-trajectories.hpp"
//#include "tsid/bindings/python/tasks/expose-tasks.hpp"
//#include "tsid/bindings/python/solvers/expose-solvers.hpp"
//#include "tsid/bindings/python/formulations/expose-formulations.hpp"

//#include <boost/python/module.hpp>
//#include <boost/python/def.hpp>
//#include <boost/python/tuple.hpp>
//#include <boost/python/to_python_converter.hpp>

//namespace bp = boost::python;
//using namespace tsid::python;

//BOOST_PYTHON_MODULE(libtsid_pywrap)
//{
//  eigenpy::enableEigenPy();
//  eigenpy::exposeAngleAxis();
//  eigenpy::exposeQuaternion();

//  typedef Eigen::Matrix<double,6,6> Matrix6d;
//  typedef Eigen::Matrix<double,6,1> Vector6d;
//  typedef Eigen::Matrix<double,6,Eigen::Dynamic> Matrix6x;
//  typedef Eigen::Matrix<double,3,Eigen::Dynamic> Matrix3x;

//  eigenpy::enableEigenPySpecific<Matrix6d>();
//  eigenpy::enableEigenPySpecific<Vector6d>();
//  eigenpy::enableEigenPySpecific<Matrix6x>();
//  eigenpy::enableEigenPySpecific<Matrix3x>();
//  eigenpy::enableEigenPySpecific<Eigen::MatrixXd>();
//  eigenpy::enableEigenPySpecific<Eigen::Vector3d>();

//  exposeRobots();
//  exposeConstraints();
//  exposeContact();
//  exposeTrajectories();
//  exposeTasks();
//  exposeSolvers();
//  exposeFormulations();

//}
}
