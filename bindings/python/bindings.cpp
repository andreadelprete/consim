#include <pinocchio/bindings/python/multibody/data.hpp>
#include <pinocchio/bindings/python/multibody/model.hpp>
#include <pinocchio/fwd.hpp>
#include <boost/python.hpp>
#include <iostream>
#include <eigenpy/eigenpy.hpp>

#include "consim/simulator.hpp"
#include "consim/utils/stop-watch.hpp"


//eigenpy::switchToNumpyMatrix();

namespace consim {

EulerSimulator* build_euler_simulator(
    float dt, int n_integration_steps, const pinocchio::Model& model, pinocchio::Data& data,
    double normal_spring_const, double normal_damping_coeff,
    double static_friction_spring_coeff, double static_friction_damping_spring_coeff,
    double static_friction_coeff, double dynamic_friction_coeff)
{
  LinearPenaltyContactModel *contact_model = new LinearPenaltyContactModel(
      normal_spring_const, normal_damping_coeff, static_friction_spring_coeff,
      static_friction_damping_spring_coeff, static_friction_coeff, dynamic_friction_coeff);

  Object* obj = new FloorObject("Floor", *contact_model);

  if(!model.check(data))
  {
    std::cout<<"[build_euler_simulator] Data is not consistent with specified model\n";
    data = pinocchio::Data(model);
  }
  EulerSimulator* sim = new EulerSimulator(model, data, dt, n_integration_steps);
  sim->addObject(*obj);

  return sim;
}

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

    bp::def("stop_watch_report", stop_watch_report,
            "Report all the times measured by the shared stop-watch.",
            bp::return_value_policy<bp::manage_new_object>());

    bp::class_<ContactPoint>("Contact",
                             "Contact struct")
        .def_readwrite("active", &ContactPoint::active)
        .def_readwrite("frame_id", &ContactPoint::frame_id)
        .def_readwrite("friction_flag", &ContactPoint::friction_flag)
        .ADD_PROPERTY_RETURN_BY_VALUE("x", &ContactPoint::x)
        .ADD_PROPERTY_RETURN_BY_VALUE("v", &ContactPoint::v)
        .ADD_PROPERTY_RETURN_BY_VALUE("x_start", &ContactPoint::x_start)
        .ADD_PROPERTY_RETURN_BY_VALUE("contact_surface_normal", &ContactPoint::contact_surface_normal)
        .ADD_PROPERTY_RETURN_BY_VALUE("normal", &ContactPoint::normal)
        .ADD_PROPERTY_RETURN_BY_VALUE("normvel", &ContactPoint::normvel)
        .ADD_PROPERTY_RETURN_BY_VALUE("tangent", &ContactPoint::tangent)
        .ADD_PROPERTY_RETURN_BY_VALUE("tanvel", &ContactPoint::tanvel)
        .ADD_PROPERTY_RETURN_BY_VALUE("viscvel", &ContactPoint::viscvel)
        .ADD_PROPERTY_RETURN_BY_VALUE("f", &ContactPoint::f);

    bp::class_<EulerSimulator>("EulerSimulator",
                          "Euler Simulator class",
                          bp::init<pinocchio::Model &, pinocchio::Data &, float, int>())
        .def("add_contact_point", &EulerSimulator::addContactPoint, return_internal_reference<>())
        .def("get_contact", &EulerSimulator::getContact, return_internal_reference<>())
        .def("step", &EulerSimulator::step)
        .def("add_object", &EulerSimulator::addObject)
        .def("reset_state", &EulerSimulator::resetState)
        .def("set_joint_friction", &EulerSimulator::setJointFriction)
        .def("get_q", &EulerSimulator::getQ)
        .def("get_dq", &EulerSimulator::getDq);

        // .ADD_PROPERTY_READONLY_RETURN_BY_VALUE("q", &EulerSimulator::q_)
        // .ADD_PROPERTY_READONLY_RETURN_BY_VALUE("dq", &EulerSimulator::dq_)
        // .ADD_PROPERTY_READONLY_RETURN_BY_VALUE("tau", &EulerSimulator::tau_);
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
