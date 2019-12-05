/*
 * binding.cpp
 *
 *  Created on: Oct 10, 2018
 *      Author: jviereck
 */

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

Simulator* build_simple_simulator(
    float dt, int n_integration_steps, const pinocchio::Model& model, pinocchio::Data& data,
    double normal_spring_const, double normal_damping_coeff,
    double static_friction_spring_coeff, double static_friction_damping_spring_coeff,
    double static_friction_coeff, double dynamic_friction_coeff)
{
  DampedSpringStaticFrictionContactModel* contact_model = new
    DampedSpringStaticFrictionContactModel(
      normal_spring_const, normal_damping_coeff, static_friction_spring_coeff,
      static_friction_damping_spring_coeff, static_friction_coeff, dynamic_friction_coeff);

  Object* obj = new FloorObject("Floor", *contact_model);

  if(!model.check(data))
  {
    std::cout<<"[build_simple_simulator] Data is not consistent with specified model\n";
    data = pinocchio::Data(model);
  }
  Simulator* sim = new Simulator(dt, n_integration_steps, model, data);
  sim->add_object(*obj);

  return sim;
}


Simulator* build_walking_simulator(
    float dt, int n_integration_steps, pinocchio::Model& model, pinocchio::Data& data,
    double spring_stiffness_coeff_, double spring_damping_coeff_,
    double static_friction_coeff_, double dynamic_friction_coeff_,
    double maximum_penetration_, bool enable_friction_cone_)
{
  NonlinearSpringDamperContactModel* contact_model = new
    NonlinearSpringDamperContactModel(
      spring_stiffness_coeff_, spring_damping_coeff_, static_friction_coeff_,
      dynamic_friction_coeff_, maximum_penetration_, enable_friction_cone_);

  Object* obj = new FloorObject("Floor", *contact_model);

  Simulator* sim = new Simulator(dt, n_integration_steps, model, data);
  sim->add_object(*obj);

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

    bp::def("build_simple_simulator", build_simple_simulator,
            "A simple way to create a simulator with floor object and DampedSpringStaticFrictionContactModel.",
            bp::return_value_policy<bp::manage_new_object>());

    bp::def("build_walking_simulator", build_walking_simulator,
            "A simple way to create a simulator with floor object and NonlinearSpringDamperContactModel.",
            bp::return_value_policy<bp::manage_new_object>());

    bp::def("stop_watch_report", stop_watch_report,
            "Report all the times measured by the shared stop-watch.",
            bp::return_value_policy<bp::manage_new_object>());

    bp::class_<Contact>("Contact",
    										"Contact struct")
				.def_readwrite("active", &Contact::active)
        .def_readwrite("frame_id", &Contact::frame_id)
        .def_readwrite("friction_flag", &Contact::friction_flag)
        .ADD_PROPERTY_RETURN_BY_VALUE("x", &Contact::x)
        .ADD_PROPERTY_RETURN_BY_VALUE("v", &Contact::v)
        .ADD_PROPERTY_RETURN_BY_VALUE("x_start", &Contact::x_start)
        .ADD_PROPERTY_RETURN_BY_VALUE("contact_surface_normal", &Contact::contact_surface_normal)
        .ADD_PROPERTY_RETURN_BY_VALUE("normal", &Contact::normal)
        .ADD_PROPERTY_RETURN_BY_VALUE("normvel", &Contact::normvel)
        .ADD_PROPERTY_RETURN_BY_VALUE("tangent", &Contact::tangent)
        .ADD_PROPERTY_RETURN_BY_VALUE("tanvel", &Contact::tanvel)
        .ADD_PROPERTY_RETURN_BY_VALUE("viscvel", &Contact::viscvel)
        .ADD_PROPERTY_RETURN_BY_VALUE("f", &Contact::f)
				;

    bp::class_<Simulator>("Simulator",
                          "Main simulator class",
                          bp::init<float, int, pinocchio::Model&, pinocchio::Data&>())
        .def("add_contact_point", &Simulator::add_contact_point, return_internal_reference<>())
        .def("get_contact", &Simulator::get_contact, return_internal_reference<>())
        .def("step", &Simulator::step)
        .def("add_object", &Simulator::add_object)
        .def("reset_state", &Simulator::reset_state)
        .def("set_joint_friction", &Simulator::set_joint_friction)

        .ADD_PROPERTY_READONLY_RETURN_BY_VALUE("q", &Simulator::q_)
        .ADD_PROPERTY_READONLY_RETURN_BY_VALUE("dq", &Simulator::dq_)
        .ADD_PROPERTY_READONLY_RETURN_BY_VALUE("tau", &Simulator::tau_)
        ;

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
