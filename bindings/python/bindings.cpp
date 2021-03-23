//
//  Copyright (c) 2020-2021 UNITN, NYU
//
//  This file is part of consim
//  consim is free software: you can redistribute it
//  and/or modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation, either version
//  3 of the License, or (at your option) any later version.
//  consim is distributed in the hope that it will be
//  useful, but WITHOUT ANY WARRANTY; without even the implied warranty
//  of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
//  General Lesser Public License for more details. You should have
//  received a copy of the GNU Lesser General Public License along with
//  consim If not, see
//  <http://www.gnu.org/licenses/>.

// IMPORTANT!!!!! DO NOT CHANGE THE ORDER OF THE INCLUDES HERE (COPIED FROM TSID) 
#include <pinocchio/fwd.hpp>
#include <boost/python.hpp>
#include <iostream>
#include <string>
#include <eigenpy/eigenpy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/make_constructor.hpp>

#include <pinocchio/bindings/python/multibody/data.hpp>
#include <pinocchio/bindings/python/multibody/model.hpp>

#include "consim/simulators/exponential.hpp"

#include "consim/bindings/python/common.hpp"
#include "consim/bindings/python/base.hpp"
#include "consim/bindings/python/explicit_euler.hpp"
#include "consim/bindings/python/implicit_euler.hpp"
#include "consim/bindings/python/rk4.hpp"

namespace bp = boost::python;

#define ADD_PROPERTY_RETURN_BY_VALUE(name, ref) add_property(name, \
    make_getter(ref, bp::return_value_policy<bp::return_by_value>()), \
    make_setter(ref, bp::return_value_policy<bp::return_by_value>()))

#define ADD_PROPERTY_READONLY_RETURN_BY_VALUE(name, ref) add_property(name, \
    make_getter(ref, bp::return_value_policy<bp::return_by_value>()))

namespace consim 
{

ExponentialSimulator* build_exponential_simulator(
    float dt, int n_integration_steps, const pinocchio::Model& model, pinocchio::Data& data,
    Eigen::Vector3d stifness, Eigen::Vector3d damping, double frictionCoefficient, int which_slipping,
    bool compute_predicted_forces, int whichFD, bool semi_implicit,
    int exp_max_mat_mul, int lds_max_mat_mul, bool useMatrixBalancing)
{
  LinearPenaltyContactModel *contact_model = new LinearPenaltyContactModel(
      stifness, damping, frictionCoefficient);

  ContactObject* obj = new FloorObject("Floor", *contact_model);

  if(!model.check(data))
  {
    std::cout<<"[build_exponential_simulator] Data is not consistent with specified model\n";
    data = pinocchio::Data(model);
  }
  EulerIntegrationType type = semi_implicit ? EulerIntegrationType::SEMI_IMPLICIT : EulerIntegrationType::EXPLICIT;
  ExponentialSimulator* sim = new ExponentialSimulator(model, data, dt, n_integration_steps, whichFD, type, which_slipping, 
                                  compute_predicted_forces, exp_max_mat_mul, lds_max_mat_mul);
  sim->useMatrixBalancing(useMatrixBalancing);
  sim->addObject(*obj);

  return sim;
}

ContactObject* create_half_plane(Eigen::Vector3d stifness, Eigen::Vector3d damping, 
double frictionCoefficient, double alpha)
{
  LinearPenaltyContactModel *contact_model = new LinearPenaltyContactModel(
      stifness, damping, frictionCoefficient);

  ContactObject* obj = new HalfPlaneObject("HalfPlane", *contact_model, alpha);

  return obj; 
}

void stop_watch_report(int precision)
{
  getProfiler().report_all(precision);
}

long double stop_watch_get_average_time(const std::string & perf_name)
{
  return getProfiler().get_average_time(perf_name);
}

/** Returns minimum execution time of a certain performance */
long double stop_watch_get_min_time(const std::string & perf_name)
{
  return getProfiler().get_min_time(perf_name);
}

/** Returns maximum execution time of a certain performance */
long double stop_watch_get_max_time(const std::string & perf_name)
{
  return getProfiler().get_max_time(perf_name);
}

long double stop_watch_get_total_time(const std::string & perf_name)
{
  return getProfiler().get_total_time(perf_name);
}

void stop_watch_reset_all()
{
  getProfiler().reset_all();
}

BOOST_PYTHON_MODULE(libconsim_pywrap)
{
    using namespace boost::python;
    eigenpy::enableEigenPy();

    bp::def("build_exponential_simulator", build_exponential_simulator,
            "A simple way to create a simulator using exponential integration with floor object and LinearPenaltyContactModel.",
            bp::return_value_policy<bp::manage_new_object>());

    bp::def("create_half_plane", create_half_plane,
            "A simple way to add a half plane with LinearPenaltyContactModel.",
            bp::return_value_policy<bp::manage_new_object>());
      

    bp::def("stop_watch_report", stop_watch_report,
            "Report all the times measured by the shared stop-watch.");

    bp::def("stop_watch_get_average_time", stop_watch_get_average_time,
            "Get the average time measured by the shared stop-watch for the specified task.");

    bp::def("stop_watch_get_min_time", stop_watch_get_min_time,
            "Get the min time measured by the shared stop-watch for the specified task.");

    bp::def("stop_watch_get_max_time", stop_watch_get_max_time,
            "Get the max time measured by the shared stop-watch for the specified task.");

    bp::def("stop_watch_get_total_time", stop_watch_get_total_time,
            "Get the total time measured by the shared stop-watch for the specified task.");

    bp::def("stop_watch_reset_all", stop_watch_reset_all,
            "Reset the shared stop-watch.");

    bp::class_<ContactPoint>("Contact",
                             "Contact Point",
                          bp::init<pinocchio::Model &, const std::string &, unsigned int, unsigned int, bool >())
        .def("updatePosition", &ContactPoint::updatePosition, return_internal_reference<>())
        .def("firstOrderContactKinematics", &ContactPoint::firstOrderContactKinematics, return_internal_reference<>())
        .def("secondOrderContactKinematics", &ContactPoint::secondOrderContactKinematics, return_internal_reference<>())
        .def("resetAnchorPoint", &ContactPoint::resetAnchorPoint, return_internal_reference<>())
        .ADD_PROPERTY_RETURN_BY_VALUE("frame_id", &ContactPoint::frame_id)
        .ADD_PROPERTY_RETURN_BY_VALUE("name", &ContactPoint::name_)
        .ADD_PROPERTY_RETURN_BY_VALUE("active", &ContactPoint::active)
        .ADD_PROPERTY_RETURN_BY_VALUE("slipping", &ContactPoint::slipping)
        .ADD_PROPERTY_RETURN_BY_VALUE("x", &ContactPoint::x)
        .ADD_PROPERTY_RETURN_BY_VALUE("v", &ContactPoint::v)
        .ADD_PROPERTY_RETURN_BY_VALUE("x_anchor", &ContactPoint::x_anchor)
        .ADD_PROPERTY_RETURN_BY_VALUE("normal", &ContactPoint::normal)
        .ADD_PROPERTY_RETURN_BY_VALUE("normvel", &ContactPoint::normvel)
        .ADD_PROPERTY_RETURN_BY_VALUE("tangent", &ContactPoint::tangent)
        .ADD_PROPERTY_RETURN_BY_VALUE("tanvel", &ContactPoint::tanvel)
        .ADD_PROPERTY_RETURN_BY_VALUE("f", &ContactPoint::f)
        .ADD_PROPERTY_RETURN_BY_VALUE("f_avg", &ContactPoint::f_avg)
        .ADD_PROPERTY_RETURN_BY_VALUE("f_avg2", &ContactPoint::f_avg2)
        .ADD_PROPERTY_RETURN_BY_VALUE("f_prj", &ContactPoint::f_prj)
        .ADD_PROPERTY_RETURN_BY_VALUE("f_prj2", &ContactPoint::f_prj2)
        .ADD_PROPERTY_RETURN_BY_VALUE("predicted_f", &ContactPoint::predictedF_)
        .ADD_PROPERTY_RETURN_BY_VALUE("predicted_x", &ContactPoint::predictedX_)
        .ADD_PROPERTY_RETURN_BY_VALUE("predicted_v", &ContactPoint::predictedV_)
        .ADD_PROPERTY_RETURN_BY_VALUE("predicted_x0", &ContactPoint::predictedX0_);
    

    // bp::class_<AbstractSimulatorWrapper, boost::noncopyable>("AbstractSimulator", "Abstract Simulator Class", 
    //                      bp::init<pinocchio::Model &, pinocchio::Data &, float, int, int, EulerIntegrationType>())
    //     .def("add_contact_point", &AbstractSimulatorWrapper::addContactPoint, return_internal_reference<>())
    //     .def("get_contact", &AbstractSimulatorWrapper::getContact, return_internal_reference<>())
    //     .def("add_object", &AbstractSimulatorWrapper::addObject)
    //     .def("reset_state", &AbstractSimulatorWrapper::resetState)
    //     .def("reset_contact_anchor", &AbstractSimulatorWrapper::resetContactAnchorPoint)
    //     .def("set_joint_friction", &AbstractSimulatorWrapper::setJointFriction)
    //     .def("step", bp::pure_virtual(&AbstractSimulatorWrapper::step))
    //     .def("get_q", &AbstractSimulatorWrapper::get_q,bp::return_value_policy<bp::copy_const_reference>(), "configuration state vector")
    //     .def("get_v", &AbstractSimulatorWrapper::get_v,bp::return_value_policy<bp::copy_const_reference>(), "tangent vector to configuration")
    //     .def("get_dv", &AbstractSimulatorWrapper::get_dv,bp::return_value_policy<bp::copy_const_reference>(), "time derivative of tangent vector to configuration");

    export_base();
    export_explicit_euler();
    export_implicit_euler();
    export_rk4();

    bp::class_<ExponentialSimulator, bases<AbstractSimulatorWrapper>>("ExponentialSimulator",
                          "Exponential Simulator class",
                          bp::init<pinocchio::Model &, pinocchio::Data &, float, int, int, EulerIntegrationType, int, bool, int, int>())
        .def("add_contact_point", &ExponentialSimulator::addContactPoint, return_internal_reference<>())
        .def("get_contact", &ExponentialSimulator::getContact, return_internal_reference<>())
        .def("add_object", &ExponentialSimulator::addObject)
        .def("reset_state", &ExponentialSimulator::resetState)
        .def("reset_contact_anchor", &ExponentialSimulator::resetContactAnchorPoint)
        .def("set_joint_friction", &ExponentialSimulator::setJointFriction)
        .def("step", &ExponentialSimulator::step)
        .def("get_q", &ExponentialSimulator::get_q,bp::return_value_policy<bp::copy_const_reference>(), "configuration state vector")
        .def("get_v", &ExponentialSimulator::get_v,bp::return_value_policy<bp::copy_const_reference>(), "tangent vector to configuration")
        .def("get_dv", &ExponentialSimulator::get_dv,bp::return_value_policy<bp::copy_const_reference>(), "time derivative of tangent vector to configuration")
        .def("getMatrixMultiplications", &ExponentialSimulator::getMatrixMultiplications)
        .def("getMatrixExpL1Norm", &ExponentialSimulator::getMatrixExpL1Norm)
        .def("assumeSlippageContinues", &ExponentialSimulator::assumeSlippageContinues);

    bp::class_<ContactObjectWrapper, boost::noncopyable>("ContactObject", "Abstract Contact Object Class", 
                         bp::init<const std::string & , ContactModel& >());
}

}
