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

#include "consim/bindings/python/common.hpp"
#include "consim/bindings/python/base.hpp"
#include "consim/bindings/python/explicit_euler.hpp"
#include "consim/bindings/python/implicit_euler.hpp"
#include "consim/bindings/python/rk4.hpp"
#include "consim/bindings/python/exponential.hpp"

namespace bp = boost::python;

#define ADD_PROPERTY_RETURN_BY_VALUE(name, ref) add_property(name, \
    make_getter(ref, bp::return_value_policy<bp::return_by_value>()), \
    make_setter(ref, bp::return_value_policy<bp::return_by_value>()))

#define ADD_PROPERTY_READONLY_RETURN_BY_VALUE(name, ref) add_property(name, \
    make_getter(ref, bp::return_value_policy<bp::return_by_value>()))

namespace consim 
{

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
    
    export_base();
    export_explicit_euler();
    export_implicit_euler();
    export_rk4();
    export_exponential();

    bp::class_<ContactObjectWrapper, boost::noncopyable>("ContactObject", "Abstract Contact Object Class", 
                         bp::init<const std::string & , ContactModel& >());
}

}
