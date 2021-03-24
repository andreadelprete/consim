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
#include "consim/bindings/python/contacts.hpp"
#include "consim/contact.hpp"

namespace bp = boost::python;
using namespace boost::python;

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

void export_contacts()
{
  bp::def("create_half_plane", create_half_plane,
            "A simple way to add a half plane with LinearPenaltyContactModel.",
            bp::return_value_policy<bp::manage_new_object>());

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

  bp::class_<ContactObjectWrapper, boost::noncopyable>("ContactObject", "Abstract Contact Object Class", 
                         bp::init<const std::string & , ContactModel& >());
}

}
