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

namespace bp = boost::python;
using namespace boost::python;

namespace consim 
{

void export_base()
{
  bp::class_<AbstractSimulatorWrapper, boost::noncopyable>("AbstractSimulator", "Abstract Simulator Class", 
                         bp::init<pinocchio::Model &, pinocchio::Data &, float, int, int, EulerIntegrationType>())
        .def("add_contact_point", &AbstractSimulatorWrapper::addContactPoint, return_internal_reference<>())
        .def("get_contact", &AbstractSimulatorWrapper::getContact, return_internal_reference<>())
        .def("add_object", &AbstractSimulatorWrapper::addObject)
        .def("reset_state", &AbstractSimulatorWrapper::resetState)
        .def("reset_contact_anchor", &AbstractSimulatorWrapper::resetContactAnchorPoint)
        .def("set_joint_friction", &AbstractSimulatorWrapper::setJointFriction)
        .def("step", bp::pure_virtual(&AbstractSimulatorWrapper::step))
        .def("get_q", &AbstractSimulatorWrapper::get_q,bp::return_value_policy<bp::copy_const_reference>(), "configuration state vector")
        .def("get_v", &AbstractSimulatorWrapper::get_v,bp::return_value_policy<bp::copy_const_reference>(), "tangent vector to configuration")
        .def("get_dv", &AbstractSimulatorWrapper::get_dv,bp::return_value_policy<bp::copy_const_reference>(), "time derivative of tangent vector to configuration");

}

}
