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
#include "consim/bindings/python/rk4.hpp"

namespace bp = boost::python;
using namespace boost::python;

namespace consim 
{

RK4Simulator* build_rk4_simulator(
    float dt, int n_integration_steps, const pinocchio::Model& model, pinocchio::Data& data,
    Eigen::Vector3d stifness, Eigen::Vector3d damping, double frictionCoefficient, int whichFD)
{
  LinearPenaltyContactModel *contact_model = new LinearPenaltyContactModel(
      stifness, damping, frictionCoefficient);

  ContactObject* obj = new FloorObject("Floor", *contact_model);

  if(!model.check(data))
  {
    std::cout<<"[build_rk4_simulator] Data is not consistent with specified model\n";
    data = pinocchio::Data(model);
  }
  RK4Simulator* sim = new RK4Simulator(model, data, dt, n_integration_steps, whichFD);
  sim->addObject(*obj);

  return sim;
}

void export_rk4()
{
  bp::def("build_rk4_simulator", build_rk4_simulator,
            "A simple way to create a simulator using Runge-Kutta 4 integration with floor object and LinearPenaltyContactModel.",
            bp::return_value_policy<bp::manage_new_object>());

  bp::class_<RK4Simulator, bases<AbstractSimulatorWrapper>>("RK4Simulator",
                      "Runge-Kutta 4 Simulator class",
                      bp::init<pinocchio::Model &, pinocchio::Data &, float, int, int>())
        .def("add_contact_point", &RK4Simulator::addContactPoint, return_internal_reference<>())
        .def("get_contact", &RK4Simulator::getContact, return_internal_reference<>())
        .def("add_object", &RK4Simulator::addObject)
        .def("reset_state", &RK4Simulator::resetState)
        .def("reset_contact_anchor", &RK4Simulator::resetContactAnchorPoint)
        .def("set_joint_friction", &RK4Simulator::setJointFriction)
        .def("step", &RK4Simulator::step)
        .def("get_q", &RK4Simulator::get_q,bp::return_value_policy<bp::copy_const_reference>(), "configuration state vector")
        .def("get_v", &RK4Simulator::get_v,bp::return_value_policy<bp::copy_const_reference>(), "tangent vector to configuration")
        .def("get_dv", &RK4Simulator::get_dv,bp::return_value_policy<bp::copy_const_reference>(), "time derivative of tangent vector to configuration");
}

}
