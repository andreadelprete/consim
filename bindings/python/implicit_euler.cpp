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
#include "consim/bindings/python/implicit_euler.hpp"

namespace bp = boost::python;
using namespace boost::python;

namespace consim 
{

ImplicitEulerSimulator* build_implicit_euler_simulator(
    float dt, int n_integration_steps, const pinocchio::Model& model, pinocchio::Data& data,
    Eigen::Vector3d stifness, Eigen::Vector3d damping, double frictionCoefficient)
{
  LinearPenaltyContactModel *contact_model = new LinearPenaltyContactModel(
      stifness, damping, frictionCoefficient);

  ContactObject* obj = new FloorObject("Floor", *contact_model);

  if(!model.check(data))
  {
    std::cout<<"[build_implicit_euler_simulator] Data is not consistent with specified model\n";
    data = pinocchio::Data(model);
  }
  ImplicitEulerSimulator* sim = new ImplicitEulerSimulator(model, data, dt, n_integration_steps);
  sim->addObject(*obj);

  return sim;
}

void export_implicit_euler()
{
  bp::def("build_implicit_euler_simulator", build_implicit_euler_simulator,
            "A simple way to create a simulator using implicit euler integration with floor object and LinearPenaltyContactModel.",
            bp::return_value_policy<bp::manage_new_object>());

  bp::class_<ImplicitEulerSimulator, bases<AbstractSimulatorWrapper>>("ImplicitEulerSimulator",
                          "Implicit Euler Simulator class",
                          bp::init<pinocchio::Model &, pinocchio::Data &, float, int>())
        .def("add_contact_point", &ImplicitEulerSimulator::addContactPoint, return_internal_reference<>())
        .def("get_contact", &ImplicitEulerSimulator::getContact, return_internal_reference<>())
        .def("add_object", &ImplicitEulerSimulator::addObject)
        .def("reset_state", &ImplicitEulerSimulator::resetState)
        .def("reset_contact_anchor", &ImplicitEulerSimulator::resetContactAnchorPoint)
        .def("set_joint_friction", &ImplicitEulerSimulator::setJointFriction)
        .def("set_use_finite_differences_dynamics", &ImplicitEulerSimulator::set_use_finite_differences_dynamics)
        .def("set_use_finite_differences_nle", &ImplicitEulerSimulator::set_use_finite_differences_nle)
        .def("set_use_current_state_as_initial_guess", &ImplicitEulerSimulator::set_use_current_state_as_initial_guess)
        .def("set_convergence_threshold", &ImplicitEulerSimulator::set_convergence_threshold)
        .def("get_avg_iteration_number", &ImplicitEulerSimulator::get_avg_iteration_number)
        .def("step", &ImplicitEulerSimulator::step)
        .def("get_q", &ImplicitEulerSimulator::get_q,bp::return_value_policy<bp::copy_const_reference>(), "configuration state vector")
        .def("get_v", &ImplicitEulerSimulator::get_v,bp::return_value_policy<bp::copy_const_reference>(), "tangent vector to configuration")
        .def("get_dv", &ImplicitEulerSimulator::get_dv,bp::return_value_policy<bp::copy_const_reference>(), "time derivative of tangent vector to configuration");
}

}
