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
#include "consim/bindings/python/exponential.hpp"
#include "consim/simulators/exponential.hpp"

namespace bp = boost::python;
using namespace boost::python;

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

void export_exponential()
{
  bp::def("build_exponential_simulator", build_exponential_simulator,
            "A simple way to create a simulator using exponential integration with floor object and LinearPenaltyContactModel.",
            bp::return_value_policy<bp::manage_new_object>());

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
        .def("assumeSlippageContinues", &ExponentialSimulator::assumeSlippageContinues)
        .def("setUseDiagonalMatrixExp", &ExponentialSimulator::setUseDiagonalMatrixExp)
        .def("setUpdateAFrequency", &ExponentialSimulator::setUpdateAFrequency)
        ;
}

}
