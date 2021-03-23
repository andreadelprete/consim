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

#pragma once

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

#include "consim/simulators/base.hpp"

namespace bp = boost::python;

#define ADD_PROPERTY_RETURN_BY_VALUE(name, ref) add_property(name, \
    make_getter(ref, bp::return_value_policy<bp::return_by_value>()), \
    make_setter(ref, bp::return_value_policy<bp::return_by_value>()))

#define ADD_PROPERTY_READONLY_RETURN_BY_VALUE(name, ref) add_property(name, \
    make_getter(ref, bp::return_value_policy<bp::return_by_value>()))

namespace consim 
{
  
// abstract simulator wrapper for bindings 
class AbstractSimulatorWrapper : public AbstractSimulator, public boost::python::wrapper<AbstractSimulator>
{
  public: 
    AbstractSimulatorWrapper(const pinocchio::Model &model, pinocchio::Data &data, float dt, int n_integration_steps, int whichFD,
    EulerIntegrationType type) : 
              AbstractSimulator(model, data, dt, n_integration_steps, whichFD, type), boost::python::wrapper<AbstractSimulator>() {}

    void step(const Eigen::VectorXd &tau){
      this->get_override("step")(tau);
    }

  void computeContactForces(){
    this->get_override("computeContactForces")();
  }

};

class ContactObjectWrapper : public ContactObject, public boost::python::wrapper<ContactObject>
{
  public: 
    ContactObjectWrapper(const std::string & name, ContactModel& contact_model) : 
      ContactObject(name, contact_model), boost::python::wrapper<ContactObject>() {}

    bool checkCollision(ContactPoint &cp){
      bool st = this->get_override("checkCollision")(cp);
      return st; 
    }

    void computePenetration(ContactPoint &cp){
      this->get_override("computePenetration")(cp);
    }

};

}
