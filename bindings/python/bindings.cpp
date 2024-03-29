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
#include "consim/bindings/python/rigid_euler.hpp"
#include "consim/bindings/python/contacts.hpp"
#include "consim/bindings/python/stop_watch.hpp"

namespace consim 
{

BOOST_PYTHON_MODULE(libconsim_pywrap)
{
    using namespace boost::python;
    eigenpy::enableEigenPy();      

    export_stop_watch();
    export_contacts();
    export_base();
    export_explicit_euler();
    export_implicit_euler();
    export_rk4();
    export_exponential();
    export_rigid_euler();
}

}
