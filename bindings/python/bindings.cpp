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
#include "consim/bindings/python/contacts.hpp"

namespace consim 
{

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

    export_contacts();
    export_base();
    export_explicit_euler();
    export_implicit_euler();
    export_rk4();
    export_exponential();
}

}
