// #include <pinocchio/bindings/python/multibody/data.hpp>
// #include <pinocchio/bindings/python/multibody/model.hpp>
// #include <pinocchio/fwd.hpp>
// #include <boost/python.hpp>
// #include <iostream>
// #include <eigenpy/eigenpy.hpp>

// #include "consim/simulator.hpp"


// IMPORTANT!!!!! DO NOT CHANGE THE ORDER OF THE INCLUDES HERE (COPIED FROM TSID) 
#include <pinocchio/fwd.hpp>
#include <boost/python.hpp>
#include <iostream>
// #include <Eigen/Dense>
#include <string>
#include <eigenpy/eigenpy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/make_constructor.hpp>

#include <pinocchio/bindings/python/multibody/data.hpp>
#include <pinocchio/bindings/python/multibody/model.hpp>

#include "consim/simulator.hpp"

//eigenpy::switchToNumpyMatrix();


namespace consim {
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

EulerSimulator* build_euler_simulator(
    float dt, int n_integration_steps, const pinocchio::Model& model, pinocchio::Data& data,
    Eigen::Vector3d stifness, Eigen::Vector3d damping, double frictionCoefficient, int whichFD, int type)
{
  LinearPenaltyContactModel *contact_model = new LinearPenaltyContactModel(
      stifness, damping, frictionCoefficient);

  ContactObject* obj = new FloorObject("Floor", *contact_model);

  if(!model.check(data))
  {
    std::cout<<"[build_euler_simulator] Data is not consistent with specified model\n";
    data = pinocchio::Data(model);
  }
  EulerSimulator* sim = new EulerSimulator(model, data, dt, n_integration_steps, whichFD, (EulerIntegrationType)type);
  sim->addObject(*obj);

  return sim;
}


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

namespace bp = boost::python;


#define ADD_PROPERTY_RETURN_BY_VALUE(name, ref) add_property(name, \
    make_getter(ref, bp::return_value_policy<bp::return_by_value>()), \
    make_setter(ref, bp::return_value_policy<bp::return_by_value>()))

#define ADD_PROPERTY_READONLY_RETURN_BY_VALUE(name, ref) add_property(name, \
    make_getter(ref, bp::return_value_policy<bp::return_by_value>()))

BOOST_PYTHON_MODULE(libconsim_pywrap)
{
    using namespace boost::python;
    eigenpy::enableEigenPy();

    bp::def("build_euler_simulator", build_euler_simulator,
            "A simple way to create a simulator using explicit euler integration with floor object and LinearPenaltyContactModel.",
            bp::return_value_policy<bp::manage_new_object>());

    bp::def("build_rk4_simulator", build_rk4_simulator,
            "A simple way to create a simulator using Runge-Kutta 4 integration with floor object and LinearPenaltyContactModel.",
            bp::return_value_policy<bp::manage_new_object>());

    bp::def("build_exponential_simulator", build_exponential_simulator,
            "A simple way to create a simulator using exponential integration with floor object and LinearPenaltyContactModel.",
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



    bp::class_<EulerSimulator, bases<AbstractSimulatorWrapper>>("EulerSimulator",
                          "Euler Simulator class",
                          bp::init<pinocchio::Model &, pinocchio::Data &, float, int, int, EulerIntegrationType>())
        .def("add_contact_point", &EulerSimulator::addContactPoint, return_internal_reference<>())
        .def("get_contact", &EulerSimulator::getContact, return_internal_reference<>())
        .def("add_object", &EulerSimulator::addObject)
        .def("reset_state", &EulerSimulator::resetState)
        .def("reset_contact_anchor", &EulerSimulator::resetContactAnchorPoint)
        .def("set_joint_friction", &EulerSimulator::setJointFriction)
        .def("step", &EulerSimulator::step)
        .def("get_q", &EulerSimulator::get_q,bp::return_value_policy<bp::copy_const_reference>(), "configuration state vector")
        .def("get_v", &EulerSimulator::get_v,bp::return_value_policy<bp::copy_const_reference>(), "tangent vector to configuration")
        .def("get_dv", &EulerSimulator::get_dv,bp::return_value_policy<bp::copy_const_reference>(), "time derivative of tangent vector to configuration");

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
}

}
