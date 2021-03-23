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

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/motion.hpp>

#include "consim/object.hpp"
#include "consim/contact.hpp"

#include "utils/stop-watch.h"

#define CONSIM_PROFILER
#ifndef CONSIM_PROFILER
#define CONSIM_START_PROFILER(name)
#define CONSIM_STOP_PROFILER(name)
#else
#define CONSIM_START_PROFILER(name) getProfiler().start(name)
#define CONSIM_STOP_PROFILER(name) getProfiler().stop(name)
#endif


namespace consim {
  enum EulerIntegrationType{ EXPLICIT=0, SEMI_IMPLICIT=1, CLASSIC_EXPLICIT=2};

  typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagonalMatrixXd;

  /**
   * Detect active/inactive contact points
   */
  int detectContacts_imp(pinocchio::Data &data, std::vector<ContactPoint *> &contacts, std::vector<ContactObject*> &objects);

  /**
   * Compute the contact forces associated to the specified list of contacts and objects. 
   * Moreover, it computes their net effect on the generalized joint torques tau_f.
   */
  int computeContactForces_imp(const pinocchio::Model &model, pinocchio::Data &data, 
                            const Eigen::VectorXd &q, const Eigen::VectorXd &v, Eigen::VectorXd &tau_f, 
                            std::vector<ContactPoint*> &contacts, std::vector<ContactObject*> &objects);

  /** 
   * Integrate in state space.
   */
  void integrateState(const pinocchio::Model &model, const Eigen::VectorXd &x, const Eigen::VectorXd &dx, 
                      double dt, Eigen::VectorXd &xNext);

  /**
   * Compute the difference between x1 and x0, i.e. x1-x0, where x0 and x1 might live on a Lie group.
   */
  void differenceState(const pinocchio::Model &model, const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, 
                      Eigen::VectorXd &dx);

  /** 
   * Derivatives of the function that integrates in state space.
   */
  void DintegrateState(const pinocchio::Model &model, const Eigen::VectorXd &x, const Eigen::VectorXd &dx, 
                      double dt, Eigen::MatrixXd &J);

  /**
   * Derivatives of the function that computes the difference between x1 and x0, i.e. x1-x0, where x0 and x1 might live on a Lie group.
   */
  void DdifferenceState_x0(const pinocchio::Model &model, const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, 
                           Eigen::MatrixXd &J);

  void DdifferenceState_x1(const pinocchio::Model &model, const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, 
                           Eigen::MatrixXd &J);

} // namespace consim 
