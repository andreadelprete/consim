
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

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp> // se3.integrate
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/cholesky.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>

#include "consim/object.hpp"
#include "consim/contact.hpp"
#include "consim/simulators/common.hpp"

#include <iostream>

using namespace Eigen;

namespace consim 
{

int detectContacts_imp(pinocchio::Data &data, std::vector<ContactPoint *> &contacts, std::vector<ContactObject*> &objects)
{
  // counter of number of active contacts
  int newActive = 0;
  // Loop over all the contact points, over all the objects.
  for (auto &cp : contacts) {
    cp->updatePosition(data);
    if(cp->active)
    {
      // for unilateral active contacts check if they are still in contact with same object
      if (cp->unilateral) {
        // optr: object pointer
        if (!cp->optr->checkCollision(*cp))
        {
          // if not => set them to inactive and move forward to searching a colliding object
          cp->active = false;
          cp->f.fill(0);
          // cp->friction_flag = false;
        } else {
          newActive += 1;
          // If the contact point is still active, then no need to search for
          // other contacting object (we assume there is only one object acting
          // on a contact point at each timestep).
          continue;
        }
      }
      else {
        // bilateral contacts never break
        newActive += 1;
        continue;
      }
    }
    // if a contact is bilateral and active => no need to search
    // for colliding object because bilateral contacts never break
    if(cp->unilateral || !cp->active) {  
      for (auto &optr : objects) {
        if (optr->checkCollision(*cp))
        {
          cp->active = true;
          newActive += 1; 
          cp->optr = optr;
          // if(!cp->unilateral){
          //   std::cout<<"Bilateral contact with object "<<optr->getName()<<" at point "<<cp->x.transpose()<<std::endl;
          // }
          break;
        }
      }
    }
  }
  return newActive;
}

int computeContactForces_imp(const pinocchio::Model &model, pinocchio::Data &data, const Eigen::VectorXd &q, 
                         const Eigen::VectorXd &v, Eigen::VectorXd &tau_f, 
                         std::vector<ContactPoint*> &contacts, std::vector<ContactObject*> &objects) 
{
  pinocchio::forwardKinematics(model, data, q, v);
  pinocchio::computeJointJacobians(model, data);
  pinocchio::updateFramePlacements(model, data);
  /*!< loops over all contacts and objects to detect contacts and update contact positions*/
  
  int newActive = detectContacts_imp(data, contacts, objects);
  CONSIM_START_PROFILER("compute_contact_forces");
  tau_f.setZero();
  for (auto &cp : contacts) {
    if (!cp->active) continue;
    cp->firstOrderContactKinematics(data); /*!<  must be called before computePenetration() it updates cp.v and jacobian*/   
    cp->optr->computePenetration(*cp); 
    cp->optr->contact_model_->computeForce(*cp);
    tau_f.noalias() += cp->world_J_.transpose() * cp->f; 
    // if (contactChange_){
    //     std::cout<<cp->name_<<" p ["<< cp->x.transpose() << "] v ["<< cp->v.transpose() << "] f ["<<  cp->f.transpose() <<"]"<<std::endl; 
    //   }
  }
  CONSIM_STOP_PROFILER("compute_contact_forces");
  return newActive;
}

void integrateState(const pinocchio::Model &model, const Eigen::VectorXd &x, const Eigen::VectorXd &dx, 
                    double dt, Eigen::VectorXd &xNext)
{
  pinocchio::integrate(model, x.head(model.nq), dx.head(model.nv) * dt, xNext.head(model.nq));
  xNext.tail(model.nv) = x.tail(model.nv) + dx.tail(model.nv) * dt;
}

void differenceState(const pinocchio::Model &model, const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, 
                      Eigen::VectorXd &dx)
{
  pinocchio::difference(model, x0.head(model.nq), x1.head(model.nq), dx.head(model.nv));
  dx.tail(model.nv) = x1.tail(model.nv) - x0.tail(model.nv);
}

void DintegrateState(const pinocchio::Model &model, const Eigen::VectorXd &x, const Eigen::VectorXd &dx, 
                      double dt, Eigen::MatrixXd &J)
{
  pinocchio::dIntegrate(model, x.head(model.nq), dx.head(model.nv) * dt, J.topLeftCorner(model.nv, model.nv), pinocchio::ArgumentPosition::ARG1);
  J.bottomRightCorner(model.nv, model.nv) = MatrixXd::Identity(model.nv, model.nv); // * dt;
  J.topRightCorner(model.nv, model.nv).setZero();
  J.bottomLeftCorner(model.nv, model.nv).setZero();
}

void DdifferenceState_x0(const pinocchio::Model &model, const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, 
                        Eigen::MatrixXd &J)
{
  pinocchio::dDifference(model, x0.head(model.nq), x1.head(model.nq), J.topLeftCorner(model.nv, model.nv), pinocchio::ArgumentPosition::ARG0);
  J.bottomRightCorner(model.nv, model.nv) = -MatrixXd::Identity(model.nv, model.nv);
  J.topRightCorner(model.nv, model.nv).setZero();
  J.bottomLeftCorner(model.nv, model.nv).setZero();
}

void DdifferenceState_x1(const pinocchio::Model &model, const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, 
                        Eigen::MatrixXd &J)
{
  pinocchio::dDifference(model, x0.head(model.nq), x1.head(model.nq), J.topLeftCorner(model.nv, model.nv), pinocchio::ArgumentPosition::ARG1);
  J.bottomRightCorner(model.nv, model.nv).setIdentity(model.nv, model.nv);
  J.topRightCorner(model.nv, model.nv).setZero();
  J.bottomLeftCorner(model.nv, model.nv).setZero();
}

}  // namespace consim 
