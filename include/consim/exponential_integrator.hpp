#pragma once
#include <Eigen/Eigen>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace consim{

    class ExponentialIntegrator {
        public:
            ExponentialIntegrator(double dt = 0., int nc = 0, const pinocchio::Model &model,
                                  pinocchio::Data &data);
            ~ExponentialIntegrator(){};
            Eigen::MatrixXd matrixExponential(const Eigen::MatrixXd &mat);
            Eigen::VectorXd matExpTimesVector(const Eigen::MatrixXd &mat, const Eigen::VectorXd &vec);

            Eigen::VectorXd computeXt(const Eigen::MatrixXd &mat, const Eigen::VectorXd &b, const Eigen::VectorXd &x0, const double &T, bool invertableA = false);

            Eigen::VectorXd computeIntegralXt(const Eigen::MatrixXd &mat, const Eigen::VectorXd &b, const Eigen::VectorXd &x0,
                                              const double &T, bool invertableA = false);
            Eigen::VectorXd computeDoubleIntegralXt(const Eigen::MatrixXd &mat, const Eigen::VectorXd &b, const Eigen::VectorXd &x0, const double &T,
                                                    bool compute_integral = false, bool invertableA = false);
            std::vector <Eigen::VectorXd> computeXtAndIntegrals(const Eigen::MatrixXd &mat, const Eigen::VectorXd &b, const Eigen::VectorXd &x0, const double &T);

        private:
            double dt_;
            int nc_; 
            int nk_; 
            Eigen::VectorXd f_;  // total force 
            Eigen::MatrixXd Jc_; // contact jacobian for all contacts 
            Eigen::VectorXd p0_; // reference position for contact 
            Eigen::VectorXd p_; // current contact position 
            Eigen::VectorXd dp_; // contact velocity 
            // contact acceleration components 
            Eigen::VectorXd dJv_; 
            Eigen::VectorXd a_; 
            // keep the stiffness/damping matrices fixed to the total size of contact points
            // worry about tracking index of Active sub-blocks later
            Eigen::MatrixXd K;
            Eigen::MatrixXd B;
            Eigen::MatrixXd D;
            Eigen::MatrixXd A; 
            //
            const pinocchio::Model *model_;
            pinocchio::Data *data_;
    }

}
