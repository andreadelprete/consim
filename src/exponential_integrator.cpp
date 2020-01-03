#include "consim/exponential_integrator.hpp"


namespace consim{
ExponentialIntegrator::ExponentialIntegrator(double dt, int nc, const pinocchio::Model &model,
                                             pinocchio::Data &data) : dt_(dt), nc_(nc), model_(&model), data_(&data),
{
    nk_ = 3 * nc_;
    f_.resize(nk_); f_.setZero();
    p0_.resize(nk_); p0_.setZero();
    p_.resize(nk_); p_.setZero();
    dp_.resize(nk_); dp_.setZero();
    Jc_.resize(nk_, model.nv); Jc_.setZero();
    dJv_.resize(nk_); dJv_.setZero();
    a_.resize(2 * nk_); a_.setZero();
    A.resize(2 * nk_, 2 * nk_); A.setZero();
    K.resize(nk_, nk_); K.setZero();
    B.resize(nk_, nk_); B.setZero();
    D.resize(2*nk_, nk_); D.setZero();
};

Eigen::MatrixXd ExponentialIntegrator::matrixExponential(const Eigen::MatrixXd &mat){};
Eigen::VectorXd ExponentialIntegrator::matExpTimesVector(const Eigen::MatrixXd &mat, const Eigen::VectorXd &vec){};

Eigen::VectorXd ExponentialIntegrator::computeXt(const Eigen::MatrixXd &mat, const Eigen::VectorXd &b, const Eigen::VectorXd &x0, 
                                            const double &T, bool invertableA){};

Eigen::VectorXd ExponentialIntegrator::computeIntegralXt(const Eigen::MatrixXd &mat, const Eigen::VectorXd &b, const Eigen::VectorXd &x0,
                                                         const double &T, bool invertableA){};
Eigen::VectorXd ExponentialIntegrator::computeDoubleIntegralXt(const Eigen::MatrixXd &mat, const Eigen::VectorXd &b, const Eigen::VectorXd &x0, 
                                        const double &T, bool compute_integral, bool invertableA){};
std::vector<Eigen::VectorXd> ExponentialIntegrator::computeXtAndIntegrals(const Eigen::MatrixXd &mat, const Eigen::VectorXd &b, 
                                        const Eigen::VectorXd &x0, const double &T){};

} // namespace consim