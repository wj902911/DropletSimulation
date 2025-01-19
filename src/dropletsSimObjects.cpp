#include "dropletsSimObjects.h"
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void Spring::processSpringForce(const Eigen::VectorXd& q,
                                const Eigen::VectorXd& r,
                                const SimParameters& params,
                                Eigen::Ref<Eigen::VectorXd> F) const
{
    using namespace Eigen;

    Vector2d x1 = q.segment<2>(2 * p1);
    Vector2d x2 = q.segment<2>(2 * p2);
    double r1 = r[p1];
    double r2 = r[p2];
    double resRatio = params.SpringRestLengthRatio;
    double dist = (x2 - x1).norm();
    Vector2d localF = stiffness * (dist - resRatio * (r1 + r2)) / dist * (x2 - x1);
    F.segment<2>(2 * p1) += localF;
    F.segment<2>(2 * p2) -= localF;
}

void Spring::processDampingForce(const Eigen::VectorXd& q,
                                 const Eigen::VectorXd& qprev,
                                 const SimParameters& params,
                                 Eigen::Ref<Eigen::VectorXd> F) const
{
    using namespace Eigen;
    Vector2d x1 = q.segment<2>(2 * p1);
    Vector2d x2 = q.segment<2>(2 * p2);
    Vector2d x1prev = qprev.segment<2>(2 * p1);
    Vector2d x2prev = qprev.segment<2>(2 * p2);

    Vector2d relvel = (x2 - x2prev) / params.timeStep - (x1 - x1prev) / params.timeStep;
    Vector2d localF = params.dampingStiffness * relvel;
    F.segment<2>(2 * p1) += localF;
    F.segment<2>(2 * p2) -= localF;
}

void Spring::processSpringAll(const Eigen::VectorXd& q, 
                              const Eigen::VectorXd& qprev, 
                              const Eigen::VectorXd& r, 
                              const Eigen::VectorXd& C, 
                              const SimParameters& params, 
                              Eigen::Ref<Eigen::VectorXd> F, 
                              Eigen::Ref<Eigen::VectorXd> J)
{
    using namespace Eigen;
    Vector2d x1 = q.segment<2>(2 * p1);
    Vector2d x2 = q.segment<2>(2 * p2);
    Vector2d x1prev = qprev.segment<2>(2 * p1);
    Vector2d x2prev = qprev.segment<2>(2 * p2);
    double r1 = r[p1];
    double r2 = r[p2];
    double C1 = C[p1];
    double C2 = C[p2];
    double d = (x2 - x1).norm();
    if (d > r1 + r2) 
    { 
        snapped = true;
        return; 
    }
    // Spring force
    double resRatio = params.SpringRestLengthRatio;
    Vector2d localF = stiffness * (d - resRatio * (r1 + r2)) / d * (x2 - x1);
    F.segment<2>(2 * p1) += localF;
    F.segment<2>(2 * p2) -= localF;
    // Damping force
    Vector2d relvel = (x2 - x2prev) / params.timeStep - (x1 - x1prev) / params.timeStep;
    localF = params.dampingStiffness * relvel;
    F.segment<2>(2 * p1) += localF;
    F.segment<2>(2 * p2) -= localF;
    // Flow
    double A = M_PI * (r1 * r1 - (d * d + r1 * r1 - r2 * r2) * (d * d + r1 * r1 - r2 * r2) / (4 * d * d));
    double D = params.permeability;
    double flow = D * A * (C1 - C2);
    J[p1] += flow;
    J[p2] -= flow;
}

void Spring::processFlow(const Eigen::VectorXd& q,
                         const Eigen::VectorXd& r,
                         const Eigen::VectorXd& C,
                         const SimParameters& params,
                         Eigen::Ref<Eigen::VectorXd> J) const
{
    using namespace Eigen;
	Vector2d x1 = q.segment<2>(2 * p1);
	Vector2d x2 = q.segment<2>(2 * p2);
	double r1 = r[p1];
	double r2 = r[p2];
    double C1 = C[p1];
    double C2 = C[p2];
	double d = (x2 - x1).norm();
    if (d > r1 + r2) return;
    double A = M_PI * (r1 * r1 - (d * d + r1 * r1 - r2 * r2) * (d * d + r1 * r1 - r2 * r2) / (4 * d * d));
    double D = params.permeability;
    double flow = D * A * (C1 - C2);
    J[p1] += flow;
    J[p2] -= flow;
}