#include "dropletsSimObjects_GPU.h"
#include <eigen/Core>

__device__ void Spring_GPU::processSpring(const double* q, 
	                                      const double* qprev, 
	                                      const double* r, 
	                                      const double* C, 
	                                      const SimParameters_GPU& params,
	                                      double* F, 
	                                      double* J)
{
	using namespace Eigen;
	Vector2d x1(q[2 * p1], q[2 * p1 + 1]);
	Vector2d x2(q[2 * p2], q[2 * p2 + 1]);
	double r1 = r[p1];
	double r2 = r[p2];
	double d = (x2 - x1).norm();
	if (params.canSnap)
	{
		if (d > params.toltalSeperationLengthRatio * (r1 + r2))
		{
			snapped = true;
			return;
		}
	}
	// Spring force
	double resRatio = params.SpringRestLengthRatio;
	Vector2d localF = stiffness * (d - resRatio * (r1 + r2)) / d * (x2 - x1);
	atomicAdd(&F[2 * p1], localF[0]);
	atomicAdd(&F[2 * p1 + 1], localF[1]);
	atomicAdd(&F[2 * p2], -localF[0]);
	atomicAdd(&F[2 * p2 + 1], -localF[1]);
	// Damping force
	if (params.springDampingEnabled)
	{
		Vector2d x1prev(qprev[2 * p1], qprev[2 * p1 + 1]);
		Vector2d x2prev(qprev[2 * p2], qprev[2 * p2 + 1]);
		Vector2d relvel = (x2 - x2prev) / params.timeStep - (x1 - x1prev) / params.timeStep;
		localF = params.dampingStiffness * relvel;
		atomicAdd(&F[2 * p1], localF[0]);
		atomicAdd(&F[2 * p1 + 1], localF[1]);
		atomicAdd(&F[2 * p2], -localF[0]);
		atomicAdd(&F[2 * p2 + 1], -localF[1]);
	}
	if (d < (r1 + r2) && params.flowEnabled)
	{	
		// Flow
		double C1 = C[p1];
		double C2 = C[p2];
		double A = M_PI * (r1 * r1 - (d * d + r1 * r1 - r2 * r2) * (d * d + r1 * r1 - r2 * r2) / (4 * d * d));
		double D = params.permeability;
		double flow = D * A * (C1 - C2);
		atomicAdd(&J[p1], flow);
		atomicAdd(&J[p2], -flow);
	}
}
