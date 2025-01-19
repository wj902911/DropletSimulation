#pragma once

#include <cuda_runtime.h>
#include <SimParams_GPU.h>

struct Droplet_GPU
{
public:
	double2 pos;
	double2 prevpos;
	double2 vel;
	//double2 force;
	double mass = 0.0;
	bool fixed = false;
	bool inert = false;
	double radius = 0.0;
	double prevradius = 0.0;
	double sub =0.0;

	uint32_t uid = -1;

	__device__ Droplet_GPU() = default;

	__device__ Droplet_GPU(double2 pos,
		                    double radius, 
		                    double mass, 
		                    double sub, 
		                      bool isFixed, 
		                      bool isInert,
		                  uint32_t uid)
	: pos(pos), 
	  prevpos(pos), 
	  radius(radius), 
	  prevradius(radius),
	  mass(mass), 
	  sub(sub), 
	  fixed(isFixed), 
	  inert(isInert),
	  uid(uid)
	{
		vel = make_double2(0.0, 0.0);
		//force = make_double2(0.0, 0.0);
	}

	__device__ double getMass() const
	{
		return mass;
	}
};

struct Spring_GPU
{
public:
	int p1 = 0;
	int p2 = 0;
	uint32_t uid1 = -1;
	uint32_t uid2 = -1;
	double stiffness = 0.0;
	double mass = 0.0;
	double length = 0.0;
	bool snapped = false;

	double3 pos = make_double3(0.0, 0.0, 0.0);
	double rot = 0.0;
	double lam = 0.0;
	double initialLam=0.0;
	double2 force = make_double2(0.0, 0.0);
	bool isInTension;

	__device__ Spring_GPU() = default;

	__device__
		Spring_GPU(int a, 
			       int b, 
			       double stf, 
			       double3 q, 
			       double r, 
			       double lam, 
			       double iniLam = 1.0, 
			       double2 f = make_double2(0.0, 0.0), 
			       bool isTension = 1, 
			       uint32_t uid1 = -1, 
			       uint32_t uid2 = -1, 
			       double mss = 0.0,
			       double length = 0.0)
		: p1(a), p2(b), pos(q), rot(r), lam(lam), stiffness(stf), force(f), uid1(uid1), uid2(uid2), initialLam(iniLam), isInTension(isTension), mass(mss), length(length)
	{
	}

	__device__ void processSpring(const double* q,
		                          const double* qprev,
		                          const double* r,
		                          const double* C,
		                          const SimParameters_GPU& params,
		                          double* F,
		                          double* J);

	__device__ bool isSnapped() const
	{
		return snapped;
	}
};
