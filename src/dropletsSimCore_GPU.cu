#include "dropletsSimCore_GPU.h"
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>
#include <thrust/remove.h>
#include <cmath>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>

struct is_outside_domain_functor
{
	double4 domainBoundingBox;

	is_outside_domain_functor(double4 domainBoundingBox) : domainBoundingBox(domainBoundingBox) {}

	__device__ bool operator()(const Droplet_GPU& droplet) const
	{
		return droplet.pos.x < domainBoundingBox.x || droplet.pos.x > domainBoundingBox.y || droplet.pos.y < domainBoundingBox.z || droplet.pos.y > domainBoundingBox.w;
	}

};

struct is_snapped_functor
{
	__device__ bool operator()(const Spring_GPU& spring) const
	{
		return spring.snapped;
	}
};

struct spring_period_functor
{
	__device__ double operator()(const Spring_GPU& spring) const
	{
		return M_PI * sqrt(spring.mass / spring.stiffness);
	}
};

struct get_position_functor
{
	__device__ thrust::pair<double, double> operator()(const Droplet_GPU& droplet) const
	{
		return thrust::make_pair(droplet.pos.x, droplet.pos.y);
	}
};

struct get_uid_functor
{
	__device__ uint32_t operator()(const Droplet_GPU& droplet) const
	{
		return droplet.uid;
	}
};

struct get_position_x_functor
{
	__device__ double operator()(const Droplet_GPU& droplet) const
	{
		return droplet.pos.x;
	}
};

struct get_position_y_functor
{
	__device__ double operator()(const Droplet_GPU& droplet) const
	{
		return droplet.pos.y;
	}
};

struct get_UID_functor
{
	__device__ uint32_t operator()(const Droplet_GPU& droplet) const
	{
		return droplet.uid;
	}
};

struct get_spring_config_functor
{
	__device__ thrust::tuple<double3, double, double, double> operator()(const Spring_GPU& spring) const
	{
		int sign =  spring.isInTension ? 1 : -1;
		double force = sqrt(spring.force.x * spring.force.x + spring.force.y * spring.force.y)* sign;
		return thrust::make_tuple(spring.pos, spring.rot, spring.lam * spring.initialLam, force);
	}
};

struct get_position_as_d3_functor
{
	__device__ double3 operator()(const Droplet_GPU & droplet) const
	{
		return make_double3(droplet.pos.x, droplet.pos.y, 0);
	}
};

struct min_pair_functor
{
	__device__ thrust::pair<double, double> operator()(const thrust::pair<double, double>& a,
		const thrust::pair<double, double>& b) const
	{
		return thrust::make_pair(thrust::min(a.first, b.first), thrust::min(a.second, b.second));
	}
};

struct max_pair_functor
{
	__device__ thrust::pair<double, double> operator()(const thrust::pair<double, double>& a,
		const thrust::pair<double, double>& b) const
	{
		return thrust::make_pair(thrust::max(a.first, b.first), thrust::max(a.second, b.second));
	}
};

struct sum_pair_functor
{
	__device__ thrust::pair<double, double> operator()(const thrust::pair<double, double>& a,
		const thrust::pair<double, double>& b) const
	{
		return thrust::make_pair(a.first + b.first, a.second + b.second);
	}
};

struct velocity_magnitude_functor
{
	__device__ double operator()(const Droplet_GPU& droplet) const
	{
		return sqrt(droplet.vel.x * droplet.vel.x + droplet.vel.y * droplet.vel.y);
	}
};

struct force_magnitude_functor
{
	__device__ double operator()(const double2& force) const
	{
		return sqrt(force.x * force.x + force.y * force.y);
	}
};

struct concentration_functor
{
	__device__ double operator()(const Droplet_GPU& droplet) const
	{
		return droplet.sub / (4. / 3. * M_PI * droplet.radius * droplet.radius * droplet.radius);
	}
};

struct radius_functor
{
	__device__ double operator()(const Droplet_GPU& droplet) const
	{
		return droplet.radius;
	}
};

struct radius_increment_functor
{
	__device__ double operator()(const Droplet_GPU& droplet) const
	{
		return droplet.radius - droplet.prevradius;
	}
};

struct kinetic_energy_functor
{
	__device__ double operator()(const Droplet_GPU& droplet) const
	{
		double velocity_magnitude = sqrt(droplet.vel.x * droplet.vel.x + droplet.vel.y * droplet.vel.y);
		return 0.5 * droplet.mass * velocity_magnitude * velocity_magnitude; // KE = 1/2 * m * v^2
	}
};

struct DropletUIDMatcher
{
	uint32_t uid_to_match;

	DropletUIDMatcher(uint32_t uid) : uid_to_match(uid) {}

	__device__ bool operator()(const Droplet_GPU& droplet) const
	{
		return droplet.uid == uid_to_match;
	}
};

__global__ void initDroplets(Droplet_GPU* droplets, const SimParameters_GPU params)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int numDroplets = params.numDroplets_x * params.numDroplets_y;
	if (idx >= params.numDroplets_x || idy >= params.numDroplets_y)
		return;
	double spacing = params.initialDropletRadius * 2 * params.SpringRestLengthRatio;
	double x = 0, y = 0;
	switch (params.distributionPattern)
	{
	case 1://pattern 1
		x = idx * spacing;
		y = idy * spacing;
		break;
	case 2://pattern 2
		if (idy % 2 == 0)
			x = idx * spacing;
		else
			x = idx * spacing + spacing / 2;
		y = idy * spacing * cosf(M_PI / 6);
		break;
	case 3: case 4: //maunally set
		x= params.dropletPositions[2*(idy * params.numDroplets_x + idx)];
		y= params.dropletPositions[2*(idy * params.numDroplets_x + idx)+1];
		break;
	case 6:
		x = idx * params.initialDropletRadius * 2;
		y = idy * params.initialDropletRadius * 2;
		break;
	}
	/*
#if 1
	if (idy % 2 == 0)
		x = idx * spacing;
	else
		x = idx * spacing + spacing / 2;
	    y = idy * spacing * cosf(M_PI / 6);
#else
	x = idx * spacing;
	y = idy * spacing;
#endif
	*/

	double2 pos = make_double2(x, y);
	double mass = params.dropletMass;
	if (params.distributionPattern==4)
	{
		double radius = params.dropletRadii[idy * params.numDroplets_x + idx];
		mass = 4. / 3. * M_PI * radius * radius * radius * params.dropletDensity;
	}
	bool isFixed = false;
	switch (params.distributionPattern)
	{
	case 1: case 2:
	{
		if (params.fixDroplets)
		{
#if 0
			if ((idx == 0 && idy % 2 == 0) || (idx == params.numDroplets_x - 1 && idy % 2 != 0))
#else
			if (idx == 0 || idx == params.numDroplets_x - 1)
#endif
			{
				//printf("droplet_%d_%d is fixed\n", idx, idy);
				mass = INFINITY;
				isFixed = true;
			}
			break;
		}
	}
	case 3: case 4:
	{
		if (params.fixeds[idy * params.numDroplets_x + idx])
		{
			//printf("droplet_%d_%d is fixed\n", idx, idy);
			mass = INFINITY;
			isFixed = true;
		}
		break;
	}
	case 6:
	{
		break;
	}
	}

#if 1
	if (idy < params.numDroplets_y / 2)
	{
		droplets[idy * params.numDroplets_x + idx] = Droplet_GPU(pos,
			                                                     params.initialDropletRadius,
			                                                     mass,
			                                                     params.upperSubValue,
			                                                     isFixed,
			                                                     false,
			                                                     idy * params.numDroplets_x + idx);
		//printf("droplet_%d Sub: %f\n", idy * params.numDroplets_x + idx, params.upperSubValue);
	}
	else
	{
		droplets[idy * params.numDroplets_x + idx] = Droplet_GPU(pos,
			                                                     params.initialDropletRadius,
			                                                     mass,
			                                                     params.lowerSubValue,
			                                                     isFixed,
			                                                     false,
			                                                     idy * params.numDroplets_x + idx);
		//printf("droplet_%d Sub: %f\n", idy * params.numDroplets_x + idx, params.lowerSubValue);
	}
#else
	switch (params.distributionPattern)
	{
	case 1: case 2: case 6:
		droplets[idy * params.numDroplets_x + idx] = Droplet_GPU(pos,
														         params.initialDropletRadius,
														         mass,
														         params.lowerSubValue,
														         isFixed,
														         false,
														         idy * params.numDroplets_x + idx);
		break;
	case 3:
		droplets[idy * params.numDroplets_x + idx] = Droplet_GPU(pos,
														         params.initialDropletRadius,
														         mass,
														         params.initialDropletsSubValue[idy * params.numDroplets_x + idx],
														         isFixed,
														         false,
														         idy * params.numDroplets_x + idx);
		break;
	case 4:
		droplets[idy * params.numDroplets_x + idx] = Droplet_GPU(pos,
														         params.dropletRadii[idy * params.numDroplets_x + idx],
														         mass,
														         params.lowerSubValue,
														         isFixed,
														         false,
														         idy * params.numDroplets_x + idx);
		break;
	}
	
	
	if (params.loadType==LOAD_TYPE::velocity)
	{
		double2 vel = make_double2(params.initialDropletsVelocity[2 * (idy * params.numDroplets_x + idx)], params.initialDropletsVelocity[2 * (idy * params.numDroplets_x + idx) + 1]);
		droplets[idy * params.numDroplets_x + idx].vel = vel;
	}
	
	
#endif
	//printf("droplet_%d pos: %f, %f\n", idy * params.numDroplets_x + idx, pos.x, pos.y);
	
}

__global__ void initDroplets_1D(Droplet_GPU* droplets, const SimParameters_GPU params)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= params.numDroplets)
		return;
	double x = params.dropletPositions[2 * idx];
	double y = params.dropletPositions[2 * idx + 1];
	double2 pos = make_double2(x, y);
	double radius = params.dropletRadii[idx];
	double mass = 4. / 3. * M_PI * radius * radius * radius * params.dropletDensity;
	bool isFixed = false;
	if (params.fixeds[idx])
	{
		mass = INFINITY;
		isFixed = true;
	}
	droplets[idx] = Droplet_GPU(pos, radius, mass, params.lowerSubValue, isFixed, false, idx);
}

__global__ void assignToBins(Droplet_GPU* droplets, 
	                                  int num_droplets, 
	                                 int* droplets_bin_indices, 
	                               double cellSize, 
	                                  int numCellsX, 
	                                  int numCellsY,
	                              double2 gridOrigin)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	double2 relativePos = make_double2(droplets[idx].pos.x - gridOrigin.x, droplets[idx].pos.y - gridOrigin.y);
	int2 cellIdx = make_int2(relativePos.x / cellSize, relativePos.y / cellSize);
	if (cellIdx.x < 0 || cellIdx.x >= numCellsX || cellIdx.y < 0 || cellIdx.y >= numCellsY)
	{
		droplets_bin_indices[idx] = -1;
		return;
	}
	droplets_bin_indices[idx] = cellIdx.y * numCellsX + cellIdx.x;
}

__global__ void identifyBinBoundaries(int* droplets_bin_indices, int num_droplets, int* bin_start, int* bin_end)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	if (idx == 0 || droplets_bin_indices[idx] != droplets_bin_indices[idx - 1])
		bin_start[droplets_bin_indices[idx]] = idx;
	if (idx == num_droplets - 1 || droplets_bin_indices[idx] != droplets_bin_indices[idx + 1])
		bin_end[droplets_bin_indices[idx]] = idx + 1;
}

__global__ void getConfiguration(const Droplet_GPU* droplets, 
	                                      const int num_droplets, 
	                                        double* q, 
	                                        double* qprev, 
	                                        double* v, 
	                                        double* r,
	                                        double* rprev,
	                                        double* C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;

	q[2 * idx] = droplets[idx].pos.x;
	q[2 * idx + 1] = droplets[idx].pos.y;

	if (qprev != nullptr)
	{
		qprev[2 * idx] = droplets[idx].prevpos.x;
		qprev[2 * idx + 1] = droplets[idx].prevpos.y;
	}

	if (v != nullptr)
	{
		v[2 * idx] = droplets[idx].vel.x;
		v[2 * idx + 1] = droplets[idx].vel.y;
	}

	r[idx] = droplets[idx].radius;

	if (rprev != nullptr)
		rprev[idx] = droplets[idx].prevradius;

	if (C != nullptr)
		C[idx] = droplets[idx].sub / (4. / 3. * M_PI * r[idx] * r[idx] * r[idx]);
}

__global__ void processGravityForce(const Droplet_GPU* droplets, 
	                                         const int num_droplets, 
	                           const SimParameters_GPU params, 
	                                           double* F)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	if (!droplets[idx].fixed)
		F[2 * idx + 1] += droplets[idx].mass * params.gravityG;
}

__global__ void processGlobalDampingForce(const double* v, 
	                                          const int num_droplets, 
	                            const SimParameters_GPU params, 
	                                            double* F)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	F[2 * idx] -= params.globalDamping * v[2 * idx];
	F[2 * idx + 1] -= params.globalDamping * v[2 * idx + 1];
}

__global__ void processFloorForce(const double* q, 
	                              const double* v, 
	                              const double* r, 
	                                  const int num_droplets, 
	                    const SimParameters_GPU params, 
	                                    double* F)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	double y = q[2 * idx + 1];
	double r_ = r[idx];
	if (y-r_<params.floorY)
	{
		double y_ = params.floorY + r_;
		double v_ = v[2 * idx + 1];
		double k = params.springStiffness;
		double d = y_-y;
		double f = k * d - params.dampingStiffness * v_;
		atomicAdd(&F[2 * idx + 1], f);
	}
	if (params.wallEnabled)
	{
		double x = q[2 * idx];
		if (x - r_ < params.leftWallX)
		{
			double x_ = params.leftWallX + r_;
			double v_ = v[2 * idx];
			double k = params.springStiffness;
			double d = x_ - x;
			double f = k * d - params.dampingStiffness * v_;
			atomicAdd(&F[2 * idx], f);
		}
		if (x + r_ > params.rightWallX)
		{
			double x_ = params.rightWallX - r_;
			double v_ = v[2 * idx];
			double k = params.springStiffness;
			double d = x_ - x;
			double f = k * d - params.dampingStiffness * v_;
			atomicAdd(&F[2 * idx], f);
		}
	}
}


__global__ void getNumberOfConnections(const Droplet_GPU* droplets_,
											   const int* droplets_bin_indices,
											   const int* bin_start,
											   const int* bin_end,
												const int num_droplets,
											 const double cellSize,
												const int numCellsX,
												const int numCellsY,
								  const SimParameters_GPU params,
													 int* numConnection,
	                                                 int* numOutputConnections)
{
	using namespace Eigen;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	int hashIdx = droplets_bin_indices[idx];
	//printf("droplet_%d hashIdx: %d\n", idx, hashIdx);
	int2 cellIdx = make_int2(hashIdx % numCellsX, (hashIdx / numCellsX) % numCellsY);
	Vector2d x1(droplets_[idx].pos.x, droplets_[idx].pos.y);
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int2 neighborCellIdx = make_int2(cellIdx.x + i, cellIdx.y + j);
			if (neighborCellIdx.x < 0 || neighborCellIdx.x >= numCellsX || neighborCellIdx.y < 0 || neighborCellIdx.y >= numCellsY)
				continue;
			int binIdx = neighborCellIdx.y * numCellsX + neighborCellIdx.x;
			if (bin_start[binIdx] == -1)
				continue;
			for (int k = bin_start[binIdx]; k < bin_end[binIdx]; k++)
			{
				//printf("compare between droplets: %d and %d\n", idx, k);
				if (k > idx)
				{
					Vector2d x2(droplets_[k].pos.x, droplets_[k].pos.y);
					double r1 = droplets_[idx].radius;
					double r2 = droplets_[k].radius;
					double d = (x2 - x1).norm();
					//printf("droplet_%d and %d, r1+r2: %e d: %e\n", droplets_[idx].uid, droplets_[k].uid, r1+r2, d);
					if (d < params.toltalSeperationLengthRatio * (r1 + r2))
					{
						//printf("droplet_%d and droplet_%d are connected\n", droplets_[idx].uid, droplets_[k].uid);
						atomicAdd(&numConnection[0], 1);
						for (int l = 0; l < params.outputSectionDropletUid.size(); l++)
						{
							if (droplets_[idx].uid == params.outputSectionDropletUid[l])
							{
								for (int m = 0;m<params.outputSectionDropletUid2.size();m++)
								{
									if (droplets_[k].uid == params.outputSectionDropletUid2[m])
									{
										atomicAdd(&numOutputConnections[0], 1);
										break;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

/*
__global__ void getNumberOfConnections_fineModel(const Droplet_GPU* droplets_,
	                                                     const int* droplets_bin_indices,
	                                                     const int* bin_start,
	                                                     const int* bin_end,
	                                                      const int num_droplets,
	                                                   const double cellSize,
	                                                      const int numCellsX,
	                                                      const int numCellsY,
	                                        const SimParameters_GPU params,
	                                                           int* numConnection,
	                                                           int* numOutputConnections)
{
	using namespace Eigen;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	int hashIdx = droplets_bin_indices[idx];
	//printf("droplet_%d hashIdx: %d\n", idx, hashIdx);
	int2 cellIdx = make_int2(hashIdx % numCellsX, (hashIdx / numCellsX) % numCellsY);
	Vector2d x1(droplets_[idx].pos.x, droplets_[idx].pos.y);
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int2 neighborCellIdx = make_int2(cellIdx.x + i, cellIdx.y + j);
			if (neighborCellIdx.x < 0 || neighborCellIdx.x >= numCellsX || neighborCellIdx.y < 0 || neighborCellIdx.y >= numCellsY)
				continue;
			int binIdx = neighborCellIdx.y * numCellsX + neighborCellIdx.x;
			if (bin_start[binIdx] == -1)
				continue;
			for (int k = bin_start[binIdx]; k < bin_end[binIdx]; k++)
			{
				//printf("compare between droplets: %d and %d\n", idx, k);
				if (k > idx)
				{
					Vector2d x2(droplets_[k].pos.x, droplets_[k].pos.y);
					double r1 = droplets_[idx].radius;
					double r2 = droplets_[k].radius;
					double d = (x2 - x1).norm();
					//printf("droplet_%d and %d, r1+r2: %e d: %e\n", droplets_[idx].uid, droplets_[k].uid, r1+r2, d);
					if (d < params.ir * params.initialDropletRadius)
					{
						//printf("droplet_%d and droplet_%d are connected\n", droplets_[idx].uid, droplets_[k].uid);
						atomicAdd(&numConnection[0], 1);
						for (int l = 0; l < params.outputSectionDropletUid.size(); l++)
						{
							if (droplets_[idx].uid == params.outputSectionDropletUid[l])
							{
								for (int m = 0; m < params.outputSectionDropletUid2.size(); m++)
								{
									if (droplets_[k].uid == params.outputSectionDropletUid2[m])
									{
										atomicAdd(&numOutputConnections[0], 1);
										break;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
*/

__global__ void setupConnections(const Droplet_GPU* droplets_,
										 const int* droplets_bin_indices,
										 const int* bin_start,
										 const int* bin_end,
										  const int num_droplets,
									   const double cellSize,
										  const int numCellsX,
										  const int numCellsY,
							const SimParameters_GPU params,
										       int* numConnections,
	                                           int* numOutputConnections,
	                                           int* output_spring_index,
	                                           int* num_connections_per_droplet,
										Spring_GPU* connectors)
{
	using namespace Eigen;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	int hashIdx = droplets_bin_indices[idx];
	int2 cellIdx = make_int2(hashIdx % numCellsX, (hashIdx / numCellsX) % numCellsY);
	Vector2d x1(droplets_[idx].pos.x, droplets_[idx].pos.y);
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int2 neighborCellIdx = make_int2(cellIdx.x + i, cellIdx.y + j);
			if (neighborCellIdx.x < 0 || neighborCellIdx.x >= numCellsX || neighborCellIdx.y < 0 || neighborCellIdx.y >= numCellsY)
				continue;
			int binIdx = neighborCellIdx.y * numCellsX + neighborCellIdx.x;
			if (bin_start[binIdx] == -1)
				continue;
			for (int k = bin_start[binIdx]; k < bin_end[binIdx]; k++)
			{
				if (k > idx)
				{
					Vector2d x2(droplets_[k].pos.x, droplets_[k].pos.y);
					double r1 = droplets_[idx].radius;
					//printf("r1: %f\n", r1);
					double r2 = droplets_[k].radius;
					//printf("r2: %f\n", r2);
					double d = (x2 - x1).norm();
					//printf("droplet_%d and %d, r1+r2: %e d: %e\n", droplets_[idx].uid, droplets_[k].uid, r1 + r2, d);
					if (d < params.toltalSeperationLengthRatio * (r1 + r2))
					{
						//printf("droplet_%d and droplet_%d are connected\n", droplets_[idx].uid, droplets_[k].uid);
						int connectionIdx = atomicAdd(&numConnections[0], 1);
						atomicAdd(&num_connections_per_droplet[idx], 1);
						atomicAdd(&num_connections_per_droplet[k], 1);
						Vector2d unitVec = (x2 - x1) / d;
						//printf("%f, %f\n", unitVec[0], unitVec[1]);
						double rot = atan2(unitVec.y(), unitVec.x());
						//printf("%f\n", rot);
						double lam = d / (params.SpringRestLengthRatio * (r1 + r2));
						double initialLam = (r1 + r2) / (2 * params.initialDropletRadius);
						//printf("%f\n", lam);
						double mass1 = droplets_[idx].mass;
						double mass2 = droplets_[k].mass;
						double reducedMass = mass1 * mass2 / (mass1 + mass2);
						//double restLength = sqrt(r1 * r1 + r2 * r2 + 2 * r1 * r2 * cos(2 * params.theta_eq * M_PI / 180.0));
						double restLength = d;
						double deformation = d - restLength;
						bool isTenstion = deformation > 0;
						Vector2d localF = connectors[idx].stiffness * deformation / d * (x2 - x1);
						//printf("%f", reducedMass);
						//double dampingStiffness = 2 * sqrt(params.springStiffness * reducedMass) * params.springDamplingRatio;
						connectors[connectionIdx] = Spring_GPU(idx, 
							                                   k, 
							                                   params.springStiffness, 
							                                   make_double3(x1[0], x1[1], 0.0), 
							                                   rot, 
							                                   lam, 
							                                   initialLam, 
							                                   make_double2(localF.x(), localF.y()), 
							                                   isTenstion, 
							                                   droplets_[idx].uid, 
							                                   droplets_[k].uid, 
							                                   reducedMass,
							                                   restLength);
						//printf("spring_%d p1: %d, p2: %d , stiffness: %e\n", connectionIdx, idx, k, connectors[connectionIdx].stiffness);
						for (int l = 0; l < params.outputSectionDropletUid.size(); l++)
						{
							if (droplets_[idx].uid == params.outputSectionDropletUid[l])
							{
								for (int m = 0;m<params.outputSectionDropletUid2.size();m++)
								{
									if (droplets_[k].uid == params.outputSectionDropletUid2[m])
									{
										int outputIdx = atomicAdd(&numOutputConnections[0], 1);
										output_spring_index[outputIdx] = connectionIdx;
										//printf("spring_%d p1: %d, p2: %d , uid1: %d, uid2: %d , stiffness: %e\n", connectionIdx, idx, k, droplets_[idx].uid, droplets_[k].uid, connectors[connectionIdx].stiffness);
										break;
									}
								}
							}
						}
						//printf("spring_%d p1: %d, p2: %d , stiffness: %e\n", connectionIdx, idx, k, connectors[connectionIdx].stiffness);
					}
				}
			}
		}
	}
}

__global__ void setupConnections(const double* q, 
	                             const double* r, 
	                                const int* droplets_bin_indices, 
	                                const int* bin_start, 
	                                const int* bin_end, 
	                                 const int num_droplets, 
	                              const double cellSize, 
	                                 const int numCellsX, 
	                                 const int numCellsY, 
	                   const SimParameters_GPU params, 
	                                      int* numConnections, 
	                                      int* num_connections_per_droplet,
	                               Spring_GPU* connectors)
{
	using namespace Eigen;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	int hashIdx = droplets_bin_indices[idx];
	int2 cellIdx = make_int2(hashIdx % numCellsX, (hashIdx / numCellsX) % numCellsY);
	Vector2d x1(q[2 * idx], q[2 * idx + 1]);
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int2 neighborCellIdx = make_int2(cellIdx.x + i, cellIdx.y + j);
			if (neighborCellIdx.x < 0 || neighborCellIdx.x >= numCellsX || neighborCellIdx.y < 0 || neighborCellIdx.y >= numCellsY)
				continue;
			int binIdx = neighborCellIdx.y * numCellsX + neighborCellIdx.x;
			if (bin_start[binIdx] == -1)
				continue;
			for (int k = bin_start[binIdx]; k < bin_end[binIdx]; k++)
			{
				if (k > idx)
				{
					Vector2d x2(q[2 * k], q[2 * k + 1]);
					double r1 = r[idx];
					double r2 = r[k];
					double d = (x2 - x1).norm();
					if (d < params.toltalSeperationLengthRatio * (r1 + r2))
					{
						int connectionIdx = atomicAdd(&numConnections[0], 1);
						atomicAdd(&num_connections_per_droplet[idx], 1);
						atomicAdd(&num_connections_per_droplet[k], 1);
						Vector2d unitVec = (x2 - x1) / d;
						double rot = atan2(unitVec.y(), unitVec.x());
						double resRatio = params.SpringRestLengthRatio;
						double lam = d / (resRatio * (r1 + r2));
						double initialLam = (r1 + r2) / (2 * params.initialDropletRadius);
						double mass1 = 4. / 3. * M_PI * r1 * r1 * r1 * params.dropletDensity;
						double mass2 = 4. / 3. * M_PI * r2 * r2 * r2 * params.dropletDensity;
						double reducedMass = mass1 * mass2 / (mass1 + mass2);
						//double dampingStiffness = 2 * sqrt(params.springStiffness * reducedMass) * params.springDamplingRatio;
						double deformation = d - sqrt(r1 * r1 + r2 * r2 + 2 * r1 * r2 * cos(2 * params.theta_eq * M_PI / 180.0));
						bool isTenstion = deformation > 0;
						Vector2d localF = connectors[idx].stiffness * deformation / d * (x2 - x1);
						connectors[connectionIdx] = Spring_GPU(idx, 
							                                   k, 
							                                   params.springStiffness, 
							                                   make_double3(x1[0], x1[1], 0.0), 
							                                   rot, 
							                                   lam, 
							                                   initialLam, 
							                                   make_double2(localF.x(), localF.y()), 
							                                   isTenstion, 
							                                   -1, 
							                                   -1, 
							                                   reducedMass);
						//printf("spring_%d p1: %d, p2: %d , stiffness: %e\n", connectionIdx, idx, k, connectors[connectionIdx].stiffness);
						//printf("%d ", isTenstion);
						//printf("%d\n", connectors[connectionIdx].isInTension);
					}
				}
			}
		}
	}
}

/*
__global__ void setupConnections_fineModel(const Droplet_GPU* droplets_,
										 const int* droplets_bin_indices,
										 const int* bin_start,
										 const int* bin_end,
										  const int num_droplets,
									   const double cellSize,
										  const int numCellsX,
										  const int numCellsY,
							const SimParameters_GPU params,
										       int* numConnections,
	                                           int* numOutputConnections,
	                                           int* output_spring_index,
	                                           int* num_connections_per_droplet,
										Spring_GPU* connectors)
{
	using namespace Eigen;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	int hashIdx = droplets_bin_indices[idx];
	int2 cellIdx = make_int2(hashIdx % numCellsX, (hashIdx / numCellsX) % numCellsY);
	Vector2d x1(droplets_[idx].pos.x, droplets_[idx].pos.y);
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int2 neighborCellIdx = make_int2(cellIdx.x + i, cellIdx.y + j);
			if (neighborCellIdx.x < 0 || neighborCellIdx.x >= numCellsX || neighborCellIdx.y < 0 || neighborCellIdx.y >= numCellsY)
				continue;
			int binIdx = neighborCellIdx.y * numCellsX + neighborCellIdx.x;
			if (bin_start[binIdx] == -1)
				continue;
			for (int k = bin_start[binIdx]; k < bin_end[binIdx]; k++)
			{
				if (k > idx)
				{
					Vector2d x2(droplets_[k].pos.x, droplets_[k].pos.y);
					double r1 = droplets_[idx].radius;
					//printf("r1: %f\n", r1);
					double r2 = droplets_[k].radius;
					//printf("r2: %f\n", r2);
					double d = (x2 - x1).norm();
					//printf("droplet_%d and %d, r1+r2: %e d: %e\n", droplets_[idx].uid, droplets_[k].uid, r1 + r2, d);
					if (d <  params.ir * params.initialDropletRadius)
					{
						//printf("droplet_%d and droplet_%d are connected\n", droplets_[idx].uid, droplets_[k].uid);
						int connectionIdx = atomicAdd(&numConnections[0], 1);
						atomicAdd(&num_connections_per_droplet[idx], 1);
						atomicAdd(&num_connections_per_droplet[k], 1);
						Vector2d unitVec = (x2 - x1) / d;
						//printf("%f, %f\n", unitVec[0], unitVec[1]);
						double rot = atan2(unitVec.y(), unitVec.x());
						//printf("%f\n", rot);
						double lam = d / (r1 + r2);
						double initialLam = (r1 + r2) / (2 * params.initialDropletRadius);
						//printf("%f\n", lam);
						double mass1 = droplets_[idx].mass;
						double mass2 = droplets_[k].mass;
						double reducedMass = mass1 * mass2 / (mass1 + mass2);
						//double restLength = sqrt(r1 * r1 + r2 * r2 + 2 * r1 * r2 * cos(2 * params.theta_eq * M_PI / 180.0));
						double restLength = d;
						double deformation = d - restLength;
						bool isTenstion = deformation > 0;
						Vector2d localF = connectors[idx].stiffness * deformation / d * (x2 - x1);
						//printf("%f", reducedMass);
						//double dampingStiffness = 2 * sqrt(params.springStiffness * reducedMass) * params.springDamplingRatio;
						connectors[connectionIdx] = Spring_GPU(idx, 
							                                   k, 
							                                   params.springStiffness, 
							                                   make_double3(x1[0], x1[1], 0.0), 
							                                   rot, 
							                                   lam, 
							                                   initialLam, 
							                                   make_double2(localF.x(), localF.y()), 
							                                   isTenstion, 
							                                   droplets_[idx].uid, 
							                                   droplets_[k].uid, 
							                                   reducedMass,
							                                   restLength);
						//printf("spring_%d p1: %d, p2: %d , stiffness: %e\n", connectionIdx, idx, k, connectors[connectionIdx].stiffness);
						for (int l = 0; l < params.outputSectionDropletUid.size(); l++)
						{
							if (droplets_[idx].uid == params.outputSectionDropletUid[l])
							{
								for (int m = 0;m<params.outputSectionDropletUid2.size();m++)
								{
									if (droplets_[k].uid == params.outputSectionDropletUid2[m])
									{
										int outputIdx = atomicAdd(&numOutputConnections[0], 1);
										output_spring_index[outputIdx] = connectionIdx;
										//printf("spring_%d p1: %d, p2: %d , uid1: %d, uid2: %d , stiffness: %e\n", connectionIdx, idx, k, droplets_[idx].uid, droplets_[k].uid, connectors[connectionIdx].stiffness);
										break;
									}
								}
							}
						}
						//printf("spring_%d p1: %d, p2: %d , stiffness: %e\n", connectionIdx, idx, k, connectors[connectionIdx].stiffness);
					}
				}
			}
		}
	}
}

__global__ void setupConnections_fineModel(const double* q,
	                             const double* r, 
	                                const int* droplets_bin_indices, 
	                                const int* bin_start, 
	                                const int* bin_end, 
	                                 const int num_droplets, 
	                              const double cellSize, 
	                                 const int numCellsX, 
	                                 const int numCellsY, 
	                   const SimParameters_GPU params, 
	                                      int* numConnections, 
	                                      int* num_connections_per_droplet,
	                               Spring_GPU* connectors)
{
	using namespace Eigen;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	int hashIdx = droplets_bin_indices[idx];
	int2 cellIdx = make_int2(hashIdx % numCellsX, (hashIdx / numCellsX) % numCellsY);
	Vector2d x1(q[2 * idx], q[2 * idx + 1]);
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int2 neighborCellIdx = make_int2(cellIdx.x + i, cellIdx.y + j);
			if (neighborCellIdx.x < 0 || neighborCellIdx.x >= numCellsX || neighborCellIdx.y < 0 || neighborCellIdx.y >= numCellsY)
				continue;
			int binIdx = neighborCellIdx.y * numCellsX + neighborCellIdx.x;
			if (bin_start[binIdx] == -1)
				continue;
			for (int k = bin_start[binIdx]; k < bin_end[binIdx]; k++)
			{
				if (k > idx)
				{
					Vector2d x2(q[2 * k], q[2 * k + 1]);
					double r1 = r[idx];
					double r2 = r[k];
					double d = (x2 - x1).norm();
					if (d < params.toltalSeperationLengthRatio * (r1 + r2))
					{
						int connectionIdx = atomicAdd(&numConnections[0], 1);
						atomicAdd(&num_connections_per_droplet[idx], 1);
						atomicAdd(&num_connections_per_droplet[k], 1);
						Vector2d unitVec = (x2 - x1) / d;
						double rot = atan2(unitVec.y(), unitVec.x());
						double resRatio = params.SpringRestLengthRatio;
						double lam = d / (resRatio * (r1 + r2));
						double initialLam = (r1 + r2) / (2 * params.initialDropletRadius);
						double mass1 = 4. / 3. * M_PI * r1 * r1 * r1 * params.dropletDensity;
						double mass2 = 4. / 3. * M_PI * r2 * r2 * r2 * params.dropletDensity;
						double reducedMass = mass1 * mass2 / (mass1 + mass2);
						//double dampingStiffness = 2 * sqrt(params.springStiffness * reducedMass) * params.springDamplingRatio;
						double deformation = d - sqrt(r1 * r1 + r2 * r2 + 2 * r1 * r2 * cos(2 * params.theta_eq * M_PI / 180.0));
						bool isTenstion = deformation > 0;
						Vector2d localF = connectors[idx].stiffness * deformation / d * (x2 - x1);
						connectors[connectionIdx] = Spring_GPU(idx, 
							                                   k, 
							                                   params.springStiffness, 
							                                   make_double3(x1[0], x1[1], 0.0), 
							                                   rot, 
							                                   lam, 
							                                   initialLam, 
							                                   make_double2(localF.x(), localF.y()), 
							                                   isTenstion, 
							                                   -1, 
							                                   -1, 
							                                   reducedMass);
						//printf("spring_%d p1: %d, p2: %d , stiffness: %e\n", connectionIdx, idx, k, connectors[connectionIdx].stiffness);
						//printf("%d ", isTenstion);
						//printf("%d\n", connectors[connectionIdx].isInTension);
					}
				}
			}
		}
	}
}
*/

__global__ void processSpringForceAndFlowWithoutCollosionDetection(const double* q,
																   const double* qprev,
																   const double* r,
																   const double* C,
																	   const int num_connectors,
														 const SimParameters_GPU params,
																	 Spring_GPU* connectors,
																		 double* F,
																		 double* J)
{
	using namespace Eigen;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_connectors)
		return;
	//connectors[idx].processSpring(q, qprev, r, C, params, F, J);
	int p1 = connectors[idx].p1;
	int p2 = connectors[idx].p2;
	Vector2d x1(q[2 * p1], q[2 * p1 + 1]);
	Vector2d x2(q[2 * p2], q[2 * p2 + 1]);
	double r1 = r[p1];
	double r2 = r[p2];
	double d = (x2 - x1).norm();
	if (params.canSnap)
	{
		if (d > params.toltalSeperationLengthRatio * (r1 + r2))
		{
			connectors[idx].snapped = true;
			return;
		}
	}
	// Spring force
	Vector2d unitVec = (x2 - x1) / d;
	double rot = atan2(unitVec.y(), unitVec.x());
	//double reslength = sqrt(r1 * r1 + r2 * r2 + 2 * r1 * r2 * cos(2 * params.theta_eq * M_PI / 180.0));
	double lam = d / (params.SpringRestLengthRatio * (r1 + r2));
	double initialLam = (r1 + r2) / (2 * params.initialDropletRadius);
	double deformation = d - connectors[idx].length;
	connectors[idx].isInTension = deformation > 0;
	Vector2d localF = connectors[idx].stiffness * deformation / d * (x2 - x1);
	connectors[idx].pos = make_double3(x1[0], x1[1], 0.0);
	connectors[idx].rot = rot;
	connectors[idx].lam = lam;
	connectors[idx].initialLam = initialLam;
	connectors[idx].force = make_double2(localF.x(), localF.y());
	double mass1 = 4. / 3. * M_PI * r1 * r1 * r1 * params.dropletDensity;
	double mass2 = 4. / 3. * M_PI * r2 * r2 * r2 * params.dropletDensity;
	double reducedMass = mass1 * mass2 / (mass1 + mass2);
	connectors[idx].mass = reducedMass;
	//printf("x1=(%f, %f), x2=(%f, %f)\n", x1.x(), x1.y(), x2.x(), x2.y());
	//printf("r1 = %f, r2 = %f, d = %f, resRatio = %f\n", r1, r2, d, resRatio);
	//printf("spring force: %e, %e\n", localF[0], localF[1]);
	//printf("p1 = %d, p2 = %d\n", p1, p2);
	//printf("\n");
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
		double dampingStiffness = 2 * sqrt(params.springStiffness * reducedMass) * params.springDamplingRatio;
		localF = dampingStiffness * relvel;
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

__global__ void processSpringForceAndFlowWithoutCollosionDetection_osm(const double* q,
	const double* qprev,
	const double* r,
	const double* C,
	const int num_connectors,
	const SimParameters_GPU params,
	Spring_GPU* connectors,
	double* F,
	double* J)
{
	using namespace Eigen;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_connectors)
		return;
	//connectors[idx].processSpring(q, qprev, r, C, params, F, J);
	int p1 = connectors[idx].p1;
	int p2 = connectors[idx].p2;
	Vector2d x1(q[2 * p1], q[2 * p1 + 1]);
	Vector2d x2(q[2 * p2], q[2 * p2 + 1]);
	double r1 = r[p1];
	double r2 = r[p2];
	connectors[idx].length = params.SpringRestLengthRatio * (r1 + r2);
	double d = (x2 - x1).norm();
	if (params.canSnap)
	{
		if (d > params.toltalSeperationLengthRatio * (r1 + r2))
		{
			connectors[idx].snapped = true;
			return;
		}
	}
	// Spring force
	Vector2d unitVec = (x2 - x1) / d;
	double rot = atan2(unitVec.y(), unitVec.x());
	//double reslength = sqrt(r1 * r1 + r2 * r2 + 2 * r1 * r2 * cos(2 * params.theta_eq * M_PI / 180.0));
	double lam = d / (params.SpringRestLengthRatio * (r1 + r2));
	double initialLam = (r1 + r2) / (2 * params.initialDropletRadius);
	double deformation = d - connectors[idx].length;
	connectors[idx].isInTension = deformation > 0;
	Vector2d localF = connectors[idx].stiffness * deformation / d * (x2 - x1);
	connectors[idx].pos = make_double3(x1[0], x1[1], 0.0);
	connectors[idx].rot = rot;
	connectors[idx].lam = lam;
	connectors[idx].initialLam = initialLam;
	connectors[idx].force = make_double2(localF.x(), localF.y());
	double mass1 = 4. / 3. * M_PI * r1 * r1 * r1 * params.dropletDensity;
	double mass2 = 4. / 3. * M_PI * r2 * r2 * r2 * params.dropletDensity;
	double reducedMass = mass1 * mass2 / (mass1 + mass2);
	connectors[idx].mass = reducedMass;
	//printf("x1=(%f, %f), x2=(%f, %f)\n", x1.x(), x1.y(), x2.x(), x2.y());
	//printf("r1 = %f, r2 = %f, d = %f, resRatio = %f\n", r1, r2, d, resRatio);
	//printf("spring force: %e, %e\n", localF[0], localF[1]);
	//printf("p1 = %d, p2 = %d\n", p1, p2);
	//printf("\n");
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
		double dampingStiffness = 2 * sqrt(params.springStiffness * reducedMass) * params.springDamplingRatio;
		localF = dampingStiffness * relvel;
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

__global__ void processSpringForceAndFlow(const double* q,
	                                      const double* qprev,
	                                      const double* r,
	                                      const double* C,
	                                         const int* droplets_bin_indices,
	                                         const int* bin_start,
	                                         const int* bin_end,
	                                          const int num_droplets,
	                                       const double cellSize,
	                                          const int numCellsX,
	                                          const int numCellsY,
	                            const SimParameters_GPU params,
	                                            double* F,
	                                            double* J,
	                                               int* numConnection)
{
	//printf("num_droplets: %d\n", num_droplets);
	using namespace Eigen;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	int hashIdx = droplets_bin_indices[idx];
	//printf("droplet_%d hashIdx: %d\n", idx, hashIdx);
	int2 cellIdx = make_int2(hashIdx % numCellsX, (hashIdx / numCellsX) % numCellsY);
	Vector2d x1(q[2 * idx], q[2 * idx + 1]);
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int2 neighborCellIdx = make_int2(cellIdx.x + i, cellIdx.y + j);
			if (neighborCellIdx.x < 0 || neighborCellIdx.x >= numCellsX || neighborCellIdx.y < 0 || neighborCellIdx.y >= numCellsY)
				continue;
			int binIdx = neighborCellIdx.y * numCellsX + neighborCellIdx.x;
			if (bin_start[binIdx] == -1)
				continue;
			//printf("droplet_%d binIdx: %d\n", idx, binIdx);
			for (int k = bin_start[binIdx]; k < bin_end[binIdx]; k++)
			{
				if (k > idx)
				{
					Vector2d x2(q[2 * k], q[2 * k + 1]);
					double r1 = r[idx];
					double r2 = r[k];
					double d = (x2 - x1).norm();
#if 1
					if (d < params.toltalSeperationLengthRatio * (r1 + r2))
					{
						// Spring force
						double reslength = sqrt(r1 * r1 + r2 * r2 + 2 * r1 * r2 * cos(2 * params.theta_eq * M_PI / 180.0));
						double deformation = d - reslength;
						Vector2d localF = params.springStiffness * deformation / d * (x2 - x1);
						atomicAdd(&F[2 * idx], localF.x());
						atomicAdd(&F[2 * idx + 1], localF.y());
						atomicAdd(&F[2 * k], -localF.x());
						atomicAdd(&F[2 * k + 1], -localF.y());
						if(params.springDampingEnabled)
						{
							// Damping force
							Vector2d x1prev(qprev[2 * idx], qprev[2 * idx + 1]);
							Vector2d x2prev(qprev[2 * k], qprev[2 * k + 1]);
							Vector2d relvel = (x2 - x2prev) / params.timeStep - (x1 - x1prev) / params.timeStep;
							double mass1 = 4. / 3. * M_PI * r1 * r1 * r1 * params.dropletDensity;
							double mass2 = 4. / 3. * M_PI * r2 * r2 * r2 * params.dropletDensity;
							double reducedMass = mass1 * mass2 / (mass1 + mass2);
							double dampingStiffness = 2 * sqrt(params.springStiffness * reducedMass) * params.springDamplingRatio;
							localF = dampingStiffness * relvel;
							atomicAdd(&F[2 * idx], localF.x());
							atomicAdd(&F[2 * idx + 1], localF.y());
							atomicAdd(&F[2 * k], -localF.x());
							atomicAdd(&F[2 * k + 1], -localF.y());
						}
						// Flow
						if (d < (r1 + r2) && params.flowEnabled)
						{
							double C1 = C[idx];
							double C2 = C[k];
							double A = M_PI * (r1 * r1 - (d * d + r1 * r1 - r2 * r2) * (d * d + r1 * r1 - r2 * r2) / (4 * d * d));
							double D = params.permeability;
							double flow = D * A * (C1 - C2);
							atomicAdd(&J[idx], flow);
							atomicAdd(&J[k], -flow);
						}
						atomicAdd(&numConnection[0], 1);
					}
#else
					double sepRatio = params.toltalSeperationLengthRatio;
					double maxAhRatio = params.maxAddhesionDistRatio;
					if (d < sepRatio * (r1 + r2))
					{
						double resRatio = params.SpringRestLengthRatio;
						if (d > maxAhRatio * (r1 + r2))
						{
							// Spring force
							Vector2d localF = params.springStiffness * (maxAhRatio - resRatio) / (sepRatio - maxAhRatio) * (sepRatio * (r1 + r2) - d) / d * (x2 - x1);
							atomicAdd(&F[2 * idx], localF.x());
							atomicAdd(&F[2 * idx + 1], localF.y());
							atomicAdd(&F[2 * k], -localF.x());
							atomicAdd(&F[2 * k + 1], -localF.y());
							// Damping force
							Vector2d x1prev(qprev[2 * idx], qprev[2 * idx + 1]);
							Vector2d x2prev(qprev[2 * k], qprev[2 * k + 1]);
							Vector2d relvel = (x2 - x2prev) / params.timeStep - (x1 - x1prev) / params.timeStep;
							localF = params.dampingStiffness * relvel;
							atomicAdd(&F[2 * idx], localF.x());
							atomicAdd(&F[2 * idx + 1], localF.y());
							atomicAdd(&F[2 * k], -localF.x());
							atomicAdd(&F[2 * k + 1], -localF.y());
						}
						else
						{
							// Spring force
							Vector2d localF = params.springStiffness * (d - resRatio * (r1 + r2)) / d * (x2 - x1);
							atomicAdd(&F[2 * idx], localF.x());
							atomicAdd(&F[2 * idx + 1], localF.y());
							atomicAdd(&F[2 * k], -localF.x());
							atomicAdd(&F[2 * k + 1], -localF.y());
							// Damping force
							Vector2d x1prev(qprev[2 * idx], qprev[2 * idx + 1]);
							Vector2d x2prev(qprev[2 * k], qprev[2 * k + 1]);
							Vector2d relvel = (x2 - x2prev) / params.timeStep - (x1 - x1prev) / params.timeStep;
							localF = params.dampingStiffness * relvel;
							atomicAdd(&F[2 * idx], localF.x());
							atomicAdd(&F[2 * idx + 1], localF.y());
							atomicAdd(&F[2 * k], -localF.x());
							atomicAdd(&F[2 * k + 1], -localF.y());
						}
						// Flow
						if (d < (r1 + r2) && params.flowEnabled)
						{
							double C1 = C[idx];
							double C2 = C[k];
							double A = M_PI * (r1 * r1 - (d * d + r1 * r1 - r2 * r2) * (d * d + r1 * r1 - r2 * r2) / (4 * d * d));
							double D = params.permeability;
							double flow = D * A * (C1 - C2);
							atomicAdd(&J[idx], flow);
							atomicAdd(&J[k], -flow);
						}
					}

#endif
				}
			}
		}
	}
}

__global__ void processSpringForceAndFlow_fineModel(const double* q,
	                                                const double* qprev,
	                                                const double* r,
	                                                const double* C,
	                                                   const int* droplets_bin_indices,
	                                                   const int* bin_start,
	                                                   const int* bin_end,
	                                                    const int num_droplets,
	                                                 const double cellSize,
	                                                    const int numCellsX,
	                                                    const int numCellsY,
	                                      const SimParameters_GPU params,
	                                                      double* F,
	                                                      double* J,
	                                                         int* numConnection)
{
	//printf("num_droplets: %d\n", num_droplets);
	using namespace Eigen;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	int hashIdx = droplets_bin_indices[idx];
	//printf("droplet_%d hashIdx: %d\n", idx, hashIdx);
	int2 cellIdx = make_int2(hashIdx % numCellsX, (hashIdx / numCellsX) % numCellsY);
	Vector2d x1(q[2 * idx], q[2 * idx + 1]);
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int2 neighborCellIdx = make_int2(cellIdx.x + i, cellIdx.y + j);
			if (neighborCellIdx.x < 0 || neighborCellIdx.x >= numCellsX || neighborCellIdx.y < 0 || neighborCellIdx.y >= numCellsY)
				continue;
			int binIdx = neighborCellIdx.y * numCellsX + neighborCellIdx.x;
			if (bin_start[binIdx] == -1)
				continue;
			//printf("droplet_%d binIdx: %d\n", idx, binIdx);
			for (int k = bin_start[binIdx]; k < bin_end[binIdx]; k++)
			{
				if (k > idx)
				{
					Vector2d x2(q[2 * k], q[2 * k + 1]);
					double r1 = r[idx];
					double r2 = r[k];
					double d = (x2 - x1).norm();
					if (d < params.toltalSeperationLengthRatio * (r1 + r2))
					{
						// Spring force
						//double reslength = r1 + r2;
						//double deformation = d - reslength;
						//bool inTension = deformation > 0;
						Vector2d localF = 48 * params.epsilon * (-pow(params.sigma / d, 12) / d + 0.5 * pow(params.sigma / d, 6) / d) * (x2 - x1) / d;
						atomicAdd(&F[2 * idx], localF.x());
						atomicAdd(&F[2 * idx + 1], localF.y());
						atomicAdd(&F[2 * k], -localF.x());
						atomicAdd(&F[2 * k + 1], -localF.y());
						if(params.springDampingEnabled)
						{
							// Damping force
							Vector2d x1prev(qprev[2 * idx], qprev[2 * idx + 1]);
							Vector2d x2prev(qprev[2 * k], qprev[2 * k + 1]);
							Vector2d relvel = (x2 - x2prev) / params.timeStep - (x1 - x1prev) / params.timeStep;
							//double mass1 = 4. / 3. * M_PI * r1 * r1 * r1 * params.dropletDensity;
							//double mass2 = 4. / 3. * M_PI * r2 * r2 * r2 * params.dropletDensity;
							//double reducedMass = mass1 * mass2 / (mass1 + mass2);
							//double dampingStiffness = 2 * sqrt(params.springStiffness * reducedMass) * params.springDamplingRatio;
							localF = params.dampingStiffness * relvel;
							atomicAdd(&F[2 * idx], localF.x());
							atomicAdd(&F[2 * idx + 1], localF.y());
							atomicAdd(&F[2 * k], -localF.x());
							atomicAdd(&F[2 * k + 1], -localF.y());
						}
						// Flow
						/*
						* if (d < (r1 + r2) && params.flowEnabled)
						{
							double C1 = C[idx];
							double C2 = C[k];
							double A = M_PI * (r1 * r1 - (d * d + r1 * r1 - r2 * r2) * (d * d + r1 * r1 - r2 * r2) / (4 * d * d));
							double D = params.permeability;
							double flow = D * A * (C1 - C2);
							atomicAdd(&J[idx], flow);
							atomicAdd(&J[k], -flow);
						}
						*/
						atomicAdd(&numConnection[0], 1);
					}
				}
			}
		}
	}
}

__global__ void updateRadius(const double* J, 
	                             const int num_droplets, 
	               const SimParameters_GPU params, 
	                               double* r)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	r[idx] = pow((4. / 3. * M_PI * r[idx] * r[idx] * r[idx] + params.timeStep * J[idx]) / (4. / 3. * M_PI), 1. / 3.);
}

__global__ void updateConfiguration(Droplet_GPU* droplets, 
	                                    double2* force_on_droplets,
			             const SimParameters_GPU params,
	                                   const int num_droplets, 
	                               const double* q, 
	                               const double* v, 
	                               const double* r,
	                               const double* F)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	droplets[idx].prevpos = droplets[idx].pos;
	droplets[idx].pos = make_double2(q[2 * idx], q[2 * idx + 1]);
	droplets[idx].vel = make_double2(v[2 * idx], v[2 * idx + 1]);
	force_on_droplets[idx] = make_double2(F[2 * idx], F[2 * idx + 1]);
	droplets[idx].prevradius = droplets[idx].radius;
	droplets[idx].radius = r[idx];
	if(!droplets[idx].fixed)
		droplets[idx].mass= 4. / 3. * M_PI * r[idx] * r[idx] * r[idx] * params.dropletDensity;
}

__global__ void convertToDouble3(const double* q, const int num_droplets, double3* q3)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	q3[idx] = make_double3(q[2 * idx], q[2 * idx + 1], 0);
}

__global__ void computeMassVector_kernel(const Droplet_GPU* droplets, const int num_droplets, const SimParameters_GPU params, double* mass)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_droplets)
		return;
	mass[idx * 2] = droplets[idx].mass;
	mass[idx * 2 + 1] = droplets[idx].mass;
}

__device__ void atomicDouble2Exch(double2* addr, double2 value)
{
	atomicExch(reinterpret_cast<unsigned long long*>(&addr->x), __double_as_longlong(value.x));
	atomicExch(reinterpret_cast<unsigned long long*>(&addr->y), __double_as_longlong(value.y));
}

__global__ void findePositionByUid_kernel(const Droplet_GPU* droplets, uint32_t numDroplets, uint32_t uid, double2* result)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numDroplets)
		return;
	if (droplets[idx].uid == uid)
	{
		atomicDouble2Exch(result, droplets[idx].pos);
	}
}

__global__ void applyDisplacement_kernel(const Droplet_GPU* droplets, double* q, const int* dropletUids, double displacement, int numDroplets, int numUids)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= numDroplets || idy >= numUids)
		return;
	if (droplets[idx].uid == dropletUids[idy])
	{
		q[2 * idx] += displacement;
		//printf("%d, %f\n", dropletUids[idy], q[2 * idx]);
	}
}

__global__ void applyDisplacement_y_kernel(const Droplet_GPU* droplets, double* q, const int* dropletUids, double displacement, int numDroplets, int numUids)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= numDroplets || idy >= numUids)
		return;
	if (droplets[idx].uid == dropletUids[idy])
	{
		q[2 * idx + 1] += displacement;
		//printf("%d, %f\n", dropletUids[idy], q[2 * idx]);
	}
}

__global__ void applyDisplacement_pureShear_kernel(const Droplet_GPU* droplets, double* q, const int* dropletUids, double displacement, int numDroplets, int numUids, double* q0)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= numDroplets || idy >= numUids)
		return;
	if (droplets[idx].uid == dropletUids[idy])
	{
		q[2 * idx] += q0[2 * dropletUids[idy] + 1] * displacement;
		q[2 * idx + 1] += q0[2 * dropletUids[idy]] * displacement;
		//printf("%d, %f\n", dropletUids[idy], q[2 * idx]);
	}
}

__global__ void applyDisplacement_bulk_kernel(const Droplet_GPU* droplets, double* q, const int* dropletUids, double displacement, int numDroplets, int numUids, double* q0)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= numDroplets || idy >= numUids)
		return;
	if (droplets[idx].uid == dropletUids[idy])
	{
		q[2 * idx] += q0[2 * dropletUids[idy]] * displacement;
		q[2 * idx + 1] += q0[2 * dropletUids[idy] + 1] * displacement;
		//printf("%d, %f\n", dropletUids[idy], q[2 * idx]);
	}
}

__global__ void fixDropletsWithinArea_kernel(Droplet_GPU* droplets, const int numDroplets, const double2 xRange, const double2 yRange)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numDroplets)
		return;
	if (droplets[idx].pos.x >= xRange.x && droplets[idx].pos.x <= xRange.y && droplets[idx].pos.y >= yRange.x && droplets[idx].pos.y <= yRange.y)
	{
		droplets[idx].fixed = true;
		droplets[idx].mass = INFINITY;
		droplets[idx].vel = make_double2(0.0, 0.0);
	}
}

__global__ void getNumberOfDisplacementDropletsWithinArea_kernel(const Droplet_GPU* droplets, const int numDroplets, const double2 xRange, const double2 yRange, int* numDropletsWithinArea)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numDroplets)
		return;
	//printf("droplet_%d, pos: (%f, %f)\n", idx, droplets[idx].pos.x, droplets[idx].pos.y);
	if (droplets[idx].pos.x >= xRange.x && droplets[idx].pos.x <= xRange.y && droplets[idx].pos.y >= yRange.x && droplets[idx].pos.y <= yRange.y)
	{
		//printf("is inside\n");
		atomicAdd(numDropletsWithinArea, 1);
	}
}

__global__ void getNumberOfDisplacementDropletsOutsideArea_kernel(const Droplet_GPU* droplets, const int numDroplets, const double2 xRange, const double2 yRange, int* numDropletsWithinArea)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numDroplets)
		return;
	//printf("droplet_%d, pos: (%f, %f)\n", idx, droplets[idx].pos.x, droplets[idx].pos.y);
	if (droplets[idx].pos.x <= xRange.x || droplets[idx].pos.x >= xRange.y || droplets[idx].pos.y <= yRange.x || droplets[idx].pos.y >= yRange.y)
	{
		//printf("is inside\n");
		atomicAdd(numDropletsWithinArea, 1);
	}
}

__global__ void assignDisplacementDropletsWithinArea_kernel(Droplet_GPU* droplets, const int numDroplets, const double2 xRange, const double2 yRange, int* dropletUids, int* numDropletsWithinArea)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numDroplets)
		return;
	if (droplets[idx].pos.x >= xRange.x && droplets[idx].pos.x <= xRange.y && droplets[idx].pos.y >= yRange.x && droplets[idx].pos.y <= yRange.y)
	{
		printf("droplets_%d, uid: %d, is disp\n", idx, droplets[idx].uid);
		int index = atomicAdd(numDropletsWithinArea, 1);
		dropletUids[index] = droplets[idx].uid;
		droplets[idx].fixed = true;
		droplets[idx].mass = INFINITY;
		droplets[idx].vel = make_double2(0.0, 0.0);
		
	}
}

__global__ void assignOutputDropletsWithinArea_kernel(Droplet_GPU* droplets, const int numDroplets, const double2 xRange, const double2 yRange, int* dropletUids, int* numDropletsWithinArea)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numDroplets)
		return;
	if (droplets[idx].pos.x >= xRange.x && droplets[idx].pos.x <= xRange.y && droplets[idx].pos.y >= yRange.x && droplets[idx].pos.y <= yRange.y)
	{
		printf("droplets_%d, uid: %d, is output\n", idx, droplets[idx].uid);
		int index = atomicAdd(numDropletsWithinArea, 1);
		dropletUids[index] = droplets[idx].uid;
		//droplets[idx].fixed = true;
		//droplets[idx].mass = INFINITY;
		//droplets[idx].vel = make_double2(0.0, 0.0);

	}
}

__global__ void assignDisplacementDropletsOutsideArea_kernel(Droplet_GPU* droplets, const int numDroplets, const double2 xRange, const double2 yRange, int* dropletUids, int* numDropletsWithinArea)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numDroplets)
		return;
	if (droplets[idx].pos.x <= xRange.x || droplets[idx].pos.x >= xRange.y || droplets[idx].pos.y <= yRange.x || droplets[idx].pos.y >= yRange.y)
	{
		printf("droplets_%d, uid: %d, is disp\n", idx, droplets[idx].uid);
		int index = atomicAdd(numDropletsWithinArea, 1);
		dropletUids[index] = droplets[idx].uid;
		droplets[idx].fixed = true;
		droplets[idx].mass = INFINITY;
		droplets[idx].vel = make_double2(0.0, 0.0);

	}
}

DropletsSimCore_GPU::DropletsSimCore_GPU()
{
}

DropletsSimCore_GPU::DropletsSimCore_GPU(int numDropx)
{
	params_ = SimParameters_GPU(numDropx);
}

DropletsSimCore_GPU::~DropletsSimCore_GPU()
{
}

void DropletsSimCore_GPU::initSimulation()
{
	num_droplets_ = params_.numDroplets;
	droplets_.resize(num_droplets_);
	force_on_droplets_=thrust::device_vector<double2>(num_droplets_,make_double2(0.0,0.0));
	number_of_connections_on_droplets_=thrust::device_vector<int>(num_droplets_,0);
	if (params_.distributionPattern == 4 || params_.distributionPattern == 5)
	{
		dim3 blockSize(1024);
		dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
		initDroplets_1D << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()), params_);
	}
	else
	{
		dim3 blockSize(32, 32);
		dim3 gridSize((params_.numDroplets_x + blockSize.x - 1) / blockSize.x, (params_.numDroplets_y + blockSize.y - 1) / blockSize.y);
		initDroplets << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()), params_);
	}
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization (initDroplets): " << cudaGetErrorString(err) << std::endl;
	}
	/*
	thrust::device_vector<int> droplet_uids(num_droplets_);
	thrust::transform(droplets_.begin(), droplets_.end(), droplet_uids.begin(), get_uid_functor());
	for(int i = 0; i < num_droplets_; i++)
	{
		std::cout << droplet_uids[i] << " ";
	}
	*/

	thrust::pair<double, double> init(DBL_MAX, DBL_MAX);
	thrust::pair<double, double> min_pos = thrust::transform_reduce(droplets_.begin(),
		droplets_.end(),
		get_position_functor(),
		init,
		min_pair_functor());

	double2 gridOrigin = make_double2(min_pos.first - params_.initialDropletRadius, min_pos.second - params_.initialDropletRadius);

	thrust::device_vector<int> droplets_bin_indices(num_droplets_);
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	assignToBins << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()),
		                                           num_droplets_,
		                                           thrust::raw_pointer_cast(droplets_bin_indices.data()),
		                                           params_.cellSize,
		                                           params_.numCells_x,
		                                           params_.numCells_y,
		                                           gridOrigin);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization (assignToBins in init): " << cudaGetErrorString(err) << std::endl;
	}
	thrust::sort_by_key(droplets_bin_indices.begin(), droplets_bin_indices.end(), droplets_.begin());

	thrust::device_vector<int> bin_start(params_.numCells_x * params_.numCells_y, -1);
	thrust::device_vector<int> bin_end(params_.numCells_x * params_.numCells_y, -1);
	identifyBinBoundaries << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_bin_indices.data()),
		                                                    num_droplets_,
		                                                    thrust::raw_pointer_cast(bin_start.data()),
		                                                    thrust::raw_pointer_cast(bin_end.data()));
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization (identifyBinBoundaries): " << cudaGetErrorString(err) << std::endl;
	}

	/*
	for(int i = 0; i < num_droplets_; i++)
	{
		std::cout << bin_start[i] << " ";
	}
	std::cout << std::endl;
	*/

	//blockSize = dim3(1024);
	//gridSize = dim3((num_droplets_ + blockSize.x - 1) / blockSize.x);
	//int numConnections = 0;
	thrust::device_vector<int> numConnectionsVec(1, 0);
	thrust::device_vector<int> numOutputConnections(1, 0);
	getNumberOfConnections << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()),
		                                                     thrust::raw_pointer_cast(droplets_bin_indices.data()),
		                                                     thrust::raw_pointer_cast(bin_start.data()),
		                                                     thrust::raw_pointer_cast(bin_end.data()),
		                                                     num_droplets_,
		                                                     params_.cellSize,
		                                                     params_.numCells_x,
		                                                     params_.numCells_y,
		                                                     params_,
		                                                     thrust::raw_pointer_cast(numConnectionsVec.data()),
		                                                     thrust::raw_pointer_cast(numOutputConnections.data()));
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization (getNumberOfConnections in init): " << cudaGetErrorString(err) << std::endl;
	}
	connectors_.resize(numConnectionsVec[0]);
	output_spring_index_.resize(numOutputConnections[0]);
	std::cout<< "Connection number: " << connectors_.size() << std::endl;
	//std::cout << output_spring_index_.size() << std::endl;
	numConnectionsVec[0] = 0;
	numOutputConnections[0] = 0;
	dim3 blockSize2(512);
	dim3 gridSize2((num_droplets_ + blockSize2.x - 1) / blockSize2.x);
	setupConnections << <gridSize2, blockSize2 >> > (thrust::raw_pointer_cast(droplets_.data()),
		                                               thrust::raw_pointer_cast(droplets_bin_indices.data()),
		                                               thrust::raw_pointer_cast(bin_start.data()),
		                                               thrust::raw_pointer_cast(bin_end.data()),
		                                               num_droplets_,
		                                               params_.cellSize,
		                                               params_.numCells_x,
		                                               params_.numCells_y,
		                                               params_,
		                                               thrust::raw_pointer_cast(numConnectionsVec.data()),
		                                               thrust::raw_pointer_cast(numOutputConnections.data()),
		                                               thrust::raw_pointer_cast(output_spring_index_.data()),
			                                           thrust::raw_pointer_cast(number_of_connections_on_droplets_.data()),
		                                               thrust::raw_pointer_cast(connectors_.data()));
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization (setupConnections in init): " << cudaGetErrorString(err) << std::endl;
	}

	if (params_.distributionPattern != 6)
		setupTimeStep();

	double4 boundingBox = getCurrentBoundingBox();
	params_.boundingBoxWidth = boundingBox.y - boundingBox.x;
	params_.boundingBoxHeight = boundingBox.w - boundingBox.z;
	params_.boundingBoxCenter = make_double2((boundingBox.x + boundingBox.y) / 2, (boundingBox.z + boundingBox.w) / 2);

	/*
	for (int i = 0; i < output_spring_index_.size(); i++)
	{
		std::cout << output_spring_index_[i] << std::endl;
	}
	*/
	//thrust::host_vector<Droplet_GPU> droplets_host = droplets_;

	/*
	for (int i = 0; i < connectors_host.size(); i++)
	{
		std::cout << "spring_" << i << " p1: " << connectors_host[i].p1 << " p2: " << connectors_host[i].p2 << std::endl;
	}
	std::cout << std::endl;
	
	for (int i = 0; i < num_droplets_; i++)
	{
		std::cout << "droplet_" << i << " pos: " << droplets_host[i].pos.x << ", " << droplets_host[i].pos.y << std::endl;
	}
	std::cout << "numConnections: " << numConnectionsVec[0] << std::endl;
	*/
}

/*
void DropletsSimCore_GPU::initSimulationWithSpring()
{
	num_droplets_ = params_.numDroplets_x * params_.numDroplets_y;
	droplets_.resize(num_droplets_);
	dim3 blockSize(32, 32);
	dim3 gridSize((params_.numDroplets_x + blockSize.x - 1) / blockSize.x, (params_.numDroplets_y + blockSize.y - 1) / blockSize.y);
	initDroplets << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()), params_);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
}
*/


void DropletsSimCore_GPU::simulateOneStep(std::mutex& data_mutex)
{
	//std::cout << "step: " << time_ << std::endl;
	thrust::pair<double, double> init(DBL_MAX, DBL_MAX);
	/*
	thrust::pair<double, double> min_pos = thrust::transform_reduce(droplets_.begin(), 
		                                                            droplets_.end(), 
		                                                            get_position_functor(), 
		                                                            init, 
		                                                            min_pair_functor());
	*/
	
	//thrust::pair<double, double> min_pos = thrust::make_pair(params_.boundingBoxLowerLeft.x, params_.boundingBoxLowerLeft.y);

	//double2 gridOrigin = make_double2(min_pos.first - params_.initialDropletRadius, min_pos.second - params_.initialDropletRadius);
	double2 gridOrigin = params_.boundingBoxLowerLeft;

	thrust::device_vector<int> droplets_bin_indices(num_droplets_);
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	assignToBins<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(droplets_.data()),
		                                  num_droplets_, 
		                                  thrust::raw_pointer_cast(droplets_bin_indices.data()), 
		                                  params_.cellSize, 
		                                  params_.numCells_x,
		                                  params_.numCells_y,
		                                  gridOrigin);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization (assignToBins): " << cudaGetErrorString(err) << std::endl;
	}
	//std::cout << "assignToBins done" << std::endl;
	thrust::sort_by_key(droplets_bin_indices.begin(), droplets_bin_indices.end(), droplets_.begin());
	
	thrust::device_vector<int> bin_start(params_.numCells_x * params_.numCells_y, -1);
	thrust::device_vector<int> bin_end(params_.numCells_x * params_.numCells_y, -1);
	identifyBinBoundaries<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(droplets_bin_indices.data()), 
		                                           num_droplets_, 
		                                           thrust::raw_pointer_cast(bin_start.data()), 
		                                           thrust::raw_pointer_cast(bin_end.data()));
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization (identifyBinBoundaries): " << cudaGetErrorString(err) << std::endl;
	}
	//std::cout << "identifyBinBoundaries done" << std::endl;

	if (!params_.fixedPermeability)
		adjustLoad();

	thrust::device_vector<double> q, qprev, v, r, C;
	{
		//std::lock_guard<std::mutex> lock(data_mutex);
		std::tie(q, qprev, v, r, C) = buildConfiguration();
	}
	//std::cout << "buildConfiguration done" << std::endl;

	//for (int i = 0; i < q.size(); i++)
		//std::cout << q[i] << std::endl;
	//std::cout << std::endl;
	//for (int i = 0; i < v.size(); i++)
		//std::cout << v[i] << std::endl;
	//std::cout << std::endl;

	//for (int i = 0; i < C.size(); i++)
		//std::cout << C[i] << std::endl;
	//std::cout << std::endl;

	//for (int i = 0; i < r.size(); i++)
		//std::cout << r[i] << std::endl;
	//std::cout << std::endl;

	if (params_.mechanicalLoadingEnabled)
	{
		switch (params_.loadType)
		{
		case LOAD_TYPE::displacement:
			if (params_.startApplyDisplacement)
			{
				applyDisplacement(q);
			}
			break;
		case LOAD_TYPE::velocity:
			if (connectors_.size() == 0)
				v = params_.initialDropletsVelocity;
			break;
		}
	}
	
	//for (int i = 0; i < q.size(); i++)
		//std::cout << q[i] << std::endl;
	//std::cout << std::endl;
	thrust::device_vector<double> F(num_droplets_ * 2, 0);
	numericalIntegration(droplets_bin_indices, bin_start, bin_end, q, qprev, v, F, r, C, data_mutex);
	//std::cout << "numericalIntegration done" << std::endl;
	//for(int i = 0; i< num_droplets_;i++)
	//{
	//	std::cout << "F[" << i << "]: " << F[2 * i] << ", " << F[2 * i + 1] << std::endl;
	//}
	{
		std::lock_guard<std::mutex> lock(data_mutex);
		force_on_droplets_.resize(num_droplets_);
		unbuildConfiguration(q, v, r, F);
		if (params_.distributionPattern != 6)
			removeDropletsOutsideDomain();
	}

	//for (int i = 0; i < q.size(); i++)
		//std::cout << q[i] << std::endl;
	//std::cout << std::endl;
	//for (int i = 0; i < v.size(); i++)
		//std::cout << v[i] << std::endl;
	//std::cout << std::endl;


	double4 boundingBox = getCurrentBoundingBox();
	params_.boundingBoxWidth = boundingBox.y - boundingBox.x;
	params_.boundingBoxHeight = boundingBox.w - boundingBox.z;
	params_.boundingBoxCenter = make_double2((boundingBox.x + boundingBox.y) / 2, (boundingBox.z + boundingBox.w) / 2);
	params_.boundingBoxLowerLeft = make_double2(boundingBox.x, boundingBox.z);
	params_.numCells_x = (int)ceil(params_.boundingBoxWidth * 1.5 / params_.cellSize);
	params_.numCells_y = params_.boundingBoxHeight == 0 ? 1 : (int)ceil(params_.boundingBoxHeight * 1.5 / params_.cellSize);

	if (params_.mechanicalLoadingEnabled)
	{
		if (params_.applyDisplacementAfterSettling)
		{
			//double maxDropletTotalForce = getMaxDropletTotalForce();
			//std::cout << "maxDropletTotalForce: " << maxDropletTotalForce << std::endl;
			//if (maxDropletTotalForce < params_.settlingThreshold && !params_.fixDroplets)
			if (!params_.compressionBeforeStretching)
			{
				if (time_ > params_.settlingTime && !params_.fixDroplets)
				{
					std::lock_guard<std::mutex> lock(data_mutex);
					std::cout << "boundingBox: " << boundingBox.x << ", " << boundingBox.y << ", " << boundingBox.z << ", " << boundingBox.w << std::endl;
					std::cout << "area width: " << boundingBox.y - boundingBox.x << std::endl;
					fixDropletsWithinArea(make_double2(std::numeric_limits<double>::lowest(),
						(boundingBox.y - boundingBox.x) * 0.2),
						make_double2(std::numeric_limits<double>::lowest(),
							std::numeric_limits<double>::max()));
					assignDisplacementDropletsWithinArea(make_double2(boundingBox.y - (boundingBox.y - boundingBox.x) * 0.2,
						std::numeric_limits<double>::max()),
						make_double2(std::numeric_limits<double>::lowest(),
							std::numeric_limits<double>::max()));


					//params_.delta_displacement = params_.maxDisplacement * params_.timeStep / params_.increaseTime;

					params_.fixDroplets = true;
					params_.wallEnabled = false;
					params_.startApplyDisplacement = true;
					params_.maxDisplacement = (boundingBox.y - boundingBox.x) * 0.8 * params_.maxTensileStrain;
					params_.stretchingSpeed = params_.maxDisplacement / params_.increaseTime;
					params_.loadingSpeed = params_.stretchingSpeed;
				}
			}
			else
			{
				if (time_ > params_.settlingTime && !params_.fixDroplets)
				{
					std::lock_guard<std::mutex> lock(data_mutex);
					double4 boundingBox = getCurrentBoundingBox();
					fixDropletsWithinArea(make_double2(std::numeric_limits<double>::lowest(),
						(boundingBox.y - boundingBox.x) * 0.1),
						make_double2(std::numeric_limits<double>::lowest(),
							std::numeric_limits<double>::max()));
					assignDisplacementDropletsWithinArea(make_double2(boundingBox.y - (boundingBox.y - boundingBox.x) * 0.1,
						std::numeric_limits<double>::max()),
						make_double2(std::numeric_limits<double>::lowest(),
							std::numeric_limits<double>::max()));
					params_.fixDroplets = true;
					params_.wallEnabled = false;
					params_.startApplyDisplacement = true;
					params_.maxDisplacement = (boundingBox.y - boundingBox.x) * 0.8 * params_.maxTensileStrain;
					params_.stretchingSpeed = params_.maxDisplacement / params_.increaseTime;
					params_.maxCompressionDisplacement = (boundingBox.y - boundingBox.x) * 0.8 * params_.maxCompressionStrain;
					params_.compressionSpeed = params_.maxCompressionDisplacement / params_.compressionTime;
					params_.loadingSpeed = -params_.compressionSpeed;
				}
				if (time_ > params_.settlingTime + params_.compressionTime && !params_.compressed)
				{
					params_.loadingSpeed = params_.stretchingSpeed;
					params_.compressed = true;
				}
			}
		}
	}
	//std::cout << "unbuildConfiguration done" << std::endl;
	//for (int i =0;i< num_droplets_;i++)
	time_ += params_.timeStep;
}


void DropletsSimCore_GPU::simulateOneStepWithSpring(std::mutex& data_mutex)
{
	thrust::device_vector<double> q, qprev, v, r, C;
	std::tie(q, qprev, v, r, C) = buildConfiguration();

	if (!params_.fixedPermeability)
		adjustLoad();
	if (params_.mechanicalLoadingEnabled)
	{
		if (params_.startApplyDisplacement)
		{
			if (params_.isShear)
				applyDisplacement_pureShear(q);
			else if (params_.isBulk)
				applyDisplacement_bulk(q);
			else
				applyDisplacement(q);
		}
	}
	thrust::device_vector<double> F(num_droplets_ * 2, 0);
	numericalIntegrationWithSpring(q, qprev, v, F, r, C, data_mutex);

	/*
	for (int i = 0; i < num_droplets_; i++)
	{
		std::cout << "F[" << i << "]: " << F[2 * i] << ", " << F[2 * i + 1] << std::endl;
	}
	*/
	
	{
		std::lock_guard<std::mutex> lock(data_mutex);
		force_on_droplets_.resize(num_droplets_);
		unbuildConfiguration(q, v, r, F);
		if (params_.canSnap)
			removeSnappedSprings();
	}

	if (params_.mechanicalLoadingEnabled)
	{
		double4 boundingBox = getCurrentBoundingBox();
		params_.boundingBoxWidth = boundingBox.y - boundingBox.x;
		params_.boundingBoxHeight = boundingBox.w - boundingBox.z;
		params_.boundingBoxCenter = make_double2((boundingBox.x + boundingBox.y) / 2, (boundingBox.z + boundingBox.w) / 2);

		if (params_.applyDisplacementAfterSettling)
		{
			if (!params_.compressionBeforeStretching)
			{
				if (time_ > params_.settlingTime && !params_.fixDroplets)
				{
					std::lock_guard<std::mutex> lock(data_mutex);
					std::cout << "boundingBox: " << boundingBox.x << ", " << boundingBox.y << ", " << boundingBox.z << ", " << boundingBox.w << std::endl;
					std::cout << "area width: " << boundingBox.y - boundingBox.x << std::endl;
					if (!params_.isShear && !params_.isBulk)
					{
						fixDropletsWithinArea(make_double2(std::numeric_limits<double>::lowest(),
							                              (boundingBox.y - boundingBox.x) * 0.2),
							                  make_double2(std::numeric_limits<double>::lowest(),
								                           std::numeric_limits<double>::max()));
						assignDisplacementDropletsWithinArea(make_double2(boundingBox.y - (boundingBox.y - boundingBox.x) * 0.2,
							                                              std::numeric_limits<double>::max()),
							                                 make_double2(std::numeric_limits<double>::lowest(),
								                                          std::numeric_limits<double>::max()));
						assignOutputDropletsWithinArea(make_double2(boundingBox.y - (boundingBox.y - boundingBox.x) * 0.2,
							                                        std::numeric_limits<double>::max()),
							                           make_double2(std::numeric_limits<double>::lowest(),
								                                    std::numeric_limits<double>::max()));
						params_.fixDroplets = true;
						params_.wallEnabled = false;
						params_.startApplyDisplacement = true;
						params_.maxDisplacement = (boundingBox.y - boundingBox.x) * 0.6 * params_.maxTensileStrain;
						params_.stretchingSpeed = params_.maxDisplacement / params_.increaseTime;
						params_.loadingSpeed = params_.stretchingSpeed;
						params_.delta_displacement = params_.loadingSpeed * params_.timeStep;
					}
					else
					{
						assignDisplacementDropletsOutsideArea(make_double2(boundingBox.x + (boundingBox.y - boundingBox.x) * 0.1,
							                                               boundingBox.y - (boundingBox.y - boundingBox.x) * 0.1),
							                                  make_double2(boundingBox.z + (boundingBox.w - boundingBox.z) * 0.1,
							                                  	           boundingBox.w - (boundingBox.w - boundingBox.z) * 0.1));
						assignOutputDropletsWithinArea(make_double2(boundingBox.y - (boundingBox.y - boundingBox.x) * 0.1,
							                                        std::numeric_limits<double>::max()),
							                           make_double2(std::numeric_limits<double>::lowest(),
							                           	            std::numeric_limits<double>::max()));
						params_.fixDroplets = true;
						params_.wallEnabled = false;
						params_.startApplyDisplacement = true;
						params_.maxDisplacement = params_.maxTensileStrain;
						params_.stretchingSpeed = params_.maxDisplacement / params_.increaseTime;
						params_.loadingSpeed = params_.stretchingSpeed;
						params_.delta_displacement = params_.loadingSpeed * params_.timeStep;
					}
				}
			}
			else
			{
				if (time_ > params_.settlingTime && !params_.fixDroplets)
				{
					std::lock_guard<std::mutex> lock(data_mutex);
					double4 boundingBox = getCurrentBoundingBox();
					fixDropletsWithinArea(make_double2(std::numeric_limits<double>::lowest(),
						(boundingBox.y - boundingBox.x) * 0.1),
						make_double2(std::numeric_limits<double>::lowest(),
							std::numeric_limits<double>::max()));
					assignDisplacementDropletsWithinArea(make_double2(boundingBox.y - (boundingBox.y - boundingBox.x) * 0.1,
						std::numeric_limits<double>::max()),
						make_double2(std::numeric_limits<double>::lowest(),
							std::numeric_limits<double>::max()));
					params_.fixDroplets = true;
					params_.wallEnabled = false;
					params_.startApplyDisplacement = true;
					params_.maxDisplacement = (boundingBox.y - boundingBox.x) * 0.8 * params_.maxTensileStrain;
					params_.stretchingSpeed = params_.maxDisplacement / params_.increaseTime;
					params_.maxCompressionDisplacement = (boundingBox.y - boundingBox.x) * 0.8 * params_.maxCompressionStrain;
					params_.compressionSpeed = params_.maxCompressionDisplacement / params_.compressionTime;
					params_.loadingSpeed = -params_.compressionSpeed;
					params_.delta_displacement = params_.loadingSpeed * params_.timeStep;
				}
				if (time_ > params_.settlingTime + params_.compressionTime && !params_.compressed)
				{
					params_.loadingSpeed = params_.stretchingSpeed;
					params_.compressed = true;
					params_.delta_displacement = params_.loadingSpeed * params_.timeStep;
				}
			}
		}
	}
	

	time_ += params_.timeStep;
}



void DropletsSimCore_GPU::physicsLoop(bool& startFlag,
					     std::atomic<bool>& stopFlag, 
	                     std::mutex& data_mutex)
{
	while (!stopFlag.load())
	{
		if (!startFlag)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}
		else
		{
			//std::lock_guard<std::mutex> lock(data_mutex);
			if(params_.physicsLoopDelay)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			if (params_.springsEnabled)
				simulateOneStepWithSpring(data_mutex);
			else
				simulateOneStep(data_mutex);
		}
	}
}



double2 DropletsSimCore_GPU::getGlobalCenter() const
{
	thrust::pair<double, double> init(0.0, 0.0);
	thrust::pair<double, double> sum_pos = thrust::transform_reduce(droplets_.begin(),
		                                                            droplets_.end(),
		                                                            get_position_functor(),
		                                                            init,
		                                                            sum_pair_functor());
	return make_double2(sum_pos.first / num_droplets_, sum_pos.second / num_droplets_);
}

double2 DropletsSimCore_GPU::getGlobalSize() const
{
	thrust::pair<double, double> init(DBL_MIN, DBL_MIN);
	thrust::pair<double, double> max_pos = thrust::transform_reduce(droplets_.begin(),
		                                                            droplets_.end(),
		                                                            get_position_functor(),
		                                                            init,
		                                                            max_pair_functor());
	thrust::pair<double, double> min_pos = thrust::transform_reduce(droplets_.begin(),
		                                                            droplets_.end(),
		                                                            get_position_functor(),
		                                                            init,
		                                                            min_pair_functor());
	return make_double2(max_pos.first - min_pos.first, max_pos.second - min_pos.second);
}

void DropletsSimCore_GPU::findMaxVelocityDroplet(double& max_velocity, int& max_velocity_droplet) const
{
	thrust::device_vector<double> velocity_magnitudes(num_droplets_);
	thrust::transform(droplets_.begin(), droplets_.end(), velocity_magnitudes.begin(), velocity_magnitude_functor());
	thrust::device_vector<double>::iterator result = thrust::max_element(velocity_magnitudes.begin(), velocity_magnitudes.end());
	max_velocity = *result;
	max_velocity_droplet = result - velocity_magnitudes.begin();
}

double DropletsSimCore_GPU::getMaxConcentration() const
{
	thrust::device_vector<double> concentrations(num_droplets_);
	thrust::transform(droplets_.begin(), droplets_.end(), concentrations.begin(), concentration_functor());
	thrust::device_vector<double>::iterator result = thrust::max_element(concentrations.begin(), concentrations.end());
	return *result;
}

double DropletsSimCore_GPU::getMinConcentration() const
{
	thrust::device_vector<double> concentrations(num_droplets_);
	thrust::transform(droplets_.begin(), droplets_.end(), concentrations.begin(), concentration_functor());
	thrust::device_vector<double>::iterator result = thrust::min_element(concentrations.begin(), concentrations.end());
	return *result;
}

thrust::pair<double, double> DropletsSimCore_GPU::getConcentration(thrust::device_vector<double>& C) const
{
	C.resize(num_droplets_);
	thrust::transform(droplets_.begin(), droplets_.end(), C.begin(), concentration_functor());
	thrust::pair<thrust::device_vector<double>::iterator, thrust::device_vector<double>::iterator> result = thrust::minmax_element(C.begin(), C.end());
	return thrust::make_pair(*result.first, *result.second);
}

double DropletsSimCore_GPU::getMaxRadiusIncreaseSpeed() const
{
	thrust::device_vector<double> radius_increments(num_droplets_);
	thrust::transform(droplets_.begin(), droplets_.end(), radius_increments.begin(), radius_increment_functor());
	thrust::device_vector<double>::iterator result = thrust::max_element(radius_increments.begin(), radius_increments.end());
	return (*result) / params_.timeStep;
}

double DropletsSimCore_GPU::getMaxRadiusDecreaseSpeed() const
{
	thrust::device_vector<double> radius_increments(num_droplets_);
	thrust::transform(droplets_.begin(), droplets_.end(), radius_increments.begin(), radius_increment_functor());
	thrust::device_vector<double>::iterator result = thrust::min_element(radius_increments.begin(), radius_increments.end());
	return (*result) / params_.timeStep;
}

thrust::pair<double, double> DropletsSimCore_GPU::getMaxRadiusChangeSpeed() const
{
	thrust::device_vector<double> radius_increments(num_droplets_);
	thrust::transform(droplets_.begin(), droplets_.end(), radius_increments.begin(), radius_increment_functor());
	thrust::pair<thrust::device_vector<double>::iterator, thrust::device_vector<double>::iterator> result = thrust::minmax_element(radius_increments.begin(), radius_increments.end());
	return thrust::make_pair((*result.first) / params_.timeStep, (*result.second) / params_.timeStep);
}

double DropletsSimCore_GPU::getTotalKineticEnergy() const
{
	return thrust::transform_reduce(droplets_.begin(), droplets_.end(), kinetic_energy_functor(), 0.0, thrust::plus<double>());
}

double2 DropletsSimCore_GPU::getDropletPosition(uint32_t uid) const
{
	double2 result;
	double2* result_ptr;
	cudaMallocManaged(&result_ptr, sizeof(double2));
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	findePositionByUid_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(droplets_.data()), num_droplets_, uid, result_ptr);
	cudaDeviceSynchronize();
	cudaMemcpy(&result, result_ptr, sizeof(double2), cudaMemcpyDeviceToHost);
	cudaFree(result_ptr);
	return result;
}

double DropletsSimCore_GPU::getEndToEndDistance() const
{
	uint32_t topLeftDropletIndex = num_droplets_ - params_.numDroplets_x;
	uint32_t topRightDropletIndex = num_droplets_ - 1;

	double2 topLeftPos = getDropletPosition(topLeftDropletIndex);
	double2 topRightPos = getDropletPosition(topRightDropletIndex);

	return sqrt((topRightPos.x - topLeftPos.x) * (topRightPos.x - topLeftPos.x) + (topRightPos.y - topLeftPos.y) * (topRightPos.y - topLeftPos.y));
}

int DropletsSimCore_GPU::getIndexofDroplet(uint32_t uid) const
{
	auto result = thrust::find_if(droplets_.begin(), droplets_.end(), DropletUIDMatcher(uid));
	return result - droplets_.begin();
}

double2 DropletsSimCore_GPU::getTotalForce() const
{
	double2 totalForce = make_double2(0.0, 0.0);
	for (int i = 0; i < params_.outputDropletUids.size(); i++)
	{
		int idx = getIndexofDroplet(params_.outputDropletUids[i]);
		double2 force = force_on_droplets_[idx];
		totalForce.x += force.x;
		totalForce.y += force.y;
	}
	return totalForce;
}

double DropletsSimCore_GPU::getMaxDropletTotalForce() const
{
	thrust::device_vector<double> force_magnitudes(num_droplets_);
	thrust::transform(force_on_droplets_.begin(), force_on_droplets_.end(), force_magnitudes.begin(), force_magnitude_functor());
	thrust::device_vector<double>::iterator result = thrust::max_element(force_magnitudes.begin(), force_magnitudes.end());
	return *result;
}

double4 DropletsSimCore_GPU::getCurrentBoundingBox() const
{
	thrust::device_vector<double> xs(num_droplets_);
	thrust::device_vector<double> ys(num_droplets_);
	thrust::transform(droplets_.begin(), droplets_.end(), xs.begin(), get_position_x_functor());
	thrust::transform(droplets_.begin(), droplets_.end(), ys.begin(), get_position_y_functor());
	thrust::pair<thrust::device_vector<double>::iterator, thrust::device_vector<double>::iterator> x_result = thrust::minmax_element(xs.begin(), xs.end());
	thrust::pair<thrust::device_vector<double>::iterator, thrust::device_vector<double>::iterator> y_result = thrust::minmax_element(ys.begin(), ys.end());
	//std::cout << "x: " << *x_result.first << ", " << *x_result.second << " y: " << *y_result.first << ", " << *y_result.second << std::endl;
	return make_double4(*x_result.first, *x_result.second, *y_result.first, *y_result.second);
}

void DropletsSimCore_GPU::setupTimeStep()
{
	thrust::device_vector<double> spring_periods(connectors_.size());
	thrust::transform(connectors_.begin(), connectors_.end(), spring_periods.begin(), spring_period_functor());
	thrust::device_vector<double>::iterator result = thrust::min_element(spring_periods.begin(), spring_periods.end());
	params_.timeStep = *result / 10.0;
	params_.delta_displacement = params_.loadingSpeed * params_.timeStep;
}

void DropletsSimCore_GPU::adjustLoad()
{
	if (params_.permeability < params_.maxPermeability)
		params_.permeability += params_.delta_permeability;
}

void DropletsSimCore_GPU::applyDisplacement(thrust::device_vector<double>& q)
{
	if (displacement_ < params_.maxDisplacement)
	{
		displacement_ += params_.delta_displacement;
		dim3 blockSize(32, 32);
		dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x, (params_.displacementDropletUid.size() + blockSize.y - 1) / blockSize.y);
		applyDisplacement_kernel << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()), 
			                                                   thrust::raw_pointer_cast(q.data()), 
			                                                   thrust::raw_pointer_cast(params_.displacementDropletUid.data()), 
			                                                   params_.delta_displacement, 
			                                                   num_droplets_, 
			                                                   params_.displacementDropletUid.size());
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
		}
	}
}

void DropletsSimCore_GPU::applyDisplacement_y(thrust::device_vector<double>& q)
{
	if (displacement_ < params_.maxDisplacement)
	{
		displacement_ += params_.delta_displacement;
		dim3 blockSize(32, 32);
		dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x, (params_.displacementDropletUid.size() + blockSize.y - 1) / blockSize.y);
		applyDisplacement_y_kernel << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()),
			                                                     thrust::raw_pointer_cast(q.data()),
			                                                     thrust::raw_pointer_cast(params_.displacementDropletUid.data()),
			                                                     params_.delta_displacement,
			                                                     num_droplets_,
			                                                     params_.displacementDropletUid.size());
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
		}
	}
}

void DropletsSimCore_GPU::applyDisplacement_pureShear(thrust::device_vector<double>& q)
{
	if (displacement_ < params_.maxDisplacement)
	{
		displacement_ += params_.delta_displacement;
		dim3 blockSize(32, 32);
		dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x, (params_.displacementDropletUid.size() + blockSize.y - 1) / blockSize.y);
		applyDisplacement_pureShear_kernel << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()),
			                                                             thrust::raw_pointer_cast(q.data()),
			                                                             thrust::raw_pointer_cast(params_.displacementDropletUid.data()),
			                                                             params_.delta_displacement,
			                                                             num_droplets_,
			                                                             params_.displacementDropletUid.size(), 
			                                                             thrust::raw_pointer_cast(params_.dropletPositions.data()));
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
		}
	}
}

void DropletsSimCore_GPU::applyDisplacement_bulk(thrust::device_vector<double>& q)
{
	if (displacement_ < params_.maxDisplacement)
	{
		displacement_ += params_.delta_displacement;
		dim3 blockSize(32, 32);
		dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x, (params_.displacementDropletUid.size() + blockSize.y - 1) / blockSize.y);
		applyDisplacement_bulk_kernel << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()),
			                                                        thrust::raw_pointer_cast(q.data()),
			                                                        thrust::raw_pointer_cast(params_.displacementDropletUid.data()),
			                                                        params_.delta_displacement,
			                                                        num_droplets_,
			                                                        params_.displacementDropletUid.size(), 
			                                                        thrust::raw_pointer_cast(params_.dropletPositions.data()));
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
		}
	}
}


int DropletsSimCore_GPU::getNumberOfDisplacementDropletsWithinArea(double2 xRange, double2 yRange)
{
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	int result = 0;
	int* numDropletsWithinArea;
	cudaMalloc(&numDropletsWithinArea, sizeof(int));
	cudaMemcpy(numDropletsWithinArea, &result, sizeof(int), cudaMemcpyHostToDevice);
	std::cout << "xRange: " << xRange.x << ", " << xRange.y <<  " yRange: " << yRange.x << ", " << yRange.y << std::endl;
	getNumberOfDisplacementDropletsWithinArea_kernel << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()), num_droplets_, xRange, yRange, numDropletsWithinArea);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
	cudaMemcpy(&result, numDropletsWithinArea, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(numDropletsWithinArea);
	return result;
}

int DropletsSimCore_GPU::getNumberOfOutputDropletsWithinArea(double2 xRange, double2 yRange)
{
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	int result = 0;
	int* numDropletsWithinArea;
	cudaMalloc(&numDropletsWithinArea, sizeof(int));
	cudaMemcpy(numDropletsWithinArea, &result, sizeof(int), cudaMemcpyHostToDevice);
	std::cout << "xRange: " << xRange.x << ", " << xRange.y << " yRange: " << yRange.x << ", " << yRange.y << std::endl;
	getNumberOfDisplacementDropletsWithinArea_kernel << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()), num_droplets_, xRange, yRange, numDropletsWithinArea);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
	cudaMemcpy(&result, numDropletsWithinArea, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(numDropletsWithinArea);
	return result;
}

int DropletsSimCore_GPU::getNumberOfDisplacementDropletsOutsideArea(double2 xRange, double2 yRange)
{
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	int result = 0;
	int* numDropletsOutsideArea;
	cudaMalloc(&numDropletsOutsideArea, sizeof(int));
	cudaMemcpy(numDropletsOutsideArea, &result, sizeof(int), cudaMemcpyHostToDevice);
	std::cout << "xRange: " << xRange.x << ", " << xRange.y << " yRange: " << yRange.x << ", " << yRange.y << std::endl;
	getNumberOfDisplacementDropletsOutsideArea_kernel << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()), num_droplets_, xRange, yRange, numDropletsOutsideArea);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
	cudaMemcpy(&result, numDropletsOutsideArea, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(numDropletsOutsideArea);
	return result;
}

void DropletsSimCore_GPU::assignDisplacementDropletsWithinArea(double2 xRange, double2 yRange)
{
	int numDropletsWithinArea = getNumberOfDisplacementDropletsWithinArea(xRange, yRange);
	std::cout << "numDropletsWithinArea: " << numDropletsWithinArea << std::endl;
	params_.displacementDropletUid.resize(numDropletsWithinArea);
	numDropletsWithinArea = 0;
	int *numDropletsWithinArea_d;
	cudaMalloc(&numDropletsWithinArea_d, sizeof(int));
	cudaMemcpy(numDropletsWithinArea_d, &numDropletsWithinArea, sizeof(int), cudaMemcpyHostToDevice);
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	assignDisplacementDropletsWithinArea_kernel << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()), 
		                                                                      num_droplets_, 
		                                                                      xRange,
		                                                                      yRange,
		                                                                      thrust::raw_pointer_cast(params_.displacementDropletUid.data()),
		                                                                      numDropletsWithinArea_d);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
	for (int i = 0; i < params_.displacementDropletUid.size(); i++)
		std::cout << params_.displacementDropletUid[i] << " ";
	std::cout << std::endl;
}

void DropletsSimCore_GPU::assignOutputDropletsWithinArea(double2 xRange, double2 yRange)
{
	int numDropletsWithinArea = getNumberOfOutputDropletsWithinArea(xRange, yRange);
	std::cout << "numDropletsWithinArea: " << numDropletsWithinArea << std::endl;
	params_.outputDropletUids.resize(numDropletsWithinArea);
	numDropletsWithinArea = 0;
	int* numDropletsWithinArea_d;
	cudaMalloc(&numDropletsWithinArea_d, sizeof(int));
	cudaMemcpy(numDropletsWithinArea_d, &numDropletsWithinArea, sizeof(int), cudaMemcpyHostToDevice);
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	assignOutputDropletsWithinArea_kernel << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()),
		num_droplets_,
		xRange,
		yRange,
		thrust::raw_pointer_cast(params_.outputDropletUids.data()),
		numDropletsWithinArea_d);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
	for (int i = 0; i < params_.displacementDropletUid.size(); i++)
		std::cout << params_.displacementDropletUid[i] << " ";
	std::cout << std::endl;
}

void DropletsSimCore_GPU::assignDisplacementDropletsOutsideArea(double2 xRange, double2 yRange)
{
	int numDropletsWithinArea = getNumberOfDisplacementDropletsOutsideArea(xRange, yRange);
	std::cout << "numDropletsWithinArea: " << numDropletsWithinArea << std::endl;
	params_.displacementDropletUid.resize(numDropletsWithinArea);
	numDropletsWithinArea = 0;
	int *numDropletsWithinArea_d;
	cudaMalloc(&numDropletsWithinArea_d, sizeof(int));
	cudaMemcpy(numDropletsWithinArea_d, &numDropletsWithinArea, sizeof(int), cudaMemcpyHostToDevice);
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	assignDisplacementDropletsOutsideArea_kernel << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()),
		                                                                      num_droplets_, 
		                                                                      xRange,
		                                                                      yRange,
		                                                                      thrust::raw_pointer_cast(params_.displacementDropletUid.data()),
		                                                                      numDropletsWithinArea_d);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
	for (int i = 0; i < params_.displacementDropletUid.size(); i++)
		std::cout << params_.displacementDropletUid[i] << " ";
	std::cout << std::endl;
}

void DropletsSimCore_GPU::applyVelocity(thrust::device_vector<double>& v)
{
}

void DropletsSimCore_GPU::fixDropletsWithinArea(double2 xRange, double2 yRange)
{
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	fixDropletsWithinArea_kernel << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()), num_droplets_, xRange, yRange);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
}



void DropletsSimCore_GPU::getCurrentDropletPositionRadiusConcentration(thrust::device_vector<double3>& q_d3,
	                                                                   thrust::device_vector<double>& r,
	                                                                   thrust::device_vector<double>& C) const
{
	q_d3.resize(num_droplets_);
	r.resize(num_droplets_);
	C.resize(num_droplets_);
	thrust::device_vector<double> q(num_droplets_ * 2);
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	getConfiguration<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(droplets_.data()), 
		                                      num_droplets_, 
		                                      thrust::raw_pointer_cast(q.data()), 
		                                      nullptr, 
		                                      nullptr, 
		                                      thrust::raw_pointer_cast(r.data()),
		                                      nullptr,
		                                      thrust::raw_pointer_cast(C.data()));
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
	
	convertToDouble3<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(q.data()), 
		                                      num_droplets_, 
		                                      thrust::raw_pointer_cast(q_d3.data()));
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
}

void DropletsSimCore_GPU::getAllDropletPositionAsD3(thrust::device_vector<double3>& q) const
{
	q.resize(num_droplets_);
	thrust::transform(droplets_.begin(), droplets_.end(), q.begin(), get_position_as_d3_functor());
}

void DropletsSimCore_GPU::getAllDropletRadius(thrust::device_vector<double>& r) const
{
	r.resize(num_droplets_);
	thrust::transform(droplets_.begin(), droplets_.end(), r.begin(), radius_functor());
}

int DropletsSimCore_GPU::getAverageNumberOfConnections() const
{
	return thrust::reduce(number_of_connections_on_droplets_.begin(), number_of_connections_on_droplets_.end(), 0, thrust::plus<int>()) / num_droplets_;
}

thrust::pair<double, double> DropletsSimCore_GPU::getAllSpringConfiguration(thrust::device_vector<double3>& q,
	                                                                         thrust::device_vector<double>& r,
		                                                                     thrust::device_vector<double>& lam, 
	                                                                         thrust::device_vector<double>& f) const
{
	//std::cout << connectors_.size() << " ";
	q.resize(connectors_.size());
	r.resize(connectors_.size());
	lam.resize(connectors_.size());
	f.resize(connectors_.size());
	//std::cout << connectors_.size() << std::endl;
	auto zip_output = thrust::make_zip_iterator(thrust::make_tuple(q.begin(), r.begin(), lam.begin(), f.begin()));
	thrust::transform(connectors_.begin(), connectors_.end(), zip_output, get_spring_config_functor());
	thrust::pair<thrust::device_vector<double>::iterator, thrust::device_vector<double>::iterator> result = thrust::minmax_element(f.begin(), f.end());
	//std::cout << *result.first << " " << *result.second << std::endl;
	return thrust::make_pair(*result.first, *result.second);
	//for (int i = 0; i < q.size(); i++)
	//	std::cout << lam[i] << " ";
	//std::cout << std::endl;
}


std::tuple<thrust::device_vector<double>, // current position (q)
		   thrust::device_vector<double>, // previous position (qprev)
		   thrust::device_vector<double>, // velocity (v)
	       thrust::device_vector<double>, // radius (r)
		   thrust::device_vector<double>> // osmolarity (C)
DropletsSimCore_GPU::buildConfiguration() const
{
	thrust::device_vector<double> q, qprev, v, r, C;
	int ndofs = 2 * droplets_.size();
	q.resize(ndofs);
	v.resize(ndofs);
	qprev.resize(ndofs);
	r.resize(droplets_.size());
	C.resize(droplets_.size());

	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	getConfiguration<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(droplets_.data()), 
		                                      num_droplets_, 
		                                      thrust::raw_pointer_cast(q.data()), 
		                                      thrust::raw_pointer_cast(qprev.data()), 
		                                      thrust::raw_pointer_cast(v.data()), 
		                                      thrust::raw_pointer_cast(r.data()),
		                                      nullptr,
		                                      thrust::raw_pointer_cast(C.data()));
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
	return std::make_tuple(q, qprev, v, r, C);
}

void DropletsSimCore_GPU::unbuildConfiguration(const thrust::device_vector<double>& q, 
	                                           const thrust::device_vector<double>& v, 
		                                       const thrust::device_vector<double>& r,
	                                           const thrust::device_vector<double>& F)
{
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	updateConfiguration<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(droplets_.data()), 
		                                         thrust::raw_pointer_cast(force_on_droplets_.data()),
			                                     params_,
		                                         num_droplets_, 
		                                         thrust::raw_pointer_cast(q.data()), 
		                                         thrust::raw_pointer_cast(v.data()), 
		                                         thrust::raw_pointer_cast(r.data()), 
		                                         thrust::raw_pointer_cast(F.data()));
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
}

void DropletsSimCore_GPU::computeMassVector(thrust::device_vector<double>& mass) const
{
	mass.resize(num_droplets_ * 2);
	dim3 blockSize(1024);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	computeMassVector_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(droplets_.data()), 
		                                              num_droplets_, 
		                                              params_, 
		                                              thrust::raw_pointer_cast(mass.data()));
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
}

void DropletsSimCore_GPU::numericalIntegration(const thrust::device_vector<int>& droplets_bin_indices,
	                                           const thrust::device_vector<int>& bin_start,
	                                           const thrust::device_vector<int>& bin_end, 
	                                              thrust::device_vector<double>& q,
	                                              thrust::device_vector<double>& qprev, 
	                                              thrust::device_vector<double>& v,
	                                              thrust::device_vector<double>& F,
	                                              thrust::device_vector<double>& r,
	                                              thrust::device_vector<double>& C,
	                                              std::mutex& data_mutex)
{
	thrust::device_vector<double> J(num_droplets_, 0);
	thrust::device_vector<double> mass;
	computeMassVector(mass);
	//std::cout << "computeMassVector done" << std::endl;
	//update position
	qprev = q;
	thrust::device_vector<double> dtv(num_droplets_ * 2);
	thrust::transform(v.begin(), v.end(), dtv.begin(), thrust::placeholders::_1 * params_.timeStep);
	thrust::transform(q.begin(), q.end(), dtv.begin(), q.begin(), thrust::plus<double>());
	//std::cout << "update position done" << std::endl;

	computeForceAndFlow(q, qprev, v, r, C, droplets_bin_indices, bin_start, bin_end, F, J, data_mutex);
	//std::cout << "computeForceAndFlow done" << std::endl;
	if (params_.distributionPattern != 6)
		setupTimeStep();
	//update radius
	double dt = params_.timeStep;
	thrust::transform(r.begin(), r.end(), J.begin(), r.begin(),
		[dt] __device__(double r_val, double J_val)
	{
		return pow((4. / 3. * M_PI * r_val * r_val * r_val + dt * J_val) / (4. / 3. * M_PI), 1. / 3.);
	});
	//std::cout << "update radius done" << std::endl;


	//update velocity
	thrust::transform(v.begin(), v.end(), thrust::make_zip_iterator(thrust::make_tuple(mass.begin(), F.begin())), v.begin(),
	[dt] __device__(double v_i, thrust::tuple<double, double> bc_tuple) 
	{
		double mass_i = thrust::get<0>(bc_tuple);
		double F_i = thrust::get<1>(bc_tuple);
		return v_i + dt / mass_i * F_i;
	});
	//std::cout << "update velocity done" << std::endl;
}


void DropletsSimCore_GPU::numericalIntegrationWithSpring(thrust::device_vector<double>& q,
														 thrust::device_vector<double>& qprev,
														 thrust::device_vector<double>& v,
														 thrust::device_vector<double>& F,
														 thrust::device_vector<double>& r,
														 thrust::device_vector<double>& C,
	                                                                        std::mutex& data_mutex)
{
	thrust::device_vector<double> J(num_droplets_, 0);
	thrust::device_vector<double> mass;
	computeMassVector(mass);
	qprev = q;
	thrust::device_vector<double> dtv(num_droplets_ * 2);
	thrust::transform(v.begin(), v.end(), dtv.begin(), thrust::placeholders::_1 * params_.timeStep);
	thrust::transform(q.begin(), q.end(), dtv.begin(), q.begin(), thrust::plus<double>());

	computeForceAndFlowWIthSpring(q, qprev, v, r, C, F, J, data_mutex);
	setupTimeStep();

	//update radius
	double dt = params_.timeStep;
	thrust::transform(r.begin(), r.end(), J.begin(), r.begin(),
		[dt] __device__(double r_val, double J_val)
	{
		return pow((4. / 3. * M_PI * r_val * r_val * r_val + dt * J_val) / (4. / 3. * M_PI), 1. / 3.);
	});

	//update velocity
	thrust::transform(v.begin(), v.end(), thrust::make_zip_iterator(thrust::make_tuple(mass.begin(), F.begin())), v.begin(),
		[dt] __device__(double v_i, thrust::tuple<double, double> bc_tuple)
	{
		double mass_i = thrust::get<0>(bc_tuple);
		double F_i = thrust::get<1>(bc_tuple);
		return v_i + dt / mass_i * F_i;
	});
}


void DropletsSimCore_GPU::computeForceAndFlow(const thrust::device_vector<double>& q, 
	                                          const thrust::device_vector<double>& qprev, 
	                                          const thrust::device_vector<double>& v,
	                                          const thrust::device_vector<double>& r,
	                                          const thrust::device_vector<double>& C,
	                                             const thrust::device_vector<int>& droplets_bin_indices,
	                                             const thrust::device_vector<int>& bin_start,
	                                             const thrust::device_vector<int>& bin_end,
	                                                thrust::device_vector<double>& F, 
	                                                thrust::device_vector<double>& J,
	                                                                   std::mutex& data_mutex)
{
	if (params_.gravityEnabled)
	{
		dim3 blockSize(1024);
		dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
		processGravityForce<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(droplets_.data()), 
			                                         num_droplets_, 
			                                         params_, 
			                                         thrust::raw_pointer_cast(F.data()));
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
		}
	}
	if (params_.globalDampingEnabled)
	{
		dim3 blockSize(1024);
		dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
		processGlobalDampingForce<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(v.data()), 
			                                               num_droplets_, 
			                                               params_, 
			                                               thrust::raw_pointer_cast(F.data()));
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
		}
	}
	if (params_.floorEnabled)
	{
		dim3 blockSize(1024);
		dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
		processFloorForce<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(q.data()), 
			                                       thrust::raw_pointer_cast(v.data()),
			                                       thrust::raw_pointer_cast(r.data()),
			                                       num_droplets_, 
			                                       params_, 
			                                       thrust::raw_pointer_cast(F.data()));
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
		}
	}
	dim3 blockSize(512);
	dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
	thrust::device_vector<int> numConnectionsVec(1, 0);
	if(!params_.fineModel)
		processSpringForceAndFlow<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(q.data()), 
			                                               thrust::raw_pointer_cast(qprev.data()), 
			                                               thrust::raw_pointer_cast(r.data()), 
			                                               thrust::raw_pointer_cast(C.data()), 
			                                               thrust::raw_pointer_cast(droplets_bin_indices.data()), 
			                                               thrust::raw_pointer_cast(bin_start.data()), 
			                                               thrust::raw_pointer_cast(bin_end.data()), 
			                                               num_droplets_, 
			                                               params_.cellSize, 
			                                               params_.numCells_x, 
			                                               params_.numCells_y, 
			                                               params_, 
			                                               thrust::raw_pointer_cast(F.data()), 
			                                               thrust::raw_pointer_cast(J.data()),
			                                               thrust::raw_pointer_cast(numConnectionsVec.data()));
	else
		processSpringForceAndFlow_fineModel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(q.data()), 
			                                                         thrust::raw_pointer_cast(qprev.data()), 
			                                                         thrust::raw_pointer_cast(r.data()), 
			                                                         thrust::raw_pointer_cast(C.data()), 
			                                                         thrust::raw_pointer_cast(droplets_bin_indices.data()), 
			                                                         thrust::raw_pointer_cast(bin_start.data()), 
			                                                         thrust::raw_pointer_cast(bin_end.data()), 
			                                                         num_droplets_, 
			                                                         params_.cellSize, 
			                                                         params_.numCells_x, 
			                                                         params_.numCells_y, 
			                                                         params_, 
			                                                         thrust::raw_pointer_cast(F.data()), 
			                                                         thrust::raw_pointer_cast(J.data()),
			                                                         thrust::raw_pointer_cast(numConnectionsVec.data()));
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
	
	{
		std::lock_guard<std::mutex> lock(data_mutex);
		number_of_connections_on_droplets_.resize(num_droplets_);
		thrust::fill(number_of_connections_on_droplets_.begin(), number_of_connections_on_droplets_.end(), 0);
		connectors_.resize(numConnectionsVec[0]);
		numConnectionsVec[0] = 0;
		//std::cout << connectors_.size() << std::endl;
		dim3 blockSize2(512);
		dim3 gridSize2((num_droplets_ + blockSize2.x - 1) / blockSize2.x);
		setupConnections << <gridSize2, blockSize2 >> > (thrust::raw_pointer_cast(q.data()),
			                                             thrust::raw_pointer_cast(r.data()), 
			                                             thrust::raw_pointer_cast(droplets_bin_indices.data()),
			                                             thrust::raw_pointer_cast(bin_start.data()),
			                                             thrust::raw_pointer_cast(bin_end.data()),
			                                             num_droplets_,
			                                             params_.cellSize,
			                                             params_.numCells_x,
			                                             params_.numCells_y,
			                                             params_,
			                                             thrust::raw_pointer_cast(numConnectionsVec.data()),
			                                             thrust::raw_pointer_cast(number_of_connections_on_droplets_.data()),
			                                             thrust::raw_pointer_cast(connectors_.data()));
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
		}
	}
	
	/*
	for (int i = 0; i < num_droplets_; i++)
	{
		std::cout << F[i * 2] << " " << F[i * 2 + 1] << " ";
	}
	std::cout << std::endl;
	*/
	
	/*
	thrust::host_vector<Spring_GPU> connectors_host = connectors_;
	for (int i = 0; i < connectors_.size(); i++)
	{
		std::cout << connectors_host[i].isInTension << " ";
	}
	std::cout << std::endl;
	*/
}

void DropletsSimCore_GPU::computeForceAndFlowWIthSpring(const thrust::device_vector<double>& q, 
	                                                    const thrust::device_vector<double>& qprev, 
	                                                    const thrust::device_vector<double>& v, 
	                                                    const thrust::device_vector<double>& r, 
	                                                    const thrust::device_vector<double>& C, 
	                                                          thrust::device_vector<double>& F, 
	                                                          thrust::device_vector<double>& J,
	                                                                             std::mutex& data_mutex)
{
	if (params_.gravityEnabled)
	{
		dim3 blockSize(1024);
		dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
		processGravityForce << <gridSize, blockSize >> > (thrust::raw_pointer_cast(droplets_.data()),
			                                              num_droplets_,
			                                              params_,
			                                              thrust::raw_pointer_cast(F.data()));
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
		}
	}
	if (params_.globalDampingEnabled)
	{
		dim3 blockSize(1024);
		dim3 gridSize((num_droplets_ + blockSize.x - 1) / blockSize.x);
		processGlobalDampingForce << <gridSize, blockSize >> > (thrust::raw_pointer_cast(v.data()),
			                                                    num_droplets_,
			                                                    params_,
			                                                    thrust::raw_pointer_cast(F.data()));
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
		}
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
		}
		/*
		for (int i = 0; i < num_droplets_; i++)
		{
			std::cout << F[i * 2] << " " << F[i * 2 + 1] << std::endl;
		}
		std::cout << std::endl;
		*/
		
	}
	dim3 blockSize(1024);
	dim3 gridSize((connectors_.size() + blockSize.x - 1) / blockSize.x);
	if (params_.fixedPermeability)
	{
		processSpringForceAndFlowWithoutCollosionDetection << <gridSize, blockSize >> > (thrust::raw_pointer_cast(q.data()),
			                                                                             thrust::raw_pointer_cast(qprev.data()),
			                                                                             thrust::raw_pointer_cast(r.data()),
			                                                                             thrust::raw_pointer_cast(C.data()),
			                                                                             connectors_.size(),
			                                                                             params_,
			                                                                             thrust::raw_pointer_cast(connectors_.data()),
			                                                                             thrust::raw_pointer_cast(F.data()),
			                                                                             thrust::raw_pointer_cast(J.data()));
	}
	else
	{
		processSpringForceAndFlowWithoutCollosionDetection_osm << <gridSize, blockSize >> > (thrust::raw_pointer_cast(q.data()),
			                                                                                 thrust::raw_pointer_cast(qprev.data()),
			                                                                                 thrust::raw_pointer_cast(r.data()),
			                                                                                 thrust::raw_pointer_cast(C.data()),
			                                                                                 connectors_.size(),
			                                                                                 params_,
			                                                                                 thrust::raw_pointer_cast(connectors_.data()),
			                                                                                 thrust::raw_pointer_cast(F.data()),
			                                                                                 thrust::raw_pointer_cast(J.data()));
	}
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error during device synchronization: " << cudaGetErrorString(err) << std::endl;
	}
	/*
	for (int i = 0; i < num_droplets_; i++)
	{
		std::cout << F[i * 2] << " " << F[i * 2 + 1] << std::endl;
	}
	std::cout << std::endl;
	*/
}

void DropletsSimCore_GPU::removeSnappedSprings()
{
	thrust::device_vector<Spring_GPU>::iterator new_end = thrust::remove_if(connectors_.begin(), connectors_.end(), is_snapped_functor());
	connectors_.resize(new_end - connectors_.begin());
}

void DropletsSimCore_GPU::removeDropletsOutsideDomain()
{
	thrust::device_vector<Droplet_GPU>::iterator new_end = thrust::remove_if(droplets_.begin(), droplets_.end(), is_outside_domain_functor(params_.domainBoundingBox));
	droplets_.resize(new_end - droplets_.begin());
	num_droplets_ = droplets_.size();
}







