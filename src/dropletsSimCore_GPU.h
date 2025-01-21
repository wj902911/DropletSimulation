#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "dropletsSimObjects_GPU.h"
#include "device_launch_parameters.h"

//#define NUM_DROPLETS DROPLET_NUM_X * DROPLET_NUM_Y
//#define NUM_CONNECTORS DROPLET_NUM_X * DROPLET_NUM_Y * (DROPLET_NUM_X * DROPLET_NUM_Y - 1) / 2

__global__ void initDroplets(Droplet_GPU* droplets, const SimParameters_GPU params);
__global__ void initDroplets_1D(Droplet_GPU* droplets, const SimParameters_GPU params);

__global__ void assignToBins(Droplet_GPU* droplets, 
	                                  int num_droplets, 
	                                 int* droplets_bin_indices, 
	                               double cellSize, 
	                                  int numCellsX, 
	                                  int numCellsY,
	                              double2 gridOrigin);

__global__ void identifyBinBoundaries(int* droplets_bin_indices, 
	                                   int num_droplets, 
	                                  int* bin_start, 
	                                  int* bin_end);

__global__ void getConfiguration(const Droplet_GPU* droplets, 
	                                      const int num_droplets,
	                                        double* q, 
	                                        double* qprev, 
	                                        double* v, 
	                                        double* r,
	                                        double* rprev,
	                                        double* C);

__global__ void processGravityForce(const Droplet_GPU* droplets, 
	                                         const int num_droplets, 
	                           const SimParameters_GPU params,
	                                           double* F);

__global__ void processGlobalDampingForce(const double* v, 
	                                          const int num_droplets, 
	                            const SimParameters_GPU params, 
	                                            double* F);

__global__ void processFloorForce(const double* q, 
	                              const double* v, 
	                              const double* r, 
	                                  const int num_droplets, 
	                    const SimParameters_GPU params, 
	                                    double* F);


__global__ void getNumberOfConnections(const Droplet_GPU* droplets_,
											   const int* droplets_bin_indices,
											   const int* bin_start,
											   const int* bin_end,
												const int num_droplets,
											 const double cellSize,
												const int numCellsX,
												const int numCellsY,
								  const SimParameters_GPU params,
											         int* numConnections,
	                                                 int* numOutputConnections);

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
							                   		           int* numConnections,
	                                                           int* numOutputConnections);
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
										Spring_GPU* connectors);


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
    				   			   Spring_GPU* connectors);

/*
__global__ void setupConnections_fineModel(const Droplet_GPU* droplets_,
							           			   const int* droplets_bin_indices,
							           			   const int* bin_start,
							           			   const int* bin_end,
							           			    const int num_droplets,
							           		     const double cellSize,
							           			    const int numCell,
                                                    const int numCellsY,
							          const SimParameters_GPU params,
							           			         int* numConnections,
	                                                     int* numOutputConnections,
	                                                     int* output_spring_index,
	                                                     int* num_connections_per_droplet,
							           			  Spring_GPU* connectors);


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
    				             			 Spring_GPU* connectors);

*/

__global__ void processSpringForceAndFlowWithoutCollosionDetection(const double* q,
																   const double* qprev,
																   const double* r,
																   const double* C,
																	   const int num_connectors,
														 const SimParameters_GPU params,
																	 Spring_GPU* connectors,
																		 double* F,
																		 double* J);

__global__ void processSpringForceAndFlowWithoutCollosionDetection_osm(const double* q,
																       const double* qprev,
																       const double* r,
																       const double* C,
																	       const int num_connectors,
														     const SimParameters_GPU params,
																	     Spring_GPU* connectors,
																		     double* F,
																		     double* J);



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
	                                               int* numConnection);

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
	                                                         int* numConnection);

__global__ void updateRadius(const double* J,
	                             const int num_droplets,
	               const SimParameters_GPU params,
	                               double* r);

__global__ void updateConfiguration(Droplet_GPU* droplets,
	                                    double2* force_on_droplets,
			             const SimParameters_GPU params,
	                                   const int num_droplets,
	                               const double* q,
	                               const double* v,
	                               const double* r,
	                               const double* F);

__global__ void convertToDouble3(const double* q, 
	                                 const int num_droplets, 
	                                  double3* q3);

__global__ void computeMassVector_kernel(Droplet_GPU* droplets, 
	                                        const int num_droplets,
	                          const SimParameters_GPU params, 
	                                          double* mass);

__device__ void atomicDouble2Exch(double2* addr, double2 value);

__global__ void findePositionByUid_kernel(const Droplet_GPU* droplets, uint32_t numDroplets, uint32_t uid, double2* result);

__global__ void applyDisplacement_kernel(const Droplet_GPU* droplets, double* q, const int* dropletUids, double displacement, int numDroplets, int numUids);
__global__ void applyDisplacement_y_kernel(const Droplet_GPU* droplets, double* q, const int* dropletUids, double displacement, int numDroplets, int numUids);
__global__ void applyDisplacement_pureShear_kernel(const Droplet_GPU* droplets, double* q, const int* dropletUids, double displacement, int numDroplets, int numUids, double* q0);
__global__ void applyDisplacement_bulk_kernel(const Droplet_GPU* droplets, double* q, const int* dropletUids, double displacement, int numDroplets, int numUids, double* q0);

__global__ void fixDropletsWithinArea_kernel(Droplet_GPU* droplets, const int numDroplets, const double2 xRange, const double2 yRange);

__global__ void getNumberOfDisplacementDropletsWithinArea_kernel(const Droplet_GPU* droplets, const int numDroplets, const double2 xRange, const double2 yRange, int* numDropletsWithinArea);
__global__ void getNumberOfDisplacementDropletsOutsideArea_kernel(const Droplet_GPU* droplets, const int numDroplets, const double2 xRange, const double2 yRange, int* numDropletsWithinArea);

__global__ void assignDisplacementDropletsWithinArea_kernel(Droplet_GPU* droplets, const int numDroplets, const double2 xRange, const double2 yRange, int* dropletUids, int* numDropletsWithinArea);
__global__ void assignOutputDropletsWithinArea_kernel(Droplet_GPU* droplets, const int numDroplets, const double2 xRange, const double2 yRange, int* dropletUids, int* numDropletsWithinArea);
__global__ void assignDisplacementDropletsOutsideArea_kernel(Droplet_GPU* droplets, const int numDroplets, const double2 xRange, const double2 yRange, int* dropletUids, int* numDropletsWithinArea);

class DropletsSimCore_GPU
{
public:
	DropletsSimCore_GPU();
	DropletsSimCore_GPU(int numDropx);
	~DropletsSimCore_GPU();

	void initSimulation();
	//void initSimulationWithSpring();
	void simulateOneStep(std::mutex& data_mutex);

	void simulateOneStepWithSpring(std::mutex& data_mutex);

	
	void physicsLoop(bool& startFlag,
		std::atomic<bool>& stopFlag, 
		std::mutex& data_mutex);
	

	double getTime() const
	{
		return time_;
	}

	double getTimestep() const
	{
		return params_.timeStep;
	}


	double getDisplacement() const
	{
		return displacement_;
	}

	int getNumConnectors() const
	{
		return connectors_.size();
	}

	double2 getGlobalCenter() const;
	double2 getGlobalSize() const;
	void findMaxVelocityDroplet(double& max_velocity, int& max_velocity_droplet) const;
	double getMaxConcentration() const;
	double getMinConcentration() const;
	thrust::pair<double, double> getConcentration(thrust::device_vector<double>& C) const;
	double getMaxRadiusIncreaseSpeed() const;
	double getMaxRadiusDecreaseSpeed() const;
	thrust::pair<double,double> getMaxRadiusChangeSpeed() const;
	double getTotalKineticEnergy() const;
	double2 getDropletPosition(uint32_t uid) const;
	double getEndToEndDistance() const;
	int getIndexofDroplet(uint32_t uid) const;
	double2 getTotalForce() const;
	double getMaxDropletTotalForce() const;
	double4 getCurrentBoundingBox() const;

	void setupTimeStep();

	void adjustLoad();
	void applyDisplacement(thrust::device_vector<double>& q);
	void applyDisplacement_y(thrust::device_vector<double>& q);
	void applyDisplacement_pureShear(thrust::device_vector<double>& q);
	void applyDisplacement_bulk(thrust::device_vector<double>& q);


	int getNumberOfDisplacementDropletsWithinArea(double2 xRange, double2 yRange);
	int getNumberOfOutputDropletsWithinArea(double2 xRange, double2 yRange);
	int getNumberOfDisplacementDropletsOutsideArea(double2 xRange, double2 yRange);
	void assignDisplacementDropletsWithinArea(double2 xRange, double2 yRange);
	void assignOutputDropletsWithinArea(double2 xRange, double2 yRange);
	void assignDisplacementDropletsOutsideArea(double2 xRange, double2 yRange);

	void applyVelocity(thrust::device_vector<double>& v);
	void fixDropletsWithinArea(double2 xRange, double2 yRange);

	void getCurrentDropletPositionRadiusConcentration(thrust::device_vector<double3>& q,
		                                              thrust::device_vector<double>& r,
		                                              thrust::device_vector<double>& C) const;

	void getAllDropletPositionAsD3(thrust::device_vector<double3>& q) const;
	void getAllDropletRadius(thrust::device_vector<double>& r) const;
	int getAverageNumberOfConnections() const;

	thrust::pair<double, double> getAllSpringConfiguration(thrust::device_vector<double3>& q,
		                                                    thrust::device_vector<double>& r,
		                                                    thrust::device_vector<double>& lam, 
		                                                    thrust::device_vector<double>& f) const;

	std::tuple<thrust::device_vector<double>, // current position (q)
		       thrust::device_vector<double>, // previous position (qprev)
		       thrust::device_vector<double>, // velocity (v)
		       thrust::device_vector<double>, // radius (r)
		       thrust::device_vector<double>> // osmolarity (C)
	buildConfiguration() const;

	void unbuildConfiguration(const thrust::device_vector<double>& q,
		                      const thrust::device_vector<double>& v,
		                      const thrust::device_vector<double>& r,
	                          const thrust::device_vector<double>& F);

	void computeMassVector(thrust::device_vector<double>& mass) const;

	
	void numericalIntegration(const thrust::device_vector<int>& droplets_bin_indices,
		                      const thrust::device_vector<int>& bin_start,
		                      const thrust::device_vector<int>& bin_end,
		                         thrust::device_vector<double>& q,
		                         thrust::device_vector<double>& qprev,
					             thrust::device_vector<double>& v,
		                         thrust::device_vector<double>& F,
		                         thrust::device_vector<double>& r,
		                         thrust::device_vector<double>& C,
		                                            std::mutex& data_mutex);

	

	void computeForceAndFlow(const thrust::device_vector<double>& q,
		                     const thrust::device_vector<double>& qprev,
		                     const thrust::device_vector<double>& v,
		                     const thrust::device_vector<double>& r,
		                     const thrust::device_vector<double>& C,
		                        const thrust::device_vector<int>& droplets_bin_indices,
		                        const thrust::device_vector<int>& bin_start,
		                        const thrust::device_vector<int>& bin_end,
		                           thrust::device_vector<double>& F,
		                           thrust::device_vector<double>& J,
	                                                  std::mutex& data_mutex);

	

	
	void numericalIntegrationWithSpring(thrust::device_vector<double>& q,
		                                thrust::device_vector<double>& qprev,
		                                thrust::device_vector<double>& v,
		                                thrust::device_vector<double>& F,
		                                thrust::device_vector<double>& r,
		                                thrust::device_vector<double>& C,
	                                                       std::mutex& data_mutex);
    
	void computeForceAndFlowWIthSpring(const thrust::device_vector<double>& q,
		                               const thrust::device_vector<double>& qprev,
		                               const thrust::device_vector<double>& v,
		                               const thrust::device_vector<double>& r,
		                               const thrust::device_vector<double>& C,
		                                     thrust::device_vector<double>& F,
		                                     thrust::device_vector<double>& J,
	                                                            std::mutex& data_mutex);
    
	void removeSnappedSprings();
	void removeDropletsOutsideDomain();
	
	
	
	
	
//private:
	//uint32_t droplet_unique_id_;
	SimParameters_GPU params_;
	double time_ = 0.0;
	double displacement_ = 0.0;
	int num_droplets_ =0.0;
	thrust::device_vector<Droplet_GPU> droplets_;
	thrust::device_vector <Spring_GPU> connectors_;
	thrust::device_vector <int> output_spring_index_;
	thrust::device_vector<double2> force_on_droplets_;
	thrust::device_vector<int> number_of_connections_on_droplets_;
	//int num_connectors_ = NUM_CONNECTORS;
	//thrust::device_vector<int2> connected_droplets_;

    //void removeSnappedSproing();
	//void tuchDetection();
};