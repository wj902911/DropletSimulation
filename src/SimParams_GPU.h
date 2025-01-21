#pragma once

#include <eigen/Core>
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
#ifndef DROPLET_NUM_X
#define DROPLET_NUM_X 41
#endif
#ifndef DROPLET_NUM_Y
#define DROPLET_NUM_Y 8
#endif
*/

enum LOAD_TYPE
{
	force,
	displacement,
	permeability,
	velocity
};

enum DropletType
{
	DPhPC,
	PB_PEO
};


struct SimParameters_GPU
{
public:
	SimParameters_GPU(int numDropx = 100);

	~SimParameters_GPU();

	double sampleSizeFactor = 1.0;
	bool isShear = false;
	bool isBulk = false;

	bool fineModel = false;
	double sigma = 2.73e-10;
	double epsilon = 4.91e-21;
	double m = 2.99e-26;
	//double ir = 4.0;

	double radiiFactor = 1.0;
	//bool specifyLocation = true;
	int dropletType = DropletType::DPhPC;
	double initialDropletRadius = 1.528e-10; //3.14e-6m
	double mean_radius = 1.528e-10;
	//double E = 0.000078;//muN/mum^2
	double gamma_m = 0.001; //N/m
	double theta_eq = 41; //degree
	//double E = 78;//Pa
	//double spesimenThickness = 5000;//mum
	//double spesimenLength = 25000;//mum
	double specimenLength = 0.025;//m
	//double dropletDensity = 1e-9;//mg/mum^3   t in ms
	//double dropletDensity = 1e9;//mg/m^3   t in ms
	double dropletDensity = 1000;//kg/m^3   t in s

	int distributionPattern = 2;
	std::string path_to_diameter="resource/";
	std::string path_to_position_x = "resource/";
	std::string path_to_position_y = "resource/";
	std::string path_to_area = "resource/";
	
	double diameterEnlargeRatio = 1.00;

	int loadType = LOAD_TYPE::displacement;
	bool calculateStiffness = true;
	bool physicsLoopDelay = false;
	bool fixDroplets = false;
	bool startApplyDisplacement = false;
	bool floorEnabled = false;
	bool wallEnabled = false;
	double leftWallX = 0.0;
	double rightWallX = 0.0;
	double wallOffset = mean_radius * 3;
	double floorY = 0.0;
	bool applyDisplacementAfterSettling = true;
	double settlingTime = 0.00;//s
	bool compressionBeforeStretching = false;
	double compressionTime = 0.01;//s
	double maxCompressionDisplacement = mean_radius * 3;//m
	double compressionSpeed = maxCompressionDisplacement / compressionTime;//m/s
	bool compressed = false;
	//int displacementSign = 1;
	double boundaryOffset = mean_radius * 3;

	bool   springsEnabled = true;//true
	bool   canSnap = false;//true
	bool   springDampingEnabled = true;//false
	bool   globalDampingEnabled = true;//true
	double springDamplingRatio = 0.1;//these two sum up less than 0.5
	double globalDampingRatio = 0.01;//these two sum up less than 0.5
	//double maxAddhesionDistRatio = 0.9;
	double toltalSeperationLengthRatio = 1.0;//1.1

	bool   gravityEnabled = false;
	double gravityG = -9.8;

	bool mechanicalLoadingEnabled = false;
	bool fixedPermeability = false;
	double maxPermeability = 0.0001;//0.01, 0.05
	double maxDisplacement = mean_radius * 3;//m
	double maxCompressionStrain = 0.01;
	double maxTensileStrain = 0.001;
	double increaseTime = 0.1;//s
	double stretchingSpeed = maxDisplacement / increaseTime;//m/s
	double loadingSpeed = stretchingSpeed;
	bool   flowEnabled = true;
	//bool   addConnectorWhenTouching = true;


	//bool   fixOneDroplet = false;

	bool   outputData = 0;
	bool   saveFrames = 0;
	double saveFrameInterval = 0.0002;//s
	double cameraDistanceRatio = 1.0;

	double criticalConcentrationDifference = 1e-3;

	//based on calculation:
	double timeStep = 0.0; //s
	double cellSize = 0.0;
	double boundingBoxWidth = 0.0;
	double boundingBoxHeight = 0.0;
	double2 boundingBoxCenter = { 0.0,0.0 };
	double2 boundingBoxLowerLeft = { 0.0,0.0 };
	int    numCells_x = 0;
	int    numCells_y = 0;

	double dropletMass = 1.0;//1.0
	double springStiffness = 1.0;//1
	double springStiffness_com = 1.0;//1
	int    numDroplets_x = 20;//320
	int    numDroplets_y = 2;//64
	int    numDroplets = numDroplets_x * numDroplets_y;//20480
	double lowerSubValue = 1.0;
	double upperSubValue = 4.0;
	double dampingStiffness = 1e-10;
	double globalDamping = 0.0;//2
	double permeability = 0.0;//0.01, 0.05
	double displacement = 2.5e-5;//m
	double delta_displacement = 0.0001;
	double delta_permeability = 0.0000001;
	double SpringRestLengthRatio = 0.8;//0.8
	double specimenThickness = specimenLength / 5.0;//m
	double4 initialBoundingBox = { 0.0,0.0,0.0,0.0 };
	double4 domainBoundingBox = { 0.0,0.0,0.0,0.0 };
	thrust::device_vector<int> displacementDropletUid;
	thrust::device_vector<int> outputDropletUids;
	thrust::device_vector<int> outputSectionDropletUid;
	thrust::device_vector<int> outputSectionDropletUid2;
	thrust::device_vector<double> dropletRadii = { initialDropletRadius ,initialDropletRadius };
	thrust::device_vector<double> dropletPositions = { 
		0.0,0.0,
		initialDropletRadius * 2.01,0.0 };
	thrust::device_vector<int> fixeds = { 0,0 };
	thrust::device_vector<double> initialDropletsVelocity = { 5e-5,0.0,-5e-5,0.0 };
	thrust::device_vector<double> initialDropletsSubValue = { pow(initialDropletRadius * 2.0, 3), pow(initialDropletRadius * 2.0, 3) * 0.25 };
};

