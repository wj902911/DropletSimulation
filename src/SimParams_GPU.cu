#pragma once

#include "SimParams_GPU.h"
#include <random>
#include <thrust/extrema.h>
#include <fstream>

SimParameters_GPU::SimParameters_GPU(int numDropx)
{
	numDroplets_x = numDropx;

	switch (dropletType)
	{
	case DropletType::DPhPC:
	{
		path_to_diameter += "DPhPC_diameter.txt";
		path_to_position_x += "DPhPC_pos_x.txt";
		path_to_position_y += "DPhPC_pos_y.txt";
		path_to_area += "DPhPC_area.txt";
		break;
	}
	case DropletType::PB_PEO:
	{
		path_to_diameter += "PB_PEO_diameter.txt";
		path_to_position_x += "PB_PEO_pos_x.txt";
		path_to_position_y += "PB_PEO_pos_y.txt";
		path_to_area += "PB_PEO_area.txt";
		break;
	}
	}
	initialDropletRadius = specimenLength / SpringRestLengthRatio / numDroplets_x / 2.;
	if (distributionPattern == 6)
		dropletMass = m;
	else
		dropletMass = 4. / 3. * M_PI * pow(initialDropletRadius, 3) * dropletDensity;
	//springStiffness = E * spesimenThickness * initialDropletRadius * 2. * (numDroplets_x - 1) / (numDroplets_y + (numDroplets_y - 1) * 0.5) / spesimenLength;
	double cos_theta_eq_rad = cos(theta_eq * M_PI / 180.0);
	if (calculateStiffness)
		springStiffness = gamma_m * M_PI / 4.0 * (3.0 / cos_theta_eq_rad + 2 * cos_theta_eq_rad - pow(cos_theta_eq_rad, 3));
	double restLength = 4 * initialDropletRadius / pow(12.0 / (cos_theta_eq_rad * cos_theta_eq_rad) - 4, 1.0 / 3.0);
	SpringRestLengthRatio = restLength / (initialDropletRadius * 2.0);
	switch (distributionPattern)
	{
	case 1:
	{
		numDroplets_x = floor(specimenLength / restLength);
		numDroplets_y = floor(specimenThickness / restLength);
		numDroplets= numDroplets_x * numDroplets_y;
		break;
	}
	case 2:
	{
		//numDroplets_x = floor(specimenLength / restLength);
		numDroplets_y = floor(specimenThickness / (restLength * cosf(M_PI / 6.)));
		numDroplets = numDroplets_x * numDroplets_y;
		break;
	}
	case 3:
	{ 
		break; 
	}
	case 4:
	{
		//int N = numDroplets_x * numDroplets_y;
		double mu = std::log(mean_radius);
		//double mu = mean_radius;
		double sigma = 0.5;
		std::vector<double> host_radii;
		std::vector<double> host_pos;
		std::random_device rd;
		std::mt19937 gen(rd());
		std::lognormal_distribution<double> dis(mu, sigma);
		//for (int i = 0; i < N; i++)
		//{
			//host_radii[i] = dis(gen);
		//}
		std::vector<int> rowStartIndex(numDroplets_y + 1);
		rowStartIndex[0] = 0;
		host_radii.push_back(dis(gen));
		double maxRadiusCurrentRow = 0.0;
		//double maxRadiusLastRow = maxRadiusCurrentRow;
		double maxRadiusLastRow = 0.0;
		double currentY = 0.0;
		double currentX = 0.0;
		host_pos.push_back(currentX);
		host_pos.push_back(currentY);
		initialDropletsVelocity.clear();
		initialDropletsVelocity.push_back(0.0);
		initialDropletsVelocity.push_back(0.0);
		fixeds.clear();
		fixeds.push_back(false);
		for (int i = 1; currentX < numDroplets_x * mean_radius; i++)
		{
			host_radii.push_back(dis(gen));
			currentX += (host_radii[i - 1] + host_radii[i]) * SpringRestLengthRatio;
			host_pos.push_back(currentX);
			host_pos.push_back(currentY);
			initialDropletsVelocity.push_back(0.0);
			initialDropletsVelocity.push_back(0.0);
			//if (i == numDroplets_x - 1)
				//fixeds.push_back(true);
			//else
				fixeds.push_back(false);
		}
		rowStartIndex[1] = host_radii.size();
		maxRadiusCurrentRow = *std::max_element(host_radii.begin(), host_radii.end());
		//currentY += maxRadiusCurrentRow;
		currentY += maxRadiusCurrentRow + maxRadiusLastRow;
		maxRadiusLastRow = maxRadiusCurrentRow;
		for (int i = 0; i < host_radii.size(); i++)
		{
			host_pos[i * 2 + 1] = currentY;
		}
		for (int i = 1; i < numDroplets_y; i++)
		{
			host_radii.push_back(dis(gen));
			//maxRadiusCurrentRow = *std::max_element(host_radii.begin() + i * numDroplets_x, host_radii.begin() + (i + 1) * numDroplets_x);
			//maxRadiusCurrentRow = host_radii.back();
			currentX = 0.0;
			//currentY += (host_radii[(i - 1) * numDroplets_x] + host_radii[i * numDroplets_x]) * SpringRestLengthRatio;
			//currentY += maxRadiusLastRow + maxRadiusCurrentRow;
			host_pos.push_back(currentX);
			host_pos.push_back(currentY);
			for (int j = 1; currentX < numDroplets_x * mean_radius; j++)
			{
				host_radii.push_back(dis(gen));
				//currentX += (host_radii[i * numDroplets_x + j - 1] + host_radii[i * numDroplets_x + j]) * SpringRestLengthRatio;
				currentX += (host_radii[rowStartIndex[i] + j - 1] + host_radii[rowStartIndex[i] + j]) * SpringRestLengthRatio;
				host_pos.push_back(currentX);
				host_pos.push_back(currentY);
				initialDropletsVelocity.push_back(0.0);
				initialDropletsVelocity.push_back(0.0);
				fixeds.push_back(false);
			}
			maxRadiusCurrentRow= *std::max_element(host_radii.begin() + rowStartIndex[i], host_radii.end());
			//currentY += maxRadiusCurrentRow;
			currentY += maxRadiusCurrentRow + maxRadiusLastRow;
			maxRadiusLastRow = maxRadiusCurrentRow;
			rowStartIndex[i + 1] = host_radii.size();
			for (int j = rowStartIndex[i]; j < rowStartIndex[i + 1]; j++)
			{
				host_pos[j * 2 + 1] = currentY;
			}
		}
		dropletRadii = host_radii;
		dropletPositions = host_pos;
		numDroplets = host_radii.size();
		std::cout << "numDroplets: " << numDroplets << std::endl;
		std::cout << "numDroplets row 1: " << rowStartIndex[1] << std::endl;
		std::cout << "numDroplets row 2: " << rowStartIndex[2] - rowStartIndex[1] << std::endl;
		break;
	}
	case 5:
	{
		dropletRadii.clear();
		dropletPositions.clear();
		fixeds.clear();
		std::ifstream file(path_to_diameter);
		std::ifstream file2(path_to_area);
		if (!file.is_open() || !file2.is_open())
		{
			std::cout << "Error: Could not open the file at " << path_to_diameter << std::endl;
			exit(1);
		}
		double diameter;
		double area;
		while (file >> diameter && file2 >> area)
		{
			double equivalentDiameter = sqrt(4.0 * area / M_PI);
			double minDiameter = 0;
			double maxDiameter = 0;
			std::tie(minDiameter, maxDiameter) = std::minmax(diameter, equivalentDiameter);
			dropletRadii.push_back((minDiameter + (maxDiameter - minDiameter) * radiiFactor) * 1e-6 * 0.5 * diameterEnlargeRatio * sampleSizeFactor);
			fixeds.push_back(false);
		}
		file.close();
		file2.close();
		numDroplets = dropletRadii.size();
		file.open(path_to_position_x);
		file2.open(path_to_position_y);
		if (!file.is_open() || !file2.is_open())
		{
			std::cout << "Error: Could not open the file at " << path_to_position_x << " or " << path_to_position_y << std::endl;
			exit(1);
		}
		double x, y;
		while (file >> x && file2 >> y)
		{
			dropletPositions.push_back(x * 1e-6 * sampleSizeFactor);
			dropletPositions.push_back(y * 1e-6 * sampleSizeFactor);
		}
		file.close();
		file2.close();
		break;
	}
	case 6:
	{
		//numDroplets = numDroplets_x * numDroplets_y;
		break;
	}
	}
	switch (distributionPattern)
	{
	case 1: case 2: case 3: case 4:
	{
		for (int i = 0; i < numDroplets_y; i++)
		{
			displacementDropletUid.push_back((i + 1) * numDroplets_x - 1);
			outputSectionDropletUid.push_back((i + 1) * numDroplets_x - numDroplets_x / 2.0);
			outputSectionDropletUid2.push_back((i + 1) * numDroplets_x - numDroplets_x / 2.0 + 1);
			//std::cout << outputSectionDropletUid[i] << std::endl;
		}
		break;
	}
	case 5: case 6:
	{
		break;
	}
	}
	
	//numDroplets_y = floor(specimenThickness / (SpringRestLengthRatio * cosf(M_PI / 6.)) / initialDropletRadius / 2.);
	double minDropletRadius = *thrust::min_element(dropletRadii.begin(), dropletRadii.end());
	double minMass = 4. / 3. * M_PI * pow(minDropletRadius, 3) * dropletDensity;
	timeStep = M_PI * sqrt(minMass / springStiffness) / 20;
	if(distributionPattern == 6)
		timeStep = 1e-15;
	upperSubValue = pow(initialDropletRadius * 2.0, 3);
	lowerSubValue = upperSubValue * 0.25;
	//lowerSubValue = upperSubValue;

	//if (fixedPermeability)
		//permeability = delta_permeability = maxPermeability * timeStep / increaseTime;
	//else
		//permeability = maxPermeability;
	displacement = 0.0;
	maxDisplacement = minDropletRadius * 0.1;
	stretchingSpeed = maxDisplacement / increaseTime;
	delta_displacement = stretchingSpeed * timeStep;

	//int numDroplets = numDroplets_x * numDroplets_y;
	double totalMass = dropletMass * numDroplets;
	double globalStiffness = springStiffness / static_cast<double>(numDroplets);
	//double globalStiffness = E * spesimenThickness * (initialDropletRadius * 2.0) / spesimenLength;
	if (globalDampingEnabled)
	{
		globalDamping = 2. * sqrt(totalMass * globalStiffness) * globalDampingRatio / numDroplets;
		dampingStiffness = 2. * sqrt(minMass * springStiffness) * springDamplingRatio - globalDamping;
	}
	else
	{
		globalDamping = 0.0;
		dampingStiffness = 2. * sqrt(minMass * springStiffness) * springDamplingRatio;
	}
	double rf = 0.0;
	if (flowEnabled)
	{
		double v0 = 4. / 3. * M_PI * pow(initialDropletRadius, 3);
		double dv = (upperSubValue - lowerSubValue) / (upperSubValue + lowerSubValue) * v0;;
		double vi = v0 + dv;
		rf = cbrt(vi * 3. / (4. * M_PI));
	}
	else if (distributionPattern == 4 || distributionPattern == 5)
	{
		rf= *thrust::max_element(dropletRadii.begin(), dropletRadii.end());
	}
	else if (distributionPattern == 6)
	{
		rf = initialDropletRadius * toltalSeperationLengthRatio;
	}
	else
	{
		rf = initialDropletRadius;
	}
	cellSize = 2. * rf * 1.5;

	switch (distributionPattern)
	{
	case 4: case 5:
	{
		double min_x = std::numeric_limits<double>::max();
		double max_x = std::numeric_limits<double>::min();
		double min_y = std::numeric_limits<double>::max();
		double max_y = std::numeric_limits<double>::min();
		for (int i = 0; i < dropletRadii.size(); i++)
		{
			if (dropletPositions[i * 2] - dropletRadii[i] < min_x)
				min_x = dropletPositions[i * 2] - dropletRadii[i];
			if (dropletPositions[i * 2] + dropletRadii[i] > max_x)
				max_x = dropletPositions[i * 2] + dropletRadii[i];
			if (dropletPositions[i * 2 + 1] - dropletRadii[i] < min_y)
				min_y = dropletPositions[i * 2 + 1] - dropletRadii[i];
			if (dropletPositions[i * 2 + 1] + dropletRadii[i] > max_y)
				max_y = dropletPositions[i * 2 + 1] + dropletRadii[i];
		}
		std::cout << "min_x: " << min_x << std::endl;
		std::cout << "max_x: " << max_x << std::endl;
		std::cout << "min_y: " << min_y << std::endl;
		std::cout << "max_y: " << max_y << std::endl;
		initialBoundingBox = make_double4(min_x, max_x, min_y, max_y);
		domainBoundingBox = make_double4(min_x - (max_x - min_x) * 0.5, 
			                             max_x + (max_x - min_x) * 0.5, 
			                             min_y - (max_y - min_y) * 0.5, 
			                             max_y + (max_y - min_y) * 0.5);
		boundingBoxWidth = (max_x - min_x);
		boundingBoxHeight = (max_y - min_y);
		boundingBoxCenter = make_double2((max_x + min_x) / 2.0, (max_y + min_y) / 2.0);
		boundingBoxLowerLeft = make_double2(min_x, min_y);
		std::cout << "boundingBoxWidth: " << boundingBoxWidth << std::endl;
		std::cout << "boundingBoxHeight: " << boundingBoxHeight << std::endl;
		std::cout << "boundingBoxCenter: " << boundingBoxCenter.x << " " << boundingBoxCenter.y << std::endl;
		break;
	}
	case 1: case 2: case 3:
	{
		boundingBoxWidth = 2. * rf * SpringRestLengthRatio * numDroplets_x;
		boundingBoxHeight = boundingBoxWidth;
		break;
	}
	case 6:
	{
		boundingBoxWidth = numDroplets_x * rf * 2.0;
		boundingBoxHeight = boundingBoxWidth;
		break;
	}
	}
	//boundingBoxWidth = 2. * rf * SpringRestLengthRatio * numDroplets_x * 1.5;
	//boundingBoxHeight = boundingBoxWidth;

	numCells_x = (int)ceil(boundingBoxWidth * 1.5 / cellSize);
	numCells_y = (int)ceil(boundingBoxHeight * 1.5 / cellSize);

	leftWallX = boundingBoxCenter.x - boundingBoxWidth / 3.0 - wallOffset;
	rightWallX = boundingBoxCenter.x + boundingBoxWidth / 3.0 + wallOffset;

	if (!compressionBeforeStretching)
		compressionTime = 0.0;
}

SimParameters_GPU::~SimParameters_GPU()
	{
	}

