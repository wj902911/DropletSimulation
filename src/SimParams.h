#pragma once

#include <eigen/Core>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct SimParameters
{
	SimParameters()
	{
#if 0
		initialDropletPositions.resize(2, 2);
		initialDropletPositions <<
			0.0, 0.0,
			1.0, 0.0;
		initialDropletSubstance.resize(2);
		initialDropletSubstance <<
			1.0, 0.1;
#else
#if 0
		initialDropletPositions.resize(18, 2);
		initialDropletPositions << 
			0.0, 0.0,
			1.0, 0.0,
			2.0, 0.0,
			3.0, 0.0,
			4.0, 0.0,
			-1.0, 0.0,
			-2.0, 0.0,
			-3.0, 0.0,
			-4.0, 0.0,
			0.0, 1.0,
			1.0, 1.0,
			2.0, 1.0,
			3.0, 1.0,
			4.0, 1.0,
			-1.0, 1.0,
			-2.0, 1.0,
			-3.0, 1.0,
			-4.0, 1.0;
		initialDropletSubstance.resize(18);
		initialDropletSubstance << 
			1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
			0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
#else
		int numDroplets = 38;
		double spacing = 0.8;
		initialDropletPositions.resize(numDroplets, 2);
		initialDropletSubstance.resize(numDroplets);
		double x = 0;
		double y = 0;
		initialDropletPositions.row(0) << x, y;
		initialDropletSubstance(0) = 1.0;
		for (int i = 1; i < (numDroplets / 2 - 1) / 2 + 1; i++)
		{
			x += spacing;
			initialDropletPositions.row(i) << x, y;
			initialDropletSubstance(i) = 1.0;
		}
		x = 0;
		for (int i = (numDroplets / 2 - 1) / 2 + 1; i < numDroplets / 2; i++)
		{
			x -= spacing;
			initialDropletPositions.row(i) << x, y;
			initialDropletSubstance(i) = 1.0;
		}
		x = spacing / 2.0;
		y = spacing * cosf(M_PI / 6);
		initialDropletPositions.row(numDroplets / 2) << x, y;
		initialDropletSubstance(numDroplets / 2) = 0.1;
		for (int i = numDroplets / 2 + 1; i < (numDroplets / 2 - 1) / 2 + numDroplets / 2 + 1; i++)
		{
			x += spacing;
			initialDropletPositions.row(i) << x, y;
			initialDropletSubstance(i) = 0.1;
		}
		x = spacing / 2.0;
		for (int i = (numDroplets / 2 - 1) / 2 + numDroplets / 2 + 1; i < numDroplets; i++)
		{
			x -= spacing;
			initialDropletPositions.row(i) << x, y;
			initialDropletSubstance(i) = 0.1;
		}
		//std::cout << initialDropletPositions << std::endl << std::endl;
		//std::cout << initialDropletSubstance << std::endl << std::endl;
#endif
#endif
	}

	enum ConnectorType
	{
		CT_SPRING
	};

	Eigen::MatrixX2d initialDropletPositions;
	Eigen::VectorXd initialDropletSubstance;
	double timeStep = 0.001;
	double dropletMass = 0.2;
	double springStiffness = 50.0;
	double dampingStiffness = 1.1;
	double maxSpringDistRatio = 1.1;
	double SpringRestLengthRatio = 0.8;
	bool gravityEnabled = false;
	double gravityG = -9.8;
	bool springsEnabled = true;
	bool dampingEnabled = true;
	double permeability = 0.05;
	bool flowEnabled = true;
	double initialDropletRadius = 0.5;
	bool addConnectorWhenTouching = true;
};