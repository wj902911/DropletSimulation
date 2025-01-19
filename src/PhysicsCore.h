#pragma once

#include <tuple>
#include <Eigen/Core>

class PhysicsCore
{
public:
    PhysicsCore();
    virtual ~PhysicsCore();

    virtual void initSimulation() = 0;
    virtual bool simulateOneStep() = 0;
};