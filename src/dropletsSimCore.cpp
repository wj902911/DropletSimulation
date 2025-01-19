#include "dropletsSimCore.h"
#include "SimParams.h"
#include <Eigen/Geometry>
#include <Eigen/SparseQR>
#include <algorithm>
#include <unordered_map>
#include <iostream>

using namespace Eigen;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



DropletsSimCore::DropletsSimCore()
{
	params_ = std::make_shared<SimParameters>();
}

DropletsSimCore::~DropletsSimCore()
{
}

void DropletsSimCore::initSimulation()
{
    //std::cout << "initSimulation" << std::endl;
    droplet_unique_id_ = 0;
    time_ = 0;
    droplets_.clear();
    connectors_.clear();

    int numDroplets = params_->initialDropletPositions.rows();
    for (int i = 0; i < numDroplets; i++)
	{
        double x = params_->initialDropletPositions(i, 0);
        double y = params_->initialDropletPositions(i, 1);
        double sub = params_->initialDropletSubstance(i);
        if (i==0)
			addDroplet(x, y, sub, true);
		else
            addDroplet(x, y, sub);
	}
}

bool DropletsSimCore::simulateOneStep()
{
    VectorXd q, qprev, v, r, C;
    std::tie(q, qprev, v, r, C) = buildConfiguration();
    numericalIntegration(q, qprev, v, r, C);
    unbuildConfiguration(q, v, r);
    removeSnappedSpring();
    if (params_->addConnectorWhenTouching)
		tuchDetection();
    time_ += params_->timeStep;

    return true;
}

void DropletsSimCore::tuchDetection()
{
    int numDroplets = droplets_.size();
	for (int i = 0; i < numDroplets; i++)
	{
		for (int j = i + 1; j < numDroplets; j++)
		{
            std::pair<int, int> dropletsPair(i, j);
            if (connected_droplets_.find(dropletsPair) != connected_droplets_.end())
				continue;
			Vector2d pos1 = droplets_[i]->pos;
			Vector2d pos2 = droplets_[j]->pos;
			double r1 = droplets_[i]->radius;
			double r2 = droplets_[j]->radius;
			double dist = (pos1 - pos2).norm();
            if (dist > (r1 + r2))
                continue;
            else
                addConnector(i, j);
		}
	}
}

void DropletsSimCore::getCurrentDropletPositionAndRadius(Eigen::MatrixX2d& pos, Eigen::VectorXd& r) const
{
    int numDroplets = droplets_.size();
	pos.resize(numDroplets, 2);
	r.resize(numDroplets);
	for (int i = 0; i < numDroplets; i++)
	{
		pos(i, 0) = droplets_[i]->pos[0];
		pos(i, 1) = droplets_[i]->pos[1];
        r[i] = droplets_[i]->radius;
	}
}

uint32_t DropletsSimCore::addDroplet(double x, double y, double sub, bool isFixed)
{
    Vector2d newpos(x, y);
    double mass = params_->dropletMass;
    if (isFixed)
        mass = std::numeric_limits<double>::infinity();

    int newid = droplets_.size();
    auto ret = droplet_unique_id_;
    droplet_offset_[droplet_unique_id_] = (int)droplets_.size();
    std::shared_ptr newDroplet = std::make_shared<Droplet>(newpos, 
                                                           params_->initialDropletRadius, 
                                                           mass, 
                                                           sub, 
                                                           isFixed, 
                                                           false, 
                                                           ret);
    droplets_.emplace_back(newDroplet);
    droplet_unique_id_++;

    int numparticles = droplets_.size() - 1;
    for (int i = 0; i < numparticles; i++)
    {
        if (connected_droplets_.find(std::make_pair(newid, i)) != connected_droplets_.end())
            continue;
        if (droplets_[i]->inert)
            continue;
        Vector2d pos = droplets_[i]->pos;
        double rNew = newDroplet->radius;
        double r= droplets_[i]->radius;
        double dist = (pos - newpos).norm();
        if (dist > (rNew + r))
            continue;
        connectors_.emplace_back(std::make_shared<Spring>(newid, i, 0, params_->springStiffness, params_->SpringRestLengthRatio * (rNew + r), true));
        connected_droplets_.insert(std::make_pair(newid, i));
        std::cout << "add connector between droplet_" << newid << " and droplet_" << i << std::endl;
    }
    return ret;
}

void DropletsSimCore::addConnector(int p1, int p2)
{
	connectors_.emplace_back(std::make_shared<Spring>(
        p1, 
        p2, 
        0, 
        params_->springStiffness, 
        params_->SpringRestLengthRatio * (droplets_[p1]->radius + droplets_[p2]->radius), 
        false));
    connected_droplets_.insert(std::make_pair(p1, p2));
    std::cout << "add connector between droplet_" << p1 << " and droplet_" << p2 << std::endl;
}

std::shared_ptr<Droplet>
DropletsSimCore::queryDroplet(uint32_t uid)
{
    for (const auto& p : droplets_)
        if (p->uid == uid)
            return p;
    return std::shared_ptr<Droplet>(nullptr);
}

int DropletsSimCore::mapDropletToConfigurationIndex(uint32_t uid) const
{
    // TODO
    auto iter = droplet_offset_.find(uid);
    //int p = iter->second;
    return (iter == droplet_offset_.end()) ? -1 : iter->second;
}

uint32_t DropletsSimCore::mapConfigurationIndexToDroplet(int i) const
{
    // TODO
    for (auto iter = droplet_offset_.begin(); iter != droplet_offset_.end(); iter++)
        if (iter->second == i)
            return iter->first;
    return 0;
}

std::shared_ptr<const Connector>
DropletsSimCore::queryConnector(uint32_t uid1, uint32_t uid2) const
{
    auto p1 = mapDropletToConfigurationIndex(uid1);
    auto p2 = mapDropletToConfigurationIndex(uid2);
    for (const auto& pc : connectors_) {
        const Connector& c = *pc;
        if (c.p1 == p1 && c.p2 == p2) {
            return pc;
        }
        else if (c.p2 == p1 && c.p1 == p2) {
            return pc;
        }
    }
    return std::shared_ptr<const Connector>(nullptr);
}

double DropletsSimCore::getTotalDropletMass(int idx) const
{
    double mass = droplets_[idx]->mass;
    for (const auto& c : connectors_) {
        if (c->p1 == idx || c->p2 == idx)
            mass += 0.5 * c->mass;
    }
    return mass;
}

void DropletsSimCore::removeSnappedSpring()
{
    for (int i = 0; i < (int)connectors_.size(); i++)
	{
		if (connectors_[i]->isSnapped())
		{
			int p1 = connectors_[i]->p1;
			int p2 = connectors_[i]->p2;
            connected_droplets_.erase(connected_droplets_.find(std::make_pair(p1, p2)));
			connectors_.erase(connectors_.begin() + i);
			i--;
            std::cout << "remove connector between droplet_" << p1 << " and droplet_" << p2 << std::endl;
		}
	}
}


std::tuple<Eigen::VectorXd, // current position (q)
           Eigen::VectorXd, // previous position (qprev)
           Eigen::VectorXd, // velocity (v)
           Eigen::VectorXd, // radius (r)
           Eigen::VectorXd> // osmolarity (C)
DropletsSimCore::buildConfiguration() const
{
    Eigen::VectorXd q, qprev, v, r, C;
    int ndofs = 2 * droplets_.size();
    q.resize(ndofs);
    v.resize(ndofs);
    qprev.resize(ndofs);
    r.resize(droplets_.size());
    C.resize(droplets_.size());

    for (int i = 0; i < (int)droplets_.size(); i++)
    {
		q.segment<2>(2 * i) = droplets_[i]->pos;
		qprev.segment<2>(2 * i) = droplets_[i]->prevpos;
		v.segment<2>(2 * i) = droplets_[i]->vel;
        r[i] = droplets_[i]->radius;
        C[i] = droplets_[i]->sub / (4. / 3. * M_PI * r[i] * r[i] * r[i]);
	}
    return std::make_tuple(q, qprev, v, r, C);
}

void DropletsSimCore::unbuildConfiguration(const Eigen::Ref<Eigen::VectorXd> q,
                                           const Eigen::Ref<Eigen::VectorXd> v,
                                           const Eigen::Ref<Eigen::VectorXd> r)
{
    int ndofs = q.size();
    //assert(ndofs == int(2 * droplets_.size()));

    for (int i = 0; i < ndofs / 2; i++) {
        droplets_[i]->prevpos = droplets_[i]->pos;
        droplets_[i]->pos = q.segment<2>(2 * i);
        droplets_[i]->vel = v.segment<2>(2 * i);
        droplets_[i]->radius = r[i];
    }
}

void DropletsSimCore::numericalIntegration(VectorXd& q, VectorXd& qprev, VectorXd& v, VectorXd& r, VectorXd& C)
{

    VectorXd F = createZeroForce();
    VectorXd J = createZeroFlow();
    SparseMatrix<double> Minv;

    computeMassInverse(Minv);

    qprev = q;
    q += params_->timeStep * v;
#if 0
    if (params_->flowEnabled)
    {
        processFlow(q, r, C, J);
        r.array() = pow(((4. / 3. * M_PI * r.array() * r.array() * r.array() + params_->timeStep * J.array()) / (4. / 3. * M_PI)).array(), 1. / 3.);
    }
    computeForceWithoutHessian(q, qprev, r, F);
#else
    computeForceAndFlow(q, qprev, r, C, F, J);

    //std::cout << F << std::endl << std::endl;
    //std::cout << J << std::endl << std::endl;

    r.array() = pow(((4. / 3. * M_PI * r.array() * r.array() * r.array() + params_->timeStep * J.array()) / (4. / 3. * M_PI)).array(), 1. / 3.);
    //std::cout << r << std::endl << std::endl;
#endif
    v += params_->timeStep * Minv * F;
}

TripletArray DropletsSimCore::computeMassInverseCoeff() const
{
    int ndofs = 2 * int(droplets_.size());
    std::vector<Eigen::Triplet<double>> Minvcoeffs;
    for (int i = 0; i < ndofs / 2; i++) {
        Minvcoeffs.emplace_back(2 * i, 2 * i, 1.0 / getTotalDropletMass(i));
        Minvcoeffs.emplace_back(2 * i + 1, 2 * i + 1, 1.0 / getTotalDropletMass(i));
    }
    return Minvcoeffs;
}


VectorXd DropletsSimCore::createZeroForce() const
{
    VectorXd f;
    int ndofs = 2 * droplets_.size();
    f.setZero(ndofs);
    return f;
}

VectorXd DropletsSimCore::createZeroFlow() const
{
    VectorXd J;
    int ndofs = droplets_.size();
    J.setZero(ndofs);
    return J;
}

void DropletsSimCore::computeMassInverse(SparseMatrix<double>& Minv)
{
    auto Minvcoeffs = computeMassInverseCoeff();
    int ndofs = 2 * int(droplets_.size());
    Minv.resize(ndofs, ndofs);
    Minv.setFromTriplets(Minvcoeffs.begin(), Minvcoeffs.end());
}

void DropletsSimCore::computeForceWithoutHessian(const VectorXd& q,
                                                 const VectorXd& qprev,
                                                 const VectorXd& r,
                                                       VectorXd& F) const
{
    F = createZeroForce();

    if (params_->gravityEnabled)
        processGravityForce(F);
    if (params_->springsEnabled) {
        processSpringForce(q, r, F);
    }
    if (params_->dampingEnabled) {
        processDampingForce(q, qprev, F);
    }
}

void DropletsSimCore::computeForceAndFlow(const Eigen::VectorXd& q, 
                                          const Eigen::VectorXd& qprev, 
                                          const Eigen::VectorXd& r, 
                                          const Eigen::VectorXd& C, 
                                          Eigen::VectorXd& F, 
                                          Eigen::Ref<Eigen::VectorXd> J) const
{
    if (params_->gravityEnabled)
        processGravityForce(F);
    if (params_->springsEnabled)
        processAll(q, qprev, r, C, F, J);
}

void DropletsSimCore::processGravityForce(Ref<VectorXd> F) const
{
    int nparticles = (int)droplets_.size();
    for (int i = 0; i < nparticles; i++) {
        if (!droplets_[i]->fixed) {
            F[2 * i + 1] += params_->gravityG * getTotalDropletMass(i);
        }
    }
}

void DropletsSimCore::processSpringForce(const VectorXd& q,
                                         const VectorXd& r,
                                         Ref<VectorXd> F) const
{
    for (const auto& pc : connectors_) {
        pc->processSpringForce(q, r, *params_, F);
    }
}

void DropletsSimCore::processFlow(const VectorXd& q,
                                  const VectorXd& r,
                                  const VectorXd& C,
                                  Ref<VectorXd> J) const
{
    for (const auto& pc : connectors_) {
		pc->processFlow(q, r, C, *params_, J);
	}
}

void DropletsSimCore::processDampingForce(const VectorXd& q,
    const VectorXd& qprev,
    Ref<VectorXd> F) const
{
    for (const auto& pc : connectors_) {
        pc->processDampingForce(q, qprev, *params_, F);
    }
}

void DropletsSimCore::processAll(const Eigen::VectorXd& q, 
                                 const Eigen::VectorXd& qprev, 
                                 const Eigen::VectorXd& r, 
                                 const Eigen::VectorXd& C, 
                                 Eigen::Ref<Eigen::VectorXd> F, 
                                 Eigen::Ref<Eigen::VectorXd> J) const
{
    for (const auto& pc : connectors_)
    {
		pc->processSpringAll(q, qprev, r, C, *params_, F, J);
    }
}
