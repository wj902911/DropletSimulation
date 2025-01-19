#pragma once

#include <PhysicsCore.h>
#include <dropletsSimObjects.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/StdVector>

#include <memory>
#include <stdint.h>
#include <tuple>
#include <unordered_map>
#include <set>

struct PairComparator {
	bool operator()(const std::pair<int, int>& p1, const std::pair<int, int>& p2) const {
		return std::minmax(p1.first, p1.second) < std::minmax(p2.first, p2.second);
	}
};

class DropletsSimCore : public PhysicsCore
{
public:
	DropletsSimCore();
	~DropletsSimCore();

	virtual void initSimulation() override;
	virtual bool simulateOneStep() override;

	void tuchDetection();

	std::shared_ptr<SimParameters> getPointerToSimParameters()
	{
		return params_;
	}

	SimParameters getSimParameters() const
	{
		return *params_;
	}

	void setSimParameters(const SimParameters& nsimp)
	{
		*params_ = nsimp;
	}

	double getTime() const
	{
		return time_;
	}

	void getCurrentDropletPositionAndRadius(Eigen::MatrixX2d& pos, Eigen::VectorXd& r) const;

	uint32_t addDroplet(double x, double y, double sub, bool isFixed = false);

	void addConnector(int p1, int p2);

	std::shared_ptr<Droplet>
	queryDroplet(uint32_t uid);

	std::shared_ptr<const Connector>
	queryConnector(uint32_t uid1, uint32_t uid2) const;

	auto listAllDroplets() const { return droplets_; }

	int mapDropletToConfigurationIndex(uint32_t) const;
	uint32_t mapConfigurationIndexToDroplet(int) const;

	//std::shared_ptr<const Spring>
	//querySpring(uint32_t uid1, uint32_t uid2) const;

	std::tuple<Eigen::VectorXd, // current position (q)
			   Eigen::VectorXd, // previous position (qprev)
			   Eigen::VectorXd, // velocity (v)
			   Eigen::VectorXd, // radius (r)
			   Eigen::VectorXd> // osmolarity (C)
	buildConfiguration() const;

	void unbuildConfiguration(const Eigen::Ref<Eigen::VectorXd> q,
							  const Eigen::Ref<Eigen::VectorXd> v,
							  const Eigen::Ref<Eigen::VectorXd> r);


	TripletArray computeMassInverseCoeff() const;

	Eigen::VectorXd createZeroForce() const;
	Eigen::VectorXd createZeroFlow() const;

	void processGravityForce(Eigen::Ref<Eigen::VectorXd> F) const;

	void processSpringForce(const Eigen::VectorXd& q,
		                    const Eigen::VectorXd& r,
							Eigen::Ref<Eigen::VectorXd> F) const;

	void processFlow(const Eigen::VectorXd& q,
					 const Eigen::VectorXd& r,
					 const Eigen::VectorXd& C,
						   Eigen::Ref<Eigen::VectorXd> J) const;

	void processDampingForce(const Eigen::VectorXd& q,
							 const Eigen::VectorXd& qprev,
							 Eigen::Ref<Eigen::VectorXd> F) const;

	void processAll(const Eigen::VectorXd& q,
					const Eigen::VectorXd& qprev,
					const Eigen::VectorXd& r,
					const Eigen::VectorXd& C,
						  Eigen::Ref<Eigen::VectorXd> F,
						  Eigen::Ref<Eigen::VectorXd> J) const;

private:
	void computeMassInverse(Eigen::SparseMatrix<double>& Minv);

	void numericalIntegration(Eigen::VectorXd& q,
                              Eigen::VectorXd& qprev,
							  Eigen::VectorXd& v,
							  Eigen::VectorXd& r,
                              Eigen::VectorXd& C);

	void computeForceWithoutHessian(const Eigen::VectorXd& q,
                                    const Eigen::VectorXd& qprev,
		                            const Eigen::VectorXd& r,
                                    Eigen::VectorXd& F) const;

	void computeForceAndFlow(const Eigen::VectorXd& q,
                             const Eigen::VectorXd& qprev,
		                     const Eigen::VectorXd& r,
		                     const Eigen::VectorXd& C,
                             Eigen::VectorXd& F,
		                     Eigen::Ref<Eigen::VectorXd> J) const;


	double getTotalDropletMass(int idx) const;

	void removeSnappedSpring();


	uint32_t droplet_unique_id_;
	std::shared_ptr<SimParameters> params_;
	double time_ = 0.0;
	std::vector<std::shared_ptr<Droplet>> droplets_;
	std::vector<std::shared_ptr<Connector>> connectors_;
	std::unordered_map<uint32_t, int> droplet_offset_;
	std::set<std::pair<int, int>, PairComparator> connected_droplets_;
};