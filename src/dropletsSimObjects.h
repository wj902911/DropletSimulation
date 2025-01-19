#pragma once

#include <SimParams.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>

using TripletArray = std::vector<Eigen::Triplet<double>>;


struct Droplet
{
public:
	Droplet(Eigen::Vector2d pos, double radius, double mass, double sub, bool isFixed, bool isInert, uint32_t uid)
		: pos(pos), radius(radius), mass(mass), sub(sub), fixed(isFixed), inert(isInert), uid(uid)
	{
		vel.setZero();
		prevpos = pos;
	}

	Eigen::Vector2d pos;
	Eigen::Vector2d prevpos;
	Eigen::Vector2d vel;
	double mass;
	bool fixed;
	bool inert;
	double radius;
	double sub;

	uint32_t uid = -1;
};

struct Connector
{
public:
	Connector(int p1, int p2, double mass)
		: p1(p1)
		, p2(p2)
		, mass(mass)
	{
	}
	virtual ~Connector() {}

	virtual SimParameters::ConnectorType getType() const = 0;

	int p1;
	int p2;
	double mass;

	virtual void processSpringAll(const Eigen::VectorXd& q,
								  const Eigen::VectorXd& qprev,
		                          const Eigen::VectorXd& r,
								  const Eigen::VectorXd& C,
		                          const SimParameters& params,
								  Eigen::Ref<Eigen::VectorXd> F,
		                          Eigen::Ref<Eigen::VectorXd> J) {}

	virtual void processSpringForce(const Eigen::VectorXd& q,
		                            const Eigen::VectorXd& r,
		                            const SimParameters& params,
									Eigen::Ref<Eigen::VectorXd> F) const {}

	virtual void processDampingForce(const Eigen::VectorXd& q,
									 const Eigen::VectorXd& qprev,
									 const SimParameters& params,
									 Eigen::Ref<Eigen::VectorXd> F) const {}

	virtual void processFlow(const Eigen::VectorXd& q,
					         const Eigen::VectorXd& r,
							 const Eigen::VectorXd& C,
							 const SimParameters& params,
							 Eigen::Ref<Eigen::VectorXd> J) const {}

	virtual bool isSnapped() const { return false; }
};

struct Spring : public Connector
{
public:
	Spring(int p1, int p2, double mass, double stiffness, double restlen, bool canSnap)
		: Connector(p1, p2, mass)
		, stiffness(stiffness)
		, restlen(restlen)
		, canSnap(canSnap)
	{
	}

	virtual SimParameters::ConnectorType getType() const override
	{
		return SimParameters::CT_SPRING;
	}

	double stiffness;
	double restlen;
	bool canSnap;
	bool snapped = false;

	void processSpringAll(const Eigen::VectorXd& q,
						  const Eigen::VectorXd& qprev,
		                  const Eigen::VectorXd& r,
						  const Eigen::VectorXd& C,
		                  const SimParameters& params,
						  Eigen::Ref<Eigen::VectorXd> F,
		                  Eigen::Ref<Eigen::VectorXd> J) override;

	void processSpringForce(const Eigen::VectorXd& q,
		                    const Eigen::VectorXd& r,
		                    const SimParameters& params,
							Eigen::Ref<Eigen::VectorXd> F) const override;
	void processDampingForce(const Eigen::VectorXd& q,
							 const Eigen::VectorXd& qprev,
							 const SimParameters& params,
							 Eigen::Ref<Eigen::VectorXd> F) const override;

	void processFlow(const Eigen::VectorXd& q,
					 const Eigen::VectorXd& r,
					 const Eigen::VectorXd& C,
					 const SimParameters& params,
		             Eigen::Ref<Eigen::VectorXd> J) const override;

	bool isSnapped() const override { return snapped; }
};