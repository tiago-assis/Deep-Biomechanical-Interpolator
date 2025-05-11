/*
 * ExplicitSim - Software for solving PDEs using explicit methods.
 * Copyright (C) 2017  <Konstantinos A. Mountris> <konstantinos.mountris@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors (alphabetically):
 *      George C. BOURANTAS
 *      Grand R. JOLDES
 *      Konstantinos A. MOUNTRIS
 *      Benjamin F. ZWICK
 */

#include <Eigen/Eigenvalues>
#include <iostream>

#include "ExplicitSim/materials/ogden.hpp"

namespace ExplicitSim {

Ogden::Ogden()
{}

Ogden::~Ogden()
{}

/*void Ogden::SetAlpha(const double &alpha_value)
{
	// Check if number of material points has been assigned.
	if (this->points_number_ == -1) {
		std::string error = "ERROR: No number of points has been assigned to the material.";
		throw std::runtime_error(error.c_str());
	}

	// Clear the alpha container.
	this->alpha_.clear();

	// Assign the given alpha value to all the material points.
	for (int i = 0; i != this->points_number_; ++i) {
		this->alpha_.push_back(alpha_value);
	}
}*/

Eigen::Matrix3d Ogden::SpkStress(const Eigen::Matrix3d &FT, const size_t &integ_point_id) const
{
	// Right Cauchy-Green deformation tensor
	Eigen::Matrix3d C = Eigen::Matrix3d::Zero(3, 3);
	C.noalias() = FT * FT.transpose();
	//std::cout << "size of mu is " << mu_.size() <<std::endl;
    //std::cout << "The size of alpha_ is " << alpha_.size() << std::endl;
   // std::cout << "The value of alpha_ [20] is " << alpha_[20] << std::endl;
	//std::cout << "size of bulk_modulus_ is " << bulk_modulus_.size() <<std::endl;
	// Ogden model parameters
	double alpha = alpha_[integ_point_id];
	double mu = mu_[integ_point_id];
	double kappa = bulk_modulus_[integ_point_id];

	// Eigenvalues and eigenvectors
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
	es.computeDirect(C);
	Eigen::Vector3d Eigs = es.eigenvalues();
	Eigen::Matrix3d Dirs = es.eigenvectors();

	// Principal stretches
	double lamda1 = sqrt(Eigs(0));
	double lamda2 = sqrt(Eigs(1));
	double lamda3 = sqrt(Eigs(2));

	// Strain energy function derivatives
	double J = lamda1 * lamda2 * lamda3;
	double Slambda = std::pow(lamda1, alpha) + std::pow(lamda2, alpha) + std::pow(lamda3, alpha);
	double b = 2.0 * mu / alpha * std::pow(J, (-alpha / 3.0));
	double a = kappa * J * (J - 1.0) - b/3.0 * Slambda;

	double dUdl1 = a / lamda1 + b * std::pow(lamda1, (alpha - 1.0));
	double dUdl2 = a / lamda2 + b * std::pow(lamda2, (alpha - 1.0));
	double dUdl3 = a / lamda3 + b * std::pow(lamda3, (alpha - 1.0));

	// Second Piola-Kirchhoff stress tensor
	Eigen::Vector3d S_principal;
	S_principal << dUdl1/lamda1, dUdl2/lamda2, dUdl3/lamda3;
	Eigen::Matrix3d S;
	S.noalias() = Dirs * S_principal.asDiagonal() * Dirs.inverse();

	return S;
}

} //end of namespace ExplicitSim
