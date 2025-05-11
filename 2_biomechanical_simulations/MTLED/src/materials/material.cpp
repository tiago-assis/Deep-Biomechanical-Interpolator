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
 *		Yu. Yue
 */

#include "boost/filesystem/fstream.hpp"
#include <iostream>

#include "ExplicitSim/materials/material.hpp"

namespace ExplicitSim {

Material::Material() : points_number_(-1)
{}

Material::~Material()
{}

void Material::SetPointsNumber(const size_t &points_number)
{
    //Set the number of points of the material.
    this->points_number_ = points_number;
}

void Material::SetDensity(const double &d_value)
{
    // Check if number of material points has been assigned.
    if (this->points_number_ == -1) {
        std::string error = "ERROR: No number of points has been assigned to the material.";
        throw std::runtime_error(error.c_str());
    }

    // Clear the density container.
    this->density_.clear();

    // Assign the given density value to all the material points.
    for (int i = 0; i != this->points_number_; ++i) {
        this->density_.push_back(d_value);
    }
}

void Material::SetYoungModulus(const double &ym_value)
{
    // Check if number of material points has been assigned.
    if (this->points_number_ == -1) {
        std::string error = "ERROR: No number of points has been assigned to the material.";
        throw std::runtime_error(error.c_str());
    }

    // Clear the Young modulus container.
    this->young_modulus_.clear();

    // Assign the given Young modulus value to all the material points.
    for (int i = 0; i != this->points_number_; ++i) {
        this->young_modulus_.push_back(ym_value);
    }
}

void Material::SetPoissonRatio(const double &pr_value)
{
    // Check if number of material points has been assigned.
    if (this->points_number_ == -1) {
        std::string error = "ERROR: No number of points has been assigned to the material.";
        throw std::runtime_error(error.c_str());
    }

    // Clear the Poisson's ratio container.
    this->poisson_ratio_.clear();

    // Assign the given Poisson's ratio value to all the material points.
    for (int i = 0; i != this->points_number_; ++i) {
        this->poisson_ratio_.push_back(pr_value);
    }
}

void Material::SetAlpha(const double &alpha_value)
{
    // Check if number of material points has been assigned.
    if (this->points_number_== -1) {
        std::string error = "ERROR: No number of points has been assigned to the material.";
        throw std::runtime_error(error.c_str());
    }

    // Clear the alpha container.
    this->alpha_.clear();

    // Assign the given alpha value to all the material points.
    for (int i = 0; i != this->points_number_; ++i) {
        this->alpha_.push_back(alpha_value);
    }
}

void Material::SetD1(const double &D1_value)
{
    // Check if number of material points has been assigned.
    if (this->points_number_== -1) {
        std::string error = "ERROR: No number of points has been assigned to the material.";
        throw std::runtime_error(error.c_str());
    }

    // Clear the alpha container.
    this->D1_.clear();

    // Assign the given alpha value to all the material points.
    for (int i = 0; i != this->points_number_; ++i) {
        this->D1_.push_back(D1_value);
    }
}

void Material::vReadFromFile(const std::string input_filename)
{
	boost::filesystem::ifstream file(input_filename);

	// Clear integration points and weights containers.
	this->density_.clear();
	this->young_modulus_.clear();
	this->poisson_ratio_.clear();

	while (1)
	{
		std::string line;
		std::getline(file, line);
		if (!file.good())
			break;

		std::stringstream iss(line);
		double density, ym, pr;
		char delim;

		try
		{
			iss >> density;
			iss >> delim;
			iss >> ym;
			iss >> delim;
			iss >> pr;
            iss >> delim;

		}
		catch (...)
		{
			file.close();
		}
		this->density_.push_back(density);
		this->young_modulus_.push_back(ym);
		this->poisson_ratio_.push_back(pr);

		this->points_number_ = this->density_.size();
	}
	file.close();

    //std::cout << "The size of density_ is " << density_.size() << std::endl;
}

//---------------------------------------------------add OReadFromFile ------------------------------------
void Material::OReadFromFile(const std::string input_filename)
{
	boost::filesystem::ifstream file(input_filename);
	this->density_.clear();
	this->mu_.clear();
	this->alpha_.clear();
	this->D1_.clear();

	while (1)
	{
		std::string line;
		std::getline(file, line);
		if (!file.good())
			break;

		std::stringstream iss(line);
		double density, mu, alpha, D1;
		char delim;

		try
		{
			iss >> density;
			iss >> delim;
			iss >> mu;
			iss >> delim;
			iss >> alpha;
			iss >> delim;
			iss >> D1;

		}
		catch (...)
		{
			file.close();
		}
		this->density_.push_back(density);
		this->mu_.push_back(mu);
		this->alpha_.push_back(alpha);
		this->D1_.push_back(D1);

		this->points_number_= this->density_.size();
	}
	file.close();
    /*std::cout << "The size of density_ is " << density_.size() << std::endl;
    std::cout << "The value of density_ [20] is " << density_[20] << std::endl;

    std::cout << "The size of mu_ is " << mu_.size() << std::endl;
    std::cout << "The value of mu_ [20] is " << mu_[20] << std::endl;

    std::cout << "The size of alpha_ is " << alpha_.size() << std::endl;
    std::cout << "The value of alpha_ [20] is " << alpha_[20] << std::endl;

    std::cout << "The size of D1_ is " << D1_.size() << std::endl;
    std::cout << "The value of D1_ is " << D1_[20] << std::endl;*/
}/*end of OReadFromFile*/

//------------------------------------------------------------------------------------------------------------------------------

//----------------------------------Add SetAlpha ---------------------------------------

//---------------------------------------------------------------------------------------------end of SetAlpha---------------------

void Material::SetBulkModulus(const double &bulk_value)
{
    // Check if number of material points has been assigned.
    if (this->points_number_ == -1) {
        std::string error = "ERROR: No number of points has been assigned to the material.";
        throw std::runtime_error(error.c_str());
    }

    // Clear the Bulk modulus container.
    this->bulk_modulus_.clear();

    // Assign the given Bulk modulus value to all the material points.
    for (int i = 0; i != this->points_number_; ++i) {
        this->bulk_modulus_.push_back(bulk_value);
    }
}

void Material::SetLameLambda(const double &l_value)
{
    // Check if number of material points has been assigned.
    if (this->points_number_ == -1) {
        std::string error = "ERROR: No number of points has been assigned to the material.";
        throw std::runtime_error(error.c_str());
    }

    // Clear the Lame lambda constant container.
    this->lambda_.clear();

    // Assign the given Lame lambda value to all the material points.
    for (int i = 0; i != this->points_number_; ++i) {
        this->lambda_.push_back(l_value);
    }
}

void Material::SetLameMu(const double &mu_value)
{
    // Check if number of material points has been assigned.
    if (this->points_number_ == -1) {
        std::string error = "ERROR: No number of points has been assigned to the material.";
        throw std::runtime_error(error.c_str());
    }

    // Clear the Lame mu (shear modulus) constant container.
    this->mu_.clear();

    // Assign the given Lame mu (shear modulus) value to all the material points.
    for (int i = 0; i != this->points_number_; ++i) {
        this->mu_.push_back(mu_value);
    }
}

void Material::SetWaveSpeed(const double &wv_speed)
{
    // Check if number of material points has been assigned.
    if (this->points_number_ == -1) {
        std::string error = "ERROR: No number of points has been assigned to the material.";
        throw std::runtime_error(error.c_str());
    }

    // Clear the wave speed constant container.
    this->wave_speed_.clear();

    // Assign the given wave speed value to all the material points.
    for (int i = 0; i != this->points_number_; ++i) {
        this->wave_speed_.push_back(wv_speed);
    }
}
void Material::ComputeBulkModulus()
{
    // Check if Young modulus and Poisson's ratio are initialized and are equal.
    /*if ((this->young_modulus_.size() == 0) &&
            (this->young_modulus_.size() != this->poisson_ratio_.size()) ) {
        std::string error = "ERROR: Young modulus and Poisson's ratio containers are not consistently initialized.";
        throw std::runtime_error(error.c_str());
    }*/

    if ((this->mu_.size() == 0) && 
        (this->alpha_.size() != this->D1_.size())) {
        std::string error = "ERROR: Shear modulus, material constant and D1 containers are not consistently initialized.";
    throw std::runtime_error(error.c_str());
    }

    // Clear Bulk modulus container.
    this->bulk_modulus_.clear();

    // Calculate the Bulk modulus.
    double bulk = 0.;
    for (std::vector<double>::size_type i = 0; i != this->mu_.size(); ++i) {
        //bulk = this->young_modulus_.at(i) / (3. * (1. - 2.*this->poisson_ratio_.at(i) ) );
        bulk = 2.0/this->D1_.at(i);

        this->bulk_modulus_.push_back(bulk);
    }
/*    std::cout << "The value of D1_ is " << D1_[20] << std::endl;
    std::cout << "The size of bulk_modulus_ is " << bulk_modulus_.size() << std::endl;
    std::cout << "The value of bulk_modulus_ is " << bulk_modulus_[20] << std::endl;
*/
}

void Material::ComputeYM()
{
    //check if shear modulus, alpha.....
    if ((this->mu_.size() == 0) && 
        (this->alpha_.size() != this->D1_.size())) {
        std::string error = "ERROR: Shear modulus, material constant and D1 containers are not consistently initialized.";
    throw std::runtime_error(error.c_str());
    }

    //Clear YM container.
    this->young_modulus_.clear();

    double ym = 0;
    for (std::vector<double>::size_type i = 0; i != this->mu_.size(); ++i) {
        ym = (9. * this->bulk_modulus_.at(i) * this->mu_.at(i))/(3. * this->bulk_modulus_.at(i) + this->mu_.at(i));

        this->young_modulus_.push_back(ym);
    }
/*    std::cout << "The size of young_modulus_ is " << young_modulus_.size() << std::endl;
    std::cout << "The value of young_modulus_ is " << young_modulus_[20] << std::endl;*/
}

void Material::ComputePR()
{
    //check
    if ((this->mu_.size() == 0) && 
        (this->alpha_.size() != this->D1_.size())) {
        std::string error = "ERROR: Shear modulus, material constant and D1 containers are not consistently initialized.";
    throw std::runtime_error(error.c_str());
    }
    //clear pr
    this->poisson_ratio_.clear();

    double pr = 0;
    for (std::vector<double>::size_type i = 0; i != this->mu_.size(); ++i) {
        pr = (3. * bulk_modulus_.at(i) - 2.*this->mu_.at(i))/(2.*(3.*this->bulk_modulus_.at(i) + this->mu_.at(i)));

        this->poisson_ratio_.push_back(pr);
    }
    /*std::cout << "The size of poisson_ratio_ is " << poisson_ratio_.size() << std::endl;
    std::cout << "The value of poisson_ratio_ is " << poisson_ratio_[20] << std::endl;*/

}

void Material::ComputeLameLambdaMu()
{
    // Check if Young modulus and Poisson's ratio are initialized and are equal.
/*    if ((this->young_modulus_.size() == 0) &&
            (this->young_modulus_.size() != this->poisson_ratio_.size()) ) {
        std::string error = "ERROR: Young modulus and Poisson's ratio containers are not consistently initialized.";
        throw std::runtime_error(error.c_str());
    }*/
    if ((this->mu_.size() == 0) && 
        (this->alpha_.size() != this->D1_.size())) {
        std::string error = "ERROR: Shear modulus, material constant and D1 containers are not consistently initialized.";
    throw std::runtime_error(error.c_str());
    }

    // Clear Lame constants containers.
    this->lambda_.clear();
    //this->mu_.clear();

    // Calculate the lame constants.
    double lame_l = 0.; //double lame_m = 0.;
    for (std::vector<double>::size_type i = 0; i != this->mu_.size(); ++i) {
        // Lame lambda parameter.
        lame_l = (this->young_modulus_.at(i) * this->poisson_ratio_.at(i)) /
                ( (1. + this->poisson_ratio_.at(i)) * (1. - 2.*this->poisson_ratio_.at(i)) );

        this->lambda_.push_back(lame_l);

        // Lame mu parameter.
        //lame_m = this->young_modulus_.at(i) / (2. * (1. + this->poisson_ratio_.at(i)));

        //this->mu_.push_back(lame_m);
    }
    /*std::cout << "The size of lambda_ is " << lambda_.size() << std::endl;
    std::cout << "The value of lambda_ is " << lambda_[20] << std::endl;*/
}



void Material::ComputeWaveSpeed()
{
    // Check if Lame constants and density are initialized and are equal.
    /*if ((this->lambda_.size() == 0) ||
            (this->lambda_.size() != this->mu_.size()) ||
            (this->lambda_.size() != this->density_.size())) {
        std::string error = "[ExplicitSim ERROR] Lame constants (lambda, mu) and density containers are not consistently initialized.";
        throw std::runtime_error(error.c_str());
    }*/
    if ((this->mu_.size() == 0) && 
        (this->alpha_.size() != this->D1_.size())) {
        std::string error = "ERROR: Shear modulus, material constant and D1 containers are not consistently initialized.";
    throw std::runtime_error(error.c_str());
    }    

    // Clear wave speed container.
    this->wave_speed_.clear();

    // Calculate the wave speed.
    double speed = 0.;
    for (std::vector<double>::size_type i = 0; i != this->mu_.size(); ++i) {
        speed = std::sqrt((this->lambda_.at(i) + 2.*this->mu_.at(i)) / this->density_.at(i) );

        this->wave_speed_.push_back(speed);
    }
   /* std::cout << "The wave speed [20] is " << wave_speed_[20] << std::endl;
    std::cout << "The size of wave speed is " << wave_speed_.size() << std::endl;*/
}

} //end of namespace ExplicitSim
