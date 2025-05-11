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
 */


#include "ExplicitSim/models/weak_model_3d.hpp"

namespace ExplicitSim {

WeakModel3D::WeakModel3D() : is_mass_scaled_(false), min_mass_scale_factor_(0.), max_mass_scale_factor_(0.)
{}


WeakModel3D::~WeakModel3D()
{}


void WeakModel3D::LoadMeshRepresentation(const std::string &mesh_filename)
{
    // Load the mesh.
    this->tetramesh_.LoadFrom(mesh_filename.c_str());
}


void WeakModel3D::CreateGridRepresentation()
{
    // Check if mesh nodes are available.
    if (this->tetramesh_.Nodes().size() == 0) {
        std::string error = "ERROR: Model's mesh is not initialized. Load mesh representation before creating the grid.";
        throw std::runtime_error(error.c_str());
    }

    //Copy mesh nodes to the grid.
    this->grid_.EditNodes().clear();
    this->grid_.EditNodes() = this->tetramesh_.Nodes();
}


void WeakModel3D::CreateIntegrationPoints(const IntegOptions &options, const SupportDomain &support_dom)
{
    // Create the model's integration points.
    this->integ_points_.Generate(this->tetramesh_, options, support_dom);
}


void WeakModel3D::ComputeMass(const std::vector<double> &density, const std::vector<double> &time_steps, const double &max_time_step,
                              const std::vector< std::vector<uint32_t> > &support_nodes_ids, bool scaling)
{
    // Check if integration points weights are available.
    if (this->integ_points_.Weights().size() == 0) {
        std::string error = "[ExplicitSim ERROR] Cannot compute 3d weak model's mass. Integration points' weights are not initialized.";
        throw std::runtime_error(error.c_str());
    }

    // Check if the given containers are size-consistent.
    if ((density.size() != time_steps.size()) ||
        (density.size() != support_nodes_ids.size()) ||
        (density.size() != this->integ_points_.Weights().size())) {
        throw std::invalid_argument(Logger::Error("Cannot compute 3d weak model's mass. The given variables "
                                                  "are not consistent in size with the integration points' weights container.").c_str());
    }
	
    // Initialize the distributed mass to the model's grid points.
    this->mass_.clear();
    this->mass_.assign(this->TetrahedralMesh().NodesNum(), 0.);

    // Initialize the distributed mass 3d matrix representaion.
    this->mass_matrix_ = Eigen::MatrixXd::Zero(this->TetrahedralMesh().NodesNum(), 3);

    // Compute the distributed mass according to scale option.
	this->is_mass_scaled_ = scaling;
    if (scaling) 
	{
		// Initialize mass scaling factor.
        double scale_factor = 0.; double max_scale_factor = 0.;

        // Iterate over model's integration points' weights.
        for (auto &weight : this->integ_points_.Weights()) {
            // Get the ith integration point index.
            auto i = &weight - &this->integ_points_.Weights()[0];

            // Set scaling factor to (maximum time step/integration point's time step)^2.
            scale_factor = max_time_step / time_steps[i];
            scale_factor *= scale_factor;

            // Set the maximum mass scaling factor for information output.
            if (scale_factor > max_scale_factor) { max_scale_factor = scale_factor; }

			auto num_nodes = support_nodes_ids[i].size();

            // Iterate over the support domain nodes of the ith integration point.
            for (auto &support_id : support_nodes_ids[i]) {
                // Compute mass with scaling.
                this->mass_[support_id] += density[i] * scale_factor * weight / num_nodes;
            }
        } // End iteration over model's integration points.

        // Output the maximum mass scaling factor.
        std::cout << "[ExplicitSim] Maximum mass scaling factor: " << max_scale_factor << std::endl;
    }
    else 
	{
        // Iterate over model's integration points' weights.
        for (auto &weight : this->integ_points_.Weights()) {
            // Get the ith integration point index.
            auto i = &weight - &this->integ_points_.Weights()[0];
			auto num_nodes = support_nodes_ids[i].size();
            // Iterate over the support domain nodes of the ith integration point.
            for (auto &support_id : support_nodes_ids[i]) {
                // Compute mass with no scaling.
                this->mass_[support_id] += density[i] * weight / num_nodes;
            }
        } // End iteration over model's integration points.

    }

    // Generate the distributed mass 3d matrix representaion.
    for (auto &mass : this->mass_) {
        auto id = &mass - &this->mass_[0];
        this->mass_matrix_.coeffRef(id, 0) = mass;
        this->mass_matrix_.coeffRef(id, 1) = mass;
        this->mass_matrix_.coeffRef(id, 2) = mass;
    }

}


} //end of namespace ExplicitSim
