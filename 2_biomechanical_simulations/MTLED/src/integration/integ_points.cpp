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

#include "boost/filesystem/operations.hpp" // includes boost/filesystem/path.hpp
#include "boost/filesystem/fstream.hpp"    // ditto
#include <iostream>                        // for std::cout
#include <sstream>
#include <iomanip>

#include "ExplicitSim/integration/integ_points.hpp"

namespace ExplicitSim {

IntegPoints::IntegPoints()
{}


IntegPoints::~IntegPoints()
{}


void IntegPoints::Generate(const TetraMesh &tetramesh, const IntegOptions &options,
                           const SupportDomain &support_dom)
{
    // Clear integration points and weights containers.
    this->coordinates_.clear();
    this->weights_.clear();

    // Generate integration points.
    if (options.is_adaptive_) {
        // Use adaptive generation of integration points.
        this->GenerateAdaptivePointsPerTetra(tetramesh, options, support_dom);
    }
    else {
        // Choose generation schedule for standard generation of integration points.
        switch (options.integ_points_per_tetra_) {
            case 1 :
                this->coordinates_.reserve(tetramesh.Elements().size());
                this->weights_.reserve(tetramesh.Elements().size());
                this->GenerateOnePointPerTetra(tetramesh);
                break;
            case 4 :
                this->coordinates_.reserve(4*tetramesh.Elements().size());
                this->weights_.reserve(4*tetramesh.Elements().size());
                this->GenerateFourPointsPerTetra(tetramesh);
                break;
            case 5:
                this->coordinates_.reserve(5*tetramesh.Elements().size());
                this->weights_.reserve(5*tetramesh.Elements().size());
                this->GenerateFivePointsPerTetra(tetramesh);
                break;
            default :
                std::string error = "[ExplicitSim ERROR] Invalid number of integration points requested. Admissible numbers: [1/4/5]";
                throw std::invalid_argument(error.c_str());
                break;
        }
    } // End Generate integration points.

}

void IntegPoints::vReadFromFile(const std::string input_filename)
{
	boost::filesystem::ifstream file(input_filename);
	
	// Clear integration points and weights containers.
	this->coordinates_.clear();
	this->weights_.clear();

	while (1)
	{
		std::string line;
		std::getline(file, line);
		if (!file.good())
			break;

		std::stringstream iss(line);
		Vec3<double> coords;
		double weight;
		char delim;

		try
		{
			iss >> coords[0];
			iss >> delim;
			iss >> coords[1];
			iss >> delim;
			iss >> coords[2];
			iss >> delim;
			iss >> weight;
		}
		catch (...)
		{
			file.close();
		}
		this->coordinates_.emplace_back(coords);
		this->weights_.emplace_back(weight);
	}
	
	file.close();
}

void IntegPoints::vSaveToFile(const std::string output_filename) const
{
	boost::filesystem::ofstream file(output_filename);
	file << std::setprecision(16);
	// Iterate over all the integration points.
	auto size = Coordinates().size();
	for (auto index = 0; index < size; index++) {
		auto coord = this->coordinates_[index];
		auto weight = this->weights_[index];
		file << coord[0] << ", " << coord[1] << ", " << coord[2] << ", " << weight << "\n";
	}
	file.close();
}

void IntegPoints::GenerateOnePointPerTetra(const TetraMesh &tetramesh)
{
    // Generate one integration point for each tetrahedron of the mesh.
    for (auto &elem : tetramesh.Elements()) {

        // Set the integration point at the element's centroid.
        this->coordinates_.emplace_back((tetramesh.Nodes()[elem.N1()].Coordinates() +
                                         tetramesh.Nodes()[elem.N2()].Coordinates() +
                                         tetramesh.Nodes()[elem.N3()].Coordinates() +
                                         tetramesh.Nodes()[elem.N4()].Coordinates()) / 4.);

        // Store the volume of the element as the integration point's weight.
        this->weights_.emplace_back(elem.Volume(tetramesh.Nodes()));
    }

}


void IntegPoints::GenerateFourPointsPerTetra(const TetraMesh &tetramesh)
{
    // Parameters for 4-point integration.
    double alpha = 0.5854102; double beta = 0.1381966;

    // Initialize the weight of an integration point.
    double weight = 0.;

    // Generate four integration points for each tetrahedron of the mesh.
    for (auto &elem : tetramesh.Elements()) {

        // The weight per integration point as the quarter of the element's volume.
        weight = elem.Volume(tetramesh.Nodes()) / 4.;

        // Store the 1st integration point coordinates.
        this->coordinates_.emplace_back(alpha*tetramesh.Nodes()[elem.N1()].Coordinates() +
                                        beta*(tetramesh.Nodes()[elem.N2()].Coordinates() +
                                              tetramesh.Nodes()[elem.N3()].Coordinates() +
                                              tetramesh.Nodes()[elem.N4()].Coordinates()) );

        // Store the 2nd integration point's coordinates.
        this->coordinates_.emplace_back(alpha*tetramesh.Nodes()[elem.N2()].Coordinates() +
                                        beta*(tetramesh.Nodes()[elem.N1()].Coordinates() +
                                              tetramesh.Nodes()[elem.N3()].Coordinates() +
                                              tetramesh.Nodes()[elem.N4()].Coordinates()) );

        // Store the 3rd integration point's coordinates.
        this->coordinates_.emplace_back(alpha*tetramesh.Nodes()[elem.N3()].Coordinates() +
                                        beta*(tetramesh.Nodes()[elem.N1()].Coordinates() +
                                             tetramesh.Nodes()[elem.N2()].Coordinates() +
                                             tetramesh.Nodes()[elem.N4()].Coordinates()) );

        // Store the 4th integration point's coordinates.
        this->coordinates_.emplace_back(alpha*tetramesh.Nodes()[elem.N4()].Coordinates() +
                                        beta*(tetramesh.Nodes()[elem.N1()].Coordinates() +
                                              tetramesh.Nodes()[elem.N2()].Coordinates() +
                                              tetramesh.Nodes()[elem.N3()].Coordinates()) );

        // Store the weight for each of the 4 integration points.
        this->weights_.insert(this->weights_.end(), {weight, weight, weight, weight});

    }

}


void IntegPoints::GenerateFourPointsPerTetra(const std::vector<Node> &tet_nodes,
                                             std::vector<Vec3<double> > &ip_coords, std::vector<double> &ip_weights)
{
    // Check if the given element is a tetrahedron.
    if (tet_nodes.size() != 4) {
        std::string error = "[ExplicitSim ERROR] Cannot generate integration points. The given element is not a tetrahedron.";
        throw std::invalid_argument(error.c_str());
    }

    // Initialize integration points' coordinates and weights containers.
    ip_coords.clear(); ip_coords.reserve(4);
    ip_weights.clear(); ip_weights.reserve(4);

    // Parameters for 4-point integration.
    double alpha = 0.5854102; double beta = 0.1381966;

    // Set the tetrahedron.
    Tetrahedron tet;
    tet.SetConnectivity(0, 1, 2, 3);

    // The weight per integration point as the quarter of the element's volume.
    double weight = tet.Volume(tet_nodes) / 4.;

    // Store the 1st integration point coordinates.
    ip_coords.emplace_back(alpha*tet_nodes[tet.N1()].Coordinates() +
                           beta*(tet_nodes[tet.N2()].Coordinates() +
                                 tet_nodes[tet.N3()].Coordinates() +
                                 tet_nodes[tet.N4()].Coordinates()) );

    // Store the 2nd integration point's coordinates.
    ip_coords.emplace_back(alpha*tet_nodes[tet.N2()].Coordinates() +
                           beta*(tet_nodes[tet.N1()].Coordinates() +
                                 tet_nodes[tet.N3()].Coordinates() +
                                 tet_nodes[tet.N4()].Coordinates()) );

    // Store the 3rd integration point's coordinates.
    ip_coords.emplace_back(alpha*tet_nodes[tet.N3()].Coordinates() +
                           beta*(tet_nodes[tet.N1()].Coordinates() +
                                 tet_nodes[tet.N2()].Coordinates() +
                                 tet_nodes[tet.N4()].Coordinates()) );

    // Store the 4th integration point's coordinates.
    ip_coords.emplace_back(alpha*tet_nodes[tet.N4()].Coordinates() +
                           beta*(tet_nodes[tet.N1()].Coordinates() +
                                 tet_nodes[tet.N2()].Coordinates() +
                                 tet_nodes[tet.N3()].Coordinates()) );

    // Store the weight for each of the 4 integration points.
    ip_weights.insert(ip_weights.end(), {weight, weight, weight, weight});

}


void IntegPoints::GenerateFivePointsPerTetra(const TetraMesh &tetramesh)
{
    // Initialize the weight of the centroid (1st) integration point and the rest 4 integration points.
    double weight_cent = 0.; double weight = 0.;

    // Generate four integration points for each tetrahedron of the mesh.
    for (const auto &elem : tetramesh.Elements()) {

        // The weight of the centroid (1st) integration point.
        weight_cent = elem.Volume(tetramesh.Nodes()) * (-4. / 5.);

        // The weight of the rest 4 integration points.
        weight = elem.Volume(tetramesh.Nodes()) * (9. / 20.);

        // Store the 1st integration point's coordinates.
        this->coordinates_.emplace_back((tetramesh.Nodes()[elem.N1()].Coordinates() +
                                         tetramesh.Nodes()[elem.N2()].Coordinates() +
                                         tetramesh.Nodes()[elem.N3()].Coordinates() +
                                         tetramesh.Nodes()[elem.N4()].Coordinates()) / 4.);

        // Store the 2nd integration point's coordinates.
        this->coordinates_.emplace_back((tetramesh.Nodes()[elem.N1()].Coordinates() / 2.) +
                                        (tetramesh.Nodes()[elem.N2()].Coordinates() +
                                         tetramesh.Nodes()[elem.N3()].Coordinates() +
                                         tetramesh.Nodes()[elem.N4()].Coordinates()) / 6.);

        // Store the 3rd integration point coordinates.
        this->coordinates_.emplace_back((tetramesh.Nodes()[elem.N2()].Coordinates() / 2.) +
                                        (tetramesh.Nodes()[elem.N1()].Coordinates() +
                                         tetramesh.Nodes()[elem.N3()].Coordinates() +
                                         tetramesh.Nodes()[elem.N4()].Coordinates()) / 6.);

        // Store the 4th integration point coordinates.
        this->coordinates_.emplace_back((tetramesh.Nodes()[elem.N3()].Coordinates() / 2.) +
                                        (tetramesh.Nodes()[elem.N1()].Coordinates() +
                                         tetramesh.Nodes()[elem.N2()].Coordinates() +
                                         tetramesh.Nodes()[elem.N4()].Coordinates()) / 6.);

        // Store the 5th integration point coordinates.
        this->coordinates_.emplace_back((tetramesh.Nodes()[elem.N4()].Coordinates() / 2.) +
                                        (tetramesh.Nodes()[elem.N1()].Coordinates() +
                                         tetramesh.Nodes()[elem.N2()].Coordinates() +
                                         tetramesh.Nodes()[elem.N3()].Coordinates()) / 6.);

        // Store the weight for the centroid and the rest 4 integration points.
        this->weights_.insert(this->weights_.end(), {weight_cent, weight, weight, weight, weight});
    }

}


void IntegPoints::GenerateAdaptivePointsPerTetra(const TetraMesh &tetramesh, const IntegOptions &options,
                                                 const SupportDomain &support_dom)
{
    //Allocate memory for integration points coordinates and weights.
    size_t ip_num = 0;
    switch (options.tetra_divisions_) {
        case 2 :
            // 2 x 4 ip per element.
            ip_num = 8*tetramesh.Elements().size();
        case 4 :
            // 4 x 4 ip per element.
            ip_num = 16*tetramesh.Elements().size();
        case 8 :
            // 8 x 4 ip per element.
            ip_num = 32*tetramesh.Elements().size();
        default:
            ip_num = 0;
    }
    this->coordinates_.reserve(ip_num);
    this->weights_.reserve(ip_num);

    // Iterate over all the elements of the mesh.
    for (const auto &elem : tetramesh.Elements()) {

        // Extract the nodes of the element.
        std::vector<Node> elem_nodes(4, Node());
        elem_nodes[0] = tetramesh.Nodes()[elem.N1()];
        elem_nodes[1] = tetramesh.Nodes()[elem.N2()];
        elem_nodes[2] = tetramesh.Nodes()[elem.N3()];
        elem_nodes[3] = tetramesh.Nodes()[elem.N4()];

        // Set the adaptive integration level.
        int adaptive_level = options.adaptive_level_;

        // Declare the element's integration points contribution.
        std::vector<Vec3<double> > elem_ip_coords;
        std::vector<double> elem_ip_weights;

        // Choose generation schedule from integration options.
        switch (options.tetra_divisions_) {
            case 2 :
                this->AdaptiveTwoTetraDivisions(tetramesh, elem_nodes, support_dom, options.adaptive_eps_, adaptive_level,
                                                elem_ip_coords, elem_ip_weights);

                // Add integration points contribution from the element.
                this->coordinates_.insert(this->coordinates_.end(), elem_ip_coords.begin(), elem_ip_coords.end());
                this->weights_.insert(this->weights_.end(), elem_ip_weights.begin(), elem_ip_weights.end());

                break;
            case 4 :
                this->AdaptiveFourTetraDivisions(tetramesh, elem_nodes, support_dom, options.adaptive_eps_, adaptive_level,
                                                 elem_ip_coords, elem_ip_weights);

                // Add integration points contribution from the element.
                this->coordinates_.insert(this->coordinates_.end(), elem_ip_coords.begin(), elem_ip_coords.end());
                this->weights_.insert(this->weights_.end(), elem_ip_weights.begin(), elem_ip_weights.end());

                break;
            case 8 :
                this->AdaptiveEightTetraDivisions(tetramesh, elem_nodes, support_dom, options.adaptive_eps_,
                                                  elem_ip_coords, elem_ip_weights);

                // Add integration points contribution from the element.
                this->coordinates_.insert(this->coordinates_.end(), elem_ip_coords.begin(), elem_ip_coords.end());
                this->weights_.insert(this->weights_.end(), elem_ip_weights.begin(), elem_ip_weights.end());

                break;
            default :
                std::string error = "[ExplicitSim ERROR] Invalid number of tetrahedral divisions requested for"
                                    " adaptive integration. Admissible numbers: [2/4/8]";
                throw std::invalid_argument(error.c_str());
                break;
        }

    } // End Iterate over all the elements of the mesh.

}


void IntegPoints::AdaptiveTwoTetraDivisions(const TetraMesh &tetramesh, const std::vector<Node> &tet_nodes,
                                            const SupportDomain &support_dom, const double &adaptive_eps, int &adaptive_level,
                                            std::vector<Vec3<double> > &tet_ip_coords, std::vector<double> &tet_ip_weights)
{
    // Generate four integration points for this element.
    this->GenerateFourPointsPerTetra(tet_nodes, tet_ip_coords, tet_ip_weights);

    // Calculate the element's integration value vector.
    Vec3<double> tet_integ_val = this->IntegrationValue(tetramesh, tet_ip_coords, tet_ip_weights, support_dom);

    // Reduce the adaptation level.
    adaptive_level--;

    if (adaptive_level <= 0) {
        std::cout << "[ExplicitSim WARNING] The maximum adaptation level of integration points has been reached.\n";
        return;
    }

    // Compute the squares of the tetrahedron edges.
    double n12_sq = tet_nodes[0].Coordinates().Distance2(tet_nodes[1].Coordinates());
    double n23_sq = tet_nodes[1].Coordinates().Distance2(tet_nodes[2].Coordinates());
    double n31_sq = tet_nodes[2].Coordinates().Distance2(tet_nodes[0].Coordinates());
    double n41_sq = tet_nodes[3].Coordinates().Distance2(tet_nodes[0].Coordinates());
    double n42_sq = tet_nodes[3].Coordinates().Distance2(tet_nodes[1].Coordinates());
    double n43_sq = tet_nodes[3].Coordinates().Distance2(tet_nodes[2].Coordinates());

    // Collect the tetrahedron's edges.
    std::vector<double> edges_sq { n12_sq, n23_sq, n31_sq, n41_sq, n42_sq, n43_sq };

    // Get the maximum edge index in storage.
    std::vector<double>::iterator max_edge_sq_it = std::max_element(edges_sq.begin(), edges_sq.end());
    size_t max_edge_id = std::distance(edges_sq.begin(), max_edge_sq_it);

    // Initialize 2 tetrahedral subdivisions.
    std::vector<Node> stet1_nodes(4, Node()), stet2_nodes(4, Node());

    // Apply 2 tetrahedral subdivision by adding a new node
    // at the midpoint of the longest edge of the tetrahedron.
    if (max_edge_id == 0) {     //edge n12
        // Set the extra node's (n5) coordinates at n12 edge center.
        Vec3<double> n5_coords((tet_nodes[0].Coordinates() + tet_nodes[1].Coordinates()) / 2.);

        // Set the first sub-element (n1, n5, n3, n4).
        stet1_nodes[0] = tet_nodes[0]; stet1_nodes[1].SetCoordinates(n5_coords);
        stet1_nodes[2] = tet_nodes[2]; stet1_nodes[3] = tet_nodes[3];

        // Set the second sub-element (n2, n3, n5, n4).
        stet2_nodes[0] = tet_nodes[1]; stet2_nodes[1] = tet_nodes[2];
        stet2_nodes[2].SetCoordinates(n5_coords); stet2_nodes[3] = tet_nodes[3];
    }
    else if (max_edge_id == 1) {    //edge n23
        // Set the extra node's (n5) coordinates at n23 edge center.
        Vec3<double> n5_coords((tet_nodes[1].Coordinates() + tet_nodes[2].Coordinates()) / 2.);

        // Set the first sub-element (n2, n5, n1, n4).
        stet1_nodes[0] = tet_nodes[1]; stet1_nodes[1].SetCoordinates(n5_coords);
        stet1_nodes[2] = tet_nodes[0]; stet1_nodes[3] = tet_nodes[3];

        // Set the second sub-element (n3, n1, n5, n4).
        stet2_nodes[0] = tet_nodes[2]; stet2_nodes[1] = tet_nodes[0];
        stet2_nodes[2].SetCoordinates(n5_coords); stet2_nodes[3] = tet_nodes[3];
    }
    else if (max_edge_id == 2) {    //edge n31
        // Set the extra node's (n5) coordinates at n31 edge center.
        Vec3<double> n5_coords((tet_nodes[2].Coordinates() + tet_nodes[0].Coordinates()) / 2.);

        // Set the first sub-element (n1, n5, n2, n4).
        stet1_nodes[0] = tet_nodes[0]; stet1_nodes[1].SetCoordinates(n5_coords);
        stet1_nodes[2] = tet_nodes[1]; stet1_nodes[3] = tet_nodes[3];

        // Set the second sub-element (n3, n2, n5, n4).
        stet2_nodes[0] = tet_nodes[2]; stet2_nodes[1] = tet_nodes[1];
        stet2_nodes[2].SetCoordinates(n5_coords); stet2_nodes[3] = tet_nodes[3];
    }
    else if (max_edge_id == 3) {    //edge n41
        // Set the extra node's (n5) coordinates at n41 edge center.
        Vec3<double> n5_coords((tet_nodes[3].Coordinates() + tet_nodes[0].Coordinates()) / 2.);

        // Set the first sub-element (n1, n5, n2, n3).
        stet1_nodes[0] = tet_nodes[0]; stet1_nodes[1].SetCoordinates(n5_coords);
        stet1_nodes[2] = tet_nodes[1]; stet1_nodes[3] = tet_nodes[2];

        // Set the second sub-element (n4, n3, n5, n2).
        stet2_nodes[0] = tet_nodes[3]; stet2_nodes[1] = tet_nodes[2];
        stet2_nodes[2].SetCoordinates(n5_coords); stet2_nodes[3] = tet_nodes[1];
    }
    else if (max_edge_id == 4) {    //edge n42
        // Set the extra node's (n5) coordinates at n42 edge center.
        Vec3<double> n5_coords((tet_nodes[3].Coordinates() + tet_nodes[1].Coordinates()) / 2.);

        // Set the first sub-element (n2, n5, n1, n3).
        stet1_nodes[0] = tet_nodes[1]; stet1_nodes[1].SetCoordinates(n5_coords);
        stet1_nodes[2] = tet_nodes[0]; stet1_nodes[3] = tet_nodes[2];

        // Set the second sub-element (n4, n3, n5, n1).
        stet2_nodes[0] = tet_nodes[3]; stet2_nodes[1] = tet_nodes[2];
        stet2_nodes[2].SetCoordinates(n5_coords); stet2_nodes[3] = tet_nodes[0];
    }
    else if (max_edge_id == 5) {    //edge n43
        // Set the extra node's (n5) coordinates at n43 edge center.
        Vec3<double> n5_coords((tet_nodes[3].Coordinates() + tet_nodes[2].Coordinates()) / 2.);

        // Set the first sub-element (n3, n5, n1, n2).
        stet1_nodes[0] = tet_nodes[2]; stet1_nodes[1].SetCoordinates(n5_coords);
        stet1_nodes[2] = tet_nodes[0]; stet1_nodes[3] = tet_nodes[1];

        // Set the second sub-element (n4, n1, n5, n2).
        stet2_nodes[0] = tet_nodes[3]; stet2_nodes[1] = tet_nodes[0];
        stet2_nodes[2].SetCoordinates(n5_coords); stet2_nodes[3] = tet_nodes[1];
    }

    // Initialize sub-elements integration points coordinates and weights.
    std::vector<Vec3<double> > stet1_ip_coords, stet2_ip_coords;
    std::vector<double> stet1_ip_weights, stet2_ip_weights;

    // Generate four integration points for the sub-elements.
    this->GenerateFourPointsPerTetra(stet1_nodes, stet1_ip_coords, stet1_ip_weights);
    this->GenerateFourPointsPerTetra(stet2_nodes, stet2_ip_coords, stet2_ip_weights);


    // Calculate the integration value vector for the 2 sub-elements.
    Vec3<double> stet1_integ_val = this->IntegrationValue(tetramesh, stet1_ip_coords, stet1_ip_weights, support_dom);
    Vec3<double> stet2_integ_val = this->IntegrationValue(tetramesh, stet2_ip_coords, stet2_ip_weights, support_dom);

    // Calculate absolute approximation error.
    Vec3<double> error(1., 1., 1.);
    error -= (stet1_integ_val + stet2_integ_val).CoeffWiseDiv(tet_integ_val);
    error = error.CoeffWiseAbs();

    // Recursive adaptation.
    if (error.MaxCoeff() > adaptive_eps) {

        this->AdaptiveTwoTetraDivisions(tetramesh, stet1_nodes, support_dom, adaptive_eps,
                                        adaptive_level, stet1_ip_coords, stet1_ip_weights);

        this->AdaptiveTwoTetraDivisions(tetramesh, stet2_nodes, support_dom, adaptive_eps,
                                        adaptive_level, stet2_ip_coords, stet2_ip_weights);

        // Add integration points contribution from 1st sub-element.
        tet_ip_coords.reserve(stet1_ip_coords.size() + stet2_ip_coords.size());
        tet_ip_coords = stet1_ip_coords;
        tet_ip_coords.insert(tet_ip_coords.end(), stet2_ip_coords.begin(), stet2_ip_coords.end());

        // Add integration points contribution from 2nd sub-element.
        tet_ip_weights.reserve(stet1_ip_weights.size() + stet2_ip_weights.size());
        tet_ip_weights = stet1_ip_weights;
        tet_ip_weights.insert(tet_ip_weights.end(), stet2_ip_weights.begin(), stet2_ip_weights.end());
    }

}


void IntegPoints::AdaptiveFourTetraDivisions(const TetraMesh &tetramesh, const std::vector<Node> &tet_nodes,
                                             const SupportDomain &support_dom, const double &adaptive_eps, int &adaptive_level,
                                             std::vector<Vec3<double> > &tet_ip_coords, std::vector<double> &tet_ip_weights)
{
    // Generate four integration points for this element.
    this->GenerateFourPointsPerTetra(tet_nodes, tet_ip_coords, tet_ip_weights);

    // Calculate the element's integration value vector.
    Vec3<double> tet_integ_val = this->IntegrationValue(tetramesh, tet_ip_coords, tet_ip_weights, support_dom);

    // Reduce the adaptation level.
    adaptive_level--;

    if (adaptive_level <= 0) {
        std::cout << "[ExplicitSim WARNING] The maximum adaptation level of integration points has been reached.\n";
        return;
    }

    // Apply 4 tetrahedral subdivision by adding a new node
    // at the midpoint at each edge of one of the tetrahedron's face.

    // Set the extra node's (n5) coordinates at n14 edge center.
    Vec3<double> n5_coords((tet_nodes[0].Coordinates() + tet_nodes[3].Coordinates()) / 2.);

    // Set the extra node's (n6) coordinates at n43 edge center.
    Vec3<double> n6_coords((tet_nodes[3].Coordinates() + tet_nodes[2].Coordinates()) / 2.);

    // Set the extra node's (n7) coordinates at n31 edge center.
    Vec3<double> n7_coords((tet_nodes[2].Coordinates() + tet_nodes[0].Coordinates()) / 2.);

    // Set the 1st tetrahedral subdivision (n1, n5, n7, n2).
    std::vector<Node> stet1_nodes(4, Node());
    stet1_nodes[0] = tet_nodes[0]; stet1_nodes[1].SetCoordinates(n5_coords);
    stet1_nodes[2].SetCoordinates(n7_coords); stet1_nodes[3] = tet_nodes[1];

    // Set the 2nd tetrahedral subdivision (n5, n6, n7, n2).
    std::vector<Node> stet2_nodes(4, Node());
    stet2_nodes[0].SetCoordinates(n5_coords); stet2_nodes[1].SetCoordinates(n6_coords);
    stet2_nodes[2].SetCoordinates(n7_coords); stet2_nodes[3] = tet_nodes[1];

    // Set the 3rd tetrahedral subdivision (n3, n7, n6, n2).
    std::vector<Node> stet3_nodes(4, Node());
    stet3_nodes[0] = tet_nodes[2]; stet3_nodes[1].SetCoordinates(n7_coords);
    stet3_nodes[2].SetCoordinates(n6_coords); stet3_nodes[3] = tet_nodes[1];

    // Set the 4th tetrahedral subdivision (n5, n4, n6, n2).
    std::vector<Node> stet4_nodes(4, Node());
    stet4_nodes[0].SetCoordinates(n5_coords); stet4_nodes[1] = tet_nodes[3];
    stet4_nodes[2].SetCoordinates(n6_coords); stet4_nodes[3] = tet_nodes[1];

    // Initialize sub-elements integration points coordinates and weights.
    std::vector<Vec3<double> > stet1_ip_coords, stet2_ip_coords, stet3_ip_coords, stet4_ip_coords;
    std::vector<double> stet1_ip_weights, stet2_ip_weights, stet3_ip_weights, stet4_ip_weights;

    // Generate four integration points for the sub-elements.
    this->GenerateFourPointsPerTetra(stet1_nodes, stet1_ip_coords, stet1_ip_weights);
    this->GenerateFourPointsPerTetra(stet2_nodes, stet2_ip_coords, stet2_ip_weights);
    this->GenerateFourPointsPerTetra(stet3_nodes, stet3_ip_coords, stet3_ip_weights);
    this->GenerateFourPointsPerTetra(stet4_nodes, stet4_ip_coords, stet4_ip_weights);

    // Calculate the integration value vector for the 4 sub-elements.
    Vec3<double> stet1_integ_val = this->IntegrationValue(tetramesh, stet1_ip_coords, stet1_ip_weights, support_dom);
    Vec3<double> stet2_integ_val = this->IntegrationValue(tetramesh, stet2_ip_coords, stet2_ip_weights, support_dom);
    Vec3<double> stet3_integ_val = this->IntegrationValue(tetramesh, stet3_ip_coords, stet3_ip_weights, support_dom);
    Vec3<double> stet4_integ_val = this->IntegrationValue(tetramesh, stet4_ip_coords, stet4_ip_weights, support_dom);

    // Calculate absolute approximation error.
    Vec3<double> error(1., 1., 1.);
    error -= (stet1_integ_val + stet2_integ_val + stet3_integ_val + stet4_integ_val).CoeffWiseDiv(tet_integ_val);
    error = error.CoeffWiseAbs();

    // Recursive adaptation.
    if (error.MaxCoeff() > adaptive_eps) {

        this->AdaptiveFourTetraDivisions(tetramesh, stet1_nodes, support_dom, adaptive_eps,
                                        adaptive_level, stet1_ip_coords, stet1_ip_weights);

        this->AdaptiveFourTetraDivisions(tetramesh, stet2_nodes, support_dom, adaptive_eps,
                                        adaptive_level, stet2_ip_coords, stet2_ip_weights);

        this->AdaptiveFourTetraDivisions(tetramesh, stet3_nodes, support_dom, adaptive_eps,
                                        adaptive_level, stet3_ip_coords, stet3_ip_weights);

        this->AdaptiveFourTetraDivisions(tetramesh, stet4_nodes, support_dom, adaptive_eps,
                                        adaptive_level, stet4_ip_coords, stet4_ip_weights);

        // Add integration points contribution from 1st sub-element.
        tet_ip_coords.reserve(stet1_ip_coords.size() + stet2_ip_coords.size() + stet3_ip_coords.size() + stet4_ip_coords.size());
        tet_ip_coords = stet1_ip_coords;
        tet_ip_coords.insert(tet_ip_coords.end(), stet2_ip_coords.begin(), stet2_ip_coords.end());
        tet_ip_coords.insert(tet_ip_coords.end(), stet3_ip_coords.begin(), stet3_ip_coords.end());
        tet_ip_coords.insert(tet_ip_coords.end(), stet4_ip_coords.begin(), stet4_ip_coords.end());

        // Add integration points contribution from 2nd sub-element.
        tet_ip_weights.reserve(stet1_ip_weights.size() + stet2_ip_weights.size() + stet3_ip_weights.size() + stet4_ip_weights.size());
        tet_ip_weights = stet1_ip_weights;
        tet_ip_weights.insert(tet_ip_weights.end(), stet2_ip_weights.begin(), stet2_ip_weights.end());
        tet_ip_weights.insert(tet_ip_weights.end(), stet3_ip_weights.begin(), stet3_ip_weights.end());
        tet_ip_weights.insert(tet_ip_weights.end(), stet4_ip_weights.begin(), stet4_ip_weights.end());
    }

}


void IntegPoints::AdaptiveEightTetraDivisions(const TetraMesh &tetramesh, const std::vector<Node> &tet_nodes,
                                              const SupportDomain &support_dom, const double &adaptive_eps,
                                              std::vector<Vec3<double> > &tet_ip_coords, std::vector<double> &tet_ip_weights)
{
    // Generate four integration points for this element.
    this->GenerateFourPointsPerTetra(tet_nodes, tet_ip_coords, tet_ip_weights);

    // Calculate the element's integration value vector.
    Vec3<double> tet_integ_val = this->IntegrationValue(tetramesh, tet_ip_coords, tet_ip_weights, support_dom);

    // Apply 8 tetrahedral subdivision by adding a new node
    // at the midpoint at each edge of the tetrahedron.

    // Set the extra node's (n5) coordinates at n12 edge center.
    Vec3<double> n5_coords((tet_nodes[0].Coordinates() + tet_nodes[1].Coordinates()) / 2.);

    // Set the extra node's (n6) coordinates at n24 edge center.
    Vec3<double> n6_coords((tet_nodes[1].Coordinates() + tet_nodes[3].Coordinates()) / 2.);

    // Set the extra node's (n7) coordinates at n23 edge center.
    Vec3<double> n7_coords((tet_nodes[1].Coordinates() + tet_nodes[2].Coordinates()) / 2.);

    // Set the extra node's (n7) coordinates at n14 edge center.
    Vec3<double> n8_coords((tet_nodes[0].Coordinates() + tet_nodes[3].Coordinates()) / 2.);

    // Set the extra node's (n7) coordinates at n34 edge center.
    Vec3<double> n9_coords((tet_nodes[2].Coordinates() + tet_nodes[3].Coordinates()) / 2.);

    // Set the extra node's (n7) coordinates at n13 edge center.
    Vec3<double> n10_coords((tet_nodes[0].Coordinates() + tet_nodes[2].Coordinates()) / 2.);

    // Set the 1st tetrahedral subdivision (n1, n5, n10, n8).
    std::vector<Node> stet1_nodes(4, Node());
    stet1_nodes[0] = tet_nodes[0]; stet1_nodes[1].SetCoordinates(n5_coords);
    stet1_nodes[2].SetCoordinates(n10_coords); stet1_nodes[3].SetCoordinates(n8_coords);

    // Set the 2nd tetrahedral subdivision (n2, n5, n6, n7).
    std::vector<Node> stet2_nodes(4, Node());
    stet2_nodes[0] = tet_nodes[1]; stet2_nodes[1].SetCoordinates(n5_coords);
    stet2_nodes[2].SetCoordinates(n6_coords); stet2_nodes[3].SetCoordinates(n7_coords);

    // Set the 3rd tetrahedral subdivision (n3, n7, n9, n10).
    std::vector<Node> stet3_nodes(4, Node());
    stet3_nodes[0] = tet_nodes[2]; stet3_nodes[1].SetCoordinates(n7_coords);
    stet3_nodes[2].SetCoordinates(n9_coords); stet3_nodes[3].SetCoordinates(n10_coords);

    // Set the 4th tetrahedral subdivision (n4, n6, n8, n9).
    std::vector<Node> stet4_nodes(4, Node());
    stet4_nodes[0] = tet_nodes[3]; stet4_nodes[1].SetCoordinates(n6_coords);
    stet4_nodes[2].SetCoordinates(n8_coords); stet4_nodes[3].SetCoordinates(n9_coords);

    // Set the 5th tetrahedral subdivision (n5, n6, n9, n8).
    std::vector<Node> stet5_nodes(4, Node());
    stet5_nodes[0].SetCoordinates(n5_coords); stet5_nodes[1].SetCoordinates(n6_coords);
    stet5_nodes[2].SetCoordinates(n9_coords); stet5_nodes[3].SetCoordinates(n8_coords);

    // Set the 6th tetrahedral subdivision (n6, n5, n9, n7).
    std::vector<Node> stet6_nodes(4, Node());
    stet6_nodes[0].SetCoordinates(n6_coords); stet6_nodes[1].SetCoordinates(n5_coords);
    stet6_nodes[2].SetCoordinates(n9_coords); stet6_nodes[3].SetCoordinates(n7_coords);

    // Set the 7th tetrahedral subdivision (n7, n9, n10, n5).
    std::vector<Node> stet7_nodes(4, Node());
    stet7_nodes[0].SetCoordinates(n7_coords); stet7_nodes[1].SetCoordinates(n9_coords);
    stet7_nodes[2].SetCoordinates(n10_coords); stet7_nodes[3].SetCoordinates(n5_coords);

    // Set the 8th tetrahedral subdivision (n8, n10, n9, n5).
    std::vector<Node> stet8_nodes(4, Node());
    stet8_nodes[0].SetCoordinates(n8_coords); stet8_nodes[1].SetCoordinates(n10_coords);
    stet8_nodes[2].SetCoordinates(n9_coords); stet8_nodes[3].SetCoordinates(n5_coords);

    // Initialize sub-elements integration points coordinates and weights.
    std::vector<Vec3<double> > stet1_ip_coords, stet2_ip_coords, stet3_ip_coords, stet4_ip_coords;
    std::vector<Vec3<double> > stet5_ip_coords, stet6_ip_coords, stet7_ip_coords, stet8_ip_coords;
    std::vector<double> stet1_ip_weights, stet2_ip_weights, stet3_ip_weights, stet4_ip_weights;
    std::vector<double> stet5_ip_weights, stet6_ip_weights, stet7_ip_weights, stet8_ip_weights;

    // Generate four integration points for the sub-elements.
    this->GenerateFourPointsPerTetra(stet1_nodes, stet1_ip_coords, stet1_ip_weights);
    this->GenerateFourPointsPerTetra(stet2_nodes, stet2_ip_coords, stet2_ip_weights);
    this->GenerateFourPointsPerTetra(stet3_nodes, stet3_ip_coords, stet3_ip_weights);
    this->GenerateFourPointsPerTetra(stet4_nodes, stet4_ip_coords, stet4_ip_weights);
    this->GenerateFourPointsPerTetra(stet5_nodes, stet5_ip_coords, stet5_ip_weights);
    this->GenerateFourPointsPerTetra(stet6_nodes, stet6_ip_coords, stet6_ip_weights);
    this->GenerateFourPointsPerTetra(stet7_nodes, stet7_ip_coords, stet7_ip_weights);
    this->GenerateFourPointsPerTetra(stet8_nodes, stet8_ip_coords, stet8_ip_weights);

    // Calculate the integration value vector for the 4 sub-elements.
    Vec3<double> stet1_integ_val = this->IntegrationValue(tetramesh, stet1_ip_coords, stet1_ip_weights, support_dom);
    Vec3<double> stet2_integ_val = this->IntegrationValue(tetramesh, stet2_ip_coords, stet2_ip_weights, support_dom);
    Vec3<double> stet3_integ_val = this->IntegrationValue(tetramesh, stet3_ip_coords, stet3_ip_weights, support_dom);
    Vec3<double> stet4_integ_val = this->IntegrationValue(tetramesh, stet4_ip_coords, stet4_ip_weights, support_dom);
    Vec3<double> stet5_integ_val = this->IntegrationValue(tetramesh, stet5_ip_coords, stet5_ip_weights, support_dom);
    Vec3<double> stet6_integ_val = this->IntegrationValue(tetramesh, stet6_ip_coords, stet6_ip_weights, support_dom);
    Vec3<double> stet7_integ_val = this->IntegrationValue(tetramesh, stet7_ip_coords, stet7_ip_weights, support_dom);
    Vec3<double> stet8_integ_val = this->IntegrationValue(tetramesh, stet8_ip_coords, stet8_ip_weights, support_dom);

    // Calculate absolute approximation error.
    Vec3<double> error(1., 1., 1.);
    error -= (stet1_integ_val + stet2_integ_val + stet3_integ_val + stet4_integ_val +
              stet5_integ_val + stet6_integ_val + stet7_integ_val + stet8_integ_val).CoeffWiseDiv(tet_integ_val);
    error = error.CoeffWiseAbs();

    // Recursive adaptation.
    if (error.MaxCoeff() > adaptive_eps) {

        this->AdaptiveEightTetraDivisions(tetramesh, stet1_nodes, support_dom, adaptive_eps,
                                          stet1_ip_coords, stet1_ip_weights);

        this->AdaptiveEightTetraDivisions(tetramesh, stet2_nodes, support_dom, adaptive_eps,
                                          stet2_ip_coords, stet2_ip_weights);

        this->AdaptiveEightTetraDivisions(tetramesh, stet3_nodes, support_dom, adaptive_eps,
                                          stet3_ip_coords, stet3_ip_weights);

        this->AdaptiveEightTetraDivisions(tetramesh, stet4_nodes, support_dom, adaptive_eps,
                                          stet4_ip_coords, stet4_ip_weights);

        this->AdaptiveEightTetraDivisions(tetramesh, stet5_nodes, support_dom, adaptive_eps,
                                          stet5_ip_coords, stet5_ip_weights);

        this->AdaptiveEightTetraDivisions(tetramesh, stet6_nodes, support_dom, adaptive_eps,
                                          stet6_ip_coords, stet6_ip_weights);

        this->AdaptiveEightTetraDivisions(tetramesh, stet7_nodes, support_dom, adaptive_eps,
                                          stet7_ip_coords, stet7_ip_weights);

        this->AdaptiveEightTetraDivisions(tetramesh, stet8_nodes, support_dom, adaptive_eps,
                                          stet8_ip_coords, stet8_ip_weights);

        // Add integration points contribution from 1st sub-element.
        tet_ip_coords.reserve(stet1_ip_coords.size() + stet2_ip_coords.size() + stet3_ip_coords.size() + stet4_ip_coords.size() +
                              stet5_ip_coords.size() + stet6_ip_coords.size() + stet7_ip_coords.size() + stet8_ip_coords.size() );
        tet_ip_coords = stet1_ip_coords;
        tet_ip_coords.insert(tet_ip_coords.end(), stet2_ip_coords.begin(), stet2_ip_coords.end());
        tet_ip_coords.insert(tet_ip_coords.end(), stet3_ip_coords.begin(), stet3_ip_coords.end());
        tet_ip_coords.insert(tet_ip_coords.end(), stet4_ip_coords.begin(), stet4_ip_coords.end());
        tet_ip_coords.insert(tet_ip_coords.end(), stet5_ip_coords.begin(), stet5_ip_coords.end());
        tet_ip_coords.insert(tet_ip_coords.end(), stet6_ip_coords.begin(), stet6_ip_coords.end());
        tet_ip_coords.insert(tet_ip_coords.end(), stet7_ip_coords.begin(), stet7_ip_coords.end());
        tet_ip_coords.insert(tet_ip_coords.end(), stet8_ip_coords.begin(), stet8_ip_coords.end());

        // Add integration points contribution from 2nd sub-element.
        tet_ip_weights.reserve(stet1_ip_weights.size() + stet2_ip_weights.size() + stet3_ip_weights.size() + stet4_ip_weights.size());
        tet_ip_weights = stet1_ip_weights;
        tet_ip_weights.insert(tet_ip_weights.end(), stet2_ip_weights.begin(), stet2_ip_weights.end());
        tet_ip_weights.insert(tet_ip_weights.end(), stet3_ip_weights.begin(), stet3_ip_weights.end());
        tet_ip_weights.insert(tet_ip_weights.end(), stet4_ip_weights.begin(), stet4_ip_weights.end());
        tet_ip_weights.insert(tet_ip_weights.end(), stet5_ip_weights.begin(), stet5_ip_weights.end());
        tet_ip_weights.insert(tet_ip_weights.end(), stet6_ip_weights.begin(), stet6_ip_weights.end());
        tet_ip_weights.insert(tet_ip_weights.end(), stet7_ip_weights.begin(), stet7_ip_weights.end());
        tet_ip_weights.insert(tet_ip_weights.end(), stet8_ip_weights.begin(), stet8_ip_weights.end());
    }

}


Vec3<double> IntegPoints::IntegrationValue(const TetraMesh &tetramesh, const std::vector<Vec3<double> > &ip_coords,
                                           const std::vector<double> &ip_weights, const SupportDomain &support_dom)
{
    // Check if given integration points coordinates are initialized.
    if (ip_coords.size() == 0) {
        std::string error = "[ExplicitSim ERROR] Cannot compute integration value. "
                            "Given integration points coordinates container is empty.";
        throw std::runtime_error(error.c_str());
    }

    // Check if given integration points weights are initialized.
    if (ip_weights.size() == 0) {
        std::string error = "[ExplicitSim ERROR] Cannot compute integration value. "
                            "Given integration points weights container is empty.";
        throw std::runtime_error(error.c_str());
    }

    // Initialize the integration value vector.
    Vec3<double> val(0., 0., 0.);

    // Compute tetrahedral neighbor nodes indices for the integration points.
    auto neigh_ids = support_dom.ClosestNodesIdsTo(ip_coords);

    // Compute mmls shape functions.
    Mmls3d mmls3d;
    mmls3d.SetBasisFunctionType("linear");
    mmls3d.SetExactDerivativesMode(true);
    mmls3d.ComputeShFuncAndDerivs(tetramesh.Nodes(), tetramesh.NodesNum(), ip_coords, (uint32_t)ip_coords.size(), neigh_ids, support_dom.InfluenceNodesRadiuses());

    // Iterate over all the integration points.
    for (const auto &ip_coord : ip_coords) {
        // The integration point index.
        auto ip = &ip_coord - &ip_coords[0];

        // Gather the values of x, y, z derivatives of the shape function values at the integration point ip.
        Eigen::ArrayXd ip_derx_sq = mmls3d.ShapeFunctionDx().col(ip);
        Eigen::ArrayXd ip_dery_sq = mmls3d.ShapeFunctionDy().col(ip);
        Eigen::ArrayXd ip_derz_sq = mmls3d.ShapeFunctionDz().col(ip);

        // Square the x, y, z derivatives.
        ip_derx_sq *= ip_derx_sq;
        ip_dery_sq *= ip_dery_sq;
        ip_derz_sq *= ip_derz_sq;

        // Add the contibution of each ip to the integration value.
        val.SetX(val.X() + ip_weights[ip]*ip_derx_sq.sum());
        val.SetY(val.Y() + ip_weights[ip]*ip_dery_sq.sum());
        val.SetZ(val.Z() + ip_weights[ip]*ip_derz_sq.sum());
    }

    // Return the integration value.
    return val;

}

} //end of namespace ExplicitSim
