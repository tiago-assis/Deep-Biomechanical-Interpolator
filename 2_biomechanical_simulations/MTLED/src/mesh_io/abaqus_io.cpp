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



#include "ExplicitSim/mesh_io/abaqus_io.hpp"

namespace ExplicitSim {


AbaqusIO::AbaqusIO() : nodes_startline_(-1), elems_startline_(-1), S3_elems_startline_(-1),
                       partitions_exist(false), node_sets_exist(false)
{}


AbaqusIO::~AbaqusIO()
{}


void AbaqusIO::LoadMeshFrom(const std::string &mesh_filename)
{
    // Print reading status message.
    std::cout << "[ExplicitSim] Reading mesh file: \"" + mesh_filename + "\"\n";

    // Clear containers.
    this->input_mesh_.clear();
    this->parts_startlines_.clear();
    this->node_sets_startlines_.clear();

    // Check if mesh filename is given.
    if (mesh_filename.empty()) {
        std::string error = "[ExplicitSim ERROR] No filename was given to read mesh.";
        throw std::invalid_argument(error.c_str());
    }

    // Check if mesh is in abaqus format (.inp).
    std::string ext = mesh_filename.substr(mesh_filename.length()-4);
    if (ext != ".inp") {
        std::string error = "[ExplicitSim ERROR] The mesh \"" + mesh_filename + "\" is not in Abaqus format (.inp)";
        throw std::invalid_argument(error.c_str());
    }

    //Open mesh file.
    std::ifstream mesh_file(mesh_filename.c_str(), std::ios::in);

    // Check if mesh file opened successfully.
    if (!mesh_file.is_open()) {
        std::string error = "[ExplicitSim ERROR] Could not open the mesh file: \"" + mesh_filename + "\". Check given path.";
        throw std::runtime_error(error.c_str());
    }

    //Available mesh information.
    bool nodes_exist = false;
    bool elems_exist = false;

    //Line of the file.
    std::string line = "";
    int line_id = 0;

    //Read mesh file line by line.
    while (std::getline(mesh_file, line)) {
        // Transform mesh file line in lowercase.
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);

        // Read mesh file lines.
        this->input_mesh_.push_back(line);

        // Find the first line of the nodes set.
        if (line.find("*node") != std::string::npos) {
            this->nodes_startline_ = line_id;
            nodes_exist = true;
        }

        // Find the first line of the elements set.
        if (line.find("*element") != std::string::npos) 
		{
			// The elements type available in the mesh.
			std::string elements_type = line.substr(line.find("type="));

			if (elements_type.find("type=c3d4") != std::string::npos)
			{
				this->elems_startline_ = line_id;
				elems_exist = true;
			}
			else if (elements_type.find("type=s3") != std::string::npos)
			{
				this->S3_elems_startline_ = line_id;
			}
        }

		// Find the first lines of the partitions sets.
        if (line.find("*elset") != std::string::npos) {
            this->parts_startlines_.push_back(line_id);
            if (!this->partitions_exist) { this->partitions_exist = true; }
        }

        // Find the first lines of the boundary node sets.
        if (line.find("*nset") != std::string::npos) {
            this->node_sets_startlines_.push_back(line_id);
            if (!this->node_sets_exist) { this->node_sets_exist = true; }
        }

        // Increase line id once it is processed.
        line_id++;

    }

    // Check if nodes and elements are available in the mesh file.
    if (!nodes_exist && !elems_exist) {
        std::string error = "[ExplicitSim ERROR] Mesh file: " + mesh_filename + " is incomplete. "
                "Either nodes or elements are not available...";
		throw std::runtime_error(error.c_str());
    }

    // Close the mesh file.
    mesh_file.close();

}


void AbaqusIO::LoadNodesIn(std::vector<Node> &nodes)
{
    // Clean the mesh nodes container.
    nodes.clear();
	this->offsetted_nodes_.clear();

    // The coordinates of the nodes in the mesh.
    double x = 0.; double y = 0.; double z = 0.;

    // The id of the nodes in the mesh. Initialized to invalid value (-1).
    int32_t id = -1;
	int32_t new_id = 0;
    // The mesh node.
    Node node;

    // Iterate though mesh starting from the nodes set starting line.
    // Skip the first line start from the first node's coordinates.
    std::string line = "";
    for (std::vector<std::string>::size_type it = this->nodes_startline_ + 1;
         it < input_mesh_.size(); ++it) {

        line = this->input_mesh_.at(it);

        // Replace comma occurence in line with space.
        std::replace(line.begin(), line.end(), ',', ' ');

        // Convert line to istringstream to be passed in the coordinates variables.
        std::stringstream ss(line);

        // Get the coordinates until reach the end of the vertices set.
        if (!(ss >> id >> x >> y >> z)) {
			break; 
		}

        // Set the id and coordinates of the mesh node. 
        node.SetId(new_id);
        node.SetCoordinates(x, y, z);

        // Store the node in the nodes' container.
        nodes.emplace_back(node);

		this->offsetted_nodes_.push_back(std::make_pair(id, new_id));

		new_id++;
    }
}

void AbaqusIO::LoadSurfacesIn(std::vector<Surface> &surfaces)
{
	if (this->S3_elems_startline_ >= 0)
	{
		// The tetrahedron's index. Initialized in invalid value (-1).
		int32_t id = -1;
		std::vector<int32_t> aConnectivity;

		// The triangle's connectivity. Initialized in invalid value (-1).
		int32_t n1 = -1; int32_t n2 = -1; int32_t n3 = -1; 

		// The header line of the elements set.
		std::string elements_start = this->input_mesh_.at(this->S3_elems_startline_);

		// Iterate though mesh starting from the elements set starting line.
		// Skip the first line to start from the first element's connectivity.
		std::string line = "";
		for (std::vector<std::string>::size_type it = this->S3_elems_startline_ + 1;
			it != input_mesh_.size(); ++it) {

			line = this->input_mesh_.at(it);

			// Replace comma occurence in line with space.
			std::replace(line.begin(), line.end(), ',', ' ');

			// Convert line to stringstream to be passed in the connectivity variables.
			std::stringstream ss(line);

			// Get the id and connectivity until reach the end of the vertices set.
			if (!(ss >> id >> n1 >> n2 >> n3)) { break; }

			// Correct connectivity for nodes
			// Check for offset at node N1.
			auto offsetted_n1 = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
				[&](const std::pair<int32_t, int32_t> &element) { return element.first == n1; });

			auto offsetted_n2 = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
				[&](const std::pair<int32_t, int32_t> &element) { return element.first == n2; });

			auto offsetted_n3 = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
				[&](const std::pair<int32_t, int32_t> &element) { return element.first == n3; });

			// Apply offset correction if necessary.
			if ((offsetted_n1 == this->offsetted_nodes_.end()) ||
				(offsetted_n2 == this->offsetted_nodes_.end()) ||
				(offsetted_n3 == this->offsetted_nodes_.end()))
			{
				std::string error = "[ExplicitSim ERROR] Triangle node number not found in the list of nodes.";
				throw std::runtime_error(error.c_str());
			}

			aConnectivity.emplace_back(offsetted_n1->second);
			aConnectivity.emplace_back(offsetted_n2->second);
			aConnectivity.emplace_back(offsetted_n3->second);
		}
		Surface surf((unsigned long)aConnectivity.size()/3);

		for (std::vector<int32_t>::size_type it = 0; it < aConnectivity.size(); it++)
		{
			surf.pulConnectivity()[it] = (unsigned long)aConnectivity.at(it);
		}
		surfaces.emplace_back(surf);
	}
}

void AbaqusIO::LoadElementsIn(std::vector<Tetrahedron> &tetras)
{
    // Clean the mesh tetrahedra container.
    tetras.clear();
	this->offsetted_elems_.clear();

    // A tetrahedron of the mesh.
    Tetrahedron tetrahedron;

    // The tetrahedron's index. Initialized in invalid value (-1).
    int32_t id = -1;
	int32_t new_id = 0;

    // The tetrahedron's connectivity. Initialized in invalid value (-1).
	int32_t n1 = -1; int32_t n2 = -1; int32_t n3 = -1; int32_t n4 = -1;

    // The header line of the elements set.
    std::string elements_start = this->input_mesh_.at(this->elems_startline_);

    // Iterate though mesh starting from the elements set starting line.
    // Skip the first line to start from the first element's connectivity.
    std::string line = "";
    for (std::vector<std::string>::size_type it = this->elems_startline_ + 1;
         it != input_mesh_.size(); ++it) {

        line = this->input_mesh_.at(it);

        // Replace comma occurence in line with space.
        std::replace(line.begin(), line.end(), ',', ' ');

        // Convert line to stringstream to be passed in the connectivity variables.
        std::stringstream ss(line);

        // Get the id and connectivity until reach the end of the vertices set.
        if (!(ss >> id >> n1 >> n2 >> n3 >> n4)) { break; }

        tetrahedron.SetId(new_id);
		this->offsetted_elems_.push_back(std::make_pair(id, new_id));

		new_id++;
		
		// Correct connectivity for nodes
		// Check for offset at node N1.
		auto offsetted_n1 = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
			[&](const std::pair<int32_t, int32_t> &element) { return element.first == n1; });

		auto offsetted_n2 = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
			[&](const std::pair<int32_t, int32_t> &element) { return element.first == n2; });

		auto offsetted_n3 = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
			[&](const std::pair<int32_t, int32_t> &element) { return element.first == n3; });

		auto offsetted_n4 = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
			[&](const std::pair<int32_t, int32_t> &element) { return element.first == n4; });

		// Apply offset correction if necessary.
		if ((offsetted_n1 == this->offsetted_nodes_.end()) ||
			(offsetted_n2 == this->offsetted_nodes_.end()) ||
			(offsetted_n3 == this->offsetted_nodes_.end()) ||
			(offsetted_n4 == this->offsetted_nodes_.end()))
		{
			std::string error = "[ExplicitSim ERROR] Tetra node number not found in the list of nodes.";
			throw std::runtime_error(error.c_str());
		}

        tetrahedron.SetConnectivity(offsetted_n1->second, offsetted_n2->second, offsetted_n3->second, offsetted_n4->second);

        // Store in the tetrhadra container.
        tetras.emplace_back(tetrahedron);
    }
}


void AbaqusIO::LoadPartitionsIn(std::vector<Tetrahedron> &tetras)
{
    // The tetrahedron's index. Initialized in invalid value (-1).
	int32_t tetra_id = -1;

    // The headerline of the current part. Used in partitions iteration.
    std::string current_part_headerline = "";

    // The name of the current part. Used in partitions iteration.
    std::string current_part_name = "";

    // The mesh file line to be processed. Used in partitions iteration.
    std::string line = "";

    // Iterate though mesh starting from the partition set starting line (for each partition).
    // Skip the first line to start from the partition's elements.
    for (auto &current_part_startline: this->parts_startlines_) {

        auto part_id = &current_part_startline - &this->parts_startlines_[0];

        current_part_headerline = this->input_mesh_.at(current_part_startline);

        current_part_name = current_part_headerline.substr(current_part_headerline.find_last_of("=")+1);

        // Remove last character of current part name if it is the new line character.
//        if (current_part_name.back() == '\n') { current_part_name.pop_back(); }

        current_part_name.erase(std::remove_if(current_part_name.begin(), current_part_name.end(),
                                               [](char c) { return !isalnum(c); }), current_part_name.end());

        for (std::vector<std::string>::size_type it = current_part_startline+1;
             it != input_mesh_.size(); ++it) {

            line = this->input_mesh_.at(it);

            // Replace comma occurence in line with space.
            std::replace(line.begin(), line.end(), ',', ' ');

            // Convert line to stringstream to be passed in the connectivity variables.
            std::stringstream ss(line);

            while (ss >> tetra_id) 
			{
                // Check for offset at element in partition.
                auto offsetted_elem = std::find_if(this->offsetted_elems_.begin(), this->offsetted_elems_.end(),
                        [&](const std::pair<int32_t, int32_t> &element){ return element.first == tetra_id; } );

                // Apply offset correction if necessary.
                if (offsetted_elem == this->offsetted_elems_.end()) {
					std::string error = "[ExplicitSim ERROR] Element number not found in the list of elements.";
					throw std::runtime_error(error.c_str());
                }

                // Set the partition.
                tetras[offsetted_elem->second].SetPartition((int)part_id, current_part_name);
            }

        }

    }

}


void AbaqusIO::LoadBoundarySetsIn(std::vector<NodeSet> &node_sets)
{
    // Clear and allocate node_sets.
    node_sets.clear();
    node_sets.reserve(this->node_sets_startlines_.size());

    // The node's index. Initialized in invalid value (-1).
	int32_t node_id = -1;

    // The headerline of the current nodeset. Used in nodesets iteration.
    std::string current_nset_headerline = "";

    // The name of the current nodeset. Used in nodesets iteration.
    std::string current_nset_name = "";

    // The mesh file line to be processed. Used in nodesets iteration.
    std::string line = "";

    // Iterate though mesh starting from the nodeset starting line (for each nodeset).
    for (auto &current_nset_startline: this->node_sets_startlines_) {

        current_nset_headerline = this->input_mesh_[current_nset_startline];

        // Remove spaces from header line.
        current_nset_headerline.erase(std::remove(current_nset_headerline.begin(), current_nset_headerline.end(), ' ' ),
                                      current_nset_headerline.end());

        current_nset_name = current_nset_headerline.substr(current_nset_headerline.find_last_of("=")+1);


        // Remove special (invinsible) characters from current nodeset name.
        current_nset_name.erase(std::remove_if(current_nset_name.begin(), current_nset_name.end(),
                                               [](unsigned char c) { return !std::isprint(c); }), current_nset_name.end());

        NodeSet current_nset;
        current_nset.SetNodeSetName(current_nset_name);

        // Skip the first line to start from the nodeset's nodes.
        for (std::vector<std::string>::size_type it = current_nset_startline+1;
             it != input_mesh_.size(); ++it) {

            line = this->input_mesh_[it];

			// Stop processing the mesh file for current nodeset when the next one is found.
			if (line.find("*") != std::string::npos) { break; }

            // Replace comma occurence in line with space.
            std::replace(line.begin(), line.end(), ',', ' ');

            // Convert line to stringstream to be passed in the connectivity variables.
            std::stringstream ss(line);

			while (ss >> node_id)
			{

				// Check for offset at boundary node.
				auto offsetted_node = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
					[&](const std::pair<int32_t, int32_t> &element) { return element.first == node_id; });

				if (offsetted_node == this->offsetted_nodes_.end()) {
					std::string error = "[ExplicitSim ERROR] Set node number not found in the list of nodes.";
					throw std::runtime_error(error.c_str());
				}

				// Add node id in the current nodeset.
				current_nset.EditNodeIds().emplace_back(offsetted_node->second);

			}
        }

        // Add current nodeset in the nodesets container.
        node_sets.emplace_back(current_nset);
    }


}




} // end of namespace ExplicitSim
