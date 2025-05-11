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


#include "ExplicitSim/mesh_io/febio_io.hpp"


namespace ExplicitSim {


FebioIO::FebioIO() : is_mesh_loaded(false), boundaries_exist(false)
{}


FebioIO::~FebioIO()
{}


void FebioIO::LoadMeshFrom(const std::string &mesh_filename)
{
    // Print reading status message.
    std::cout << Logger::Message("Reading mesh file: ") << mesh_filename << "\n";

    // Check if mesh filename is empty.
    if (mesh_filename.empty()) {
        throw std::invalid_argument(Logger::Error("Could not load FeBio mesh. No mesh filename was given").c_str());
    }

    // Check if mesh is in FeBio format (.feb).
    std::string ext = mesh_filename.substr(mesh_filename.length()-4);
    if (ext != ".feb") {
        std::string error = Logger::Error("Could not load FeBio mesh. Expected format [.feb] Check: ") + mesh_filename;
        throw std::invalid_argument(error.c_str());
    }

    // Load mesh file.
    tinyxml2::XMLError load_result = this->input_mesh_.LoadFile(mesh_filename.c_str());
    if (load_result != tinyxml2::XML_SUCCESS) {
        std::string error = Logger::Error("Could not load FeBio mesh. Error while reading the mesh file. Check: ") + mesh_filename;
        throw std::invalid_argument(error.c_str());
    }

    // Declare that the mesh is loaded.
    this->is_mesh_loaded = true;

    // Get the root node of the FeBio XML tree.
    tinyxml2::XMLNode *xml_root = this->input_mesh_.RootElement();
    if (xml_root == nullptr) {
        std::string error = Logger::Error("Could not load FeBio mesh. Expected XML tree "
                                          "hierarchy was not found. Check: ") + mesh_filename;
        throw std::invalid_argument(error.c_str());
    }

    // Get Geometry branch from FeBio xml.
    tinyxml2::XMLElement *geom = xml_root->FirstChildElement("Geometry");
    if (geom == nullptr) {
        std::string error = Logger::Error("Load FeBio mesh aborted. XML Branch [Geometry] is missing. Check: ") + mesh_filename;
        throw std::invalid_argument(error.c_str());
    }

    // Check if node coordinates are available in the mesh file.
    tinyxml2::XMLElement *are_nodes = geom->FirstChildElement("Nodes");
    if (are_nodes == nullptr) {
        std::string error = Logger::Error("Load FeBio mesh aborted. XML Branch [Nodes] is missing. Check: ") + mesh_filename;
        throw std::invalid_argument(error.c_str());
    }

    // Check if elements are available in the mesh file.
    tinyxml2::XMLElement *are_elems = geom->FirstChildElement("Elements");
    if (are_elems == nullptr) {
        std::string error = Logger::Error("Load FeBio mesh aborted. XML Branch [Elements] is missing. Check: ") + mesh_filename;
        throw std::invalid_argument(error.c_str());
    }

    // Check if boundaries are available in the mesh file.
    tinyxml2::XMLElement *are_boundaries = xml_root->FirstChildElement("Boundary");
    if (are_boundaries != nullptr) { this->boundaries_exist = true; }

    // Reset created pointers.
    xml_root = nullptr; geom = nullptr; are_nodes = nullptr;
    are_elems = nullptr; are_boundaries = nullptr;

}


void FebioIO::LoadNodesIn(std::vector<Node> &nodes)
{
    // Check if the FeBio mesh has already been loaded to extract the nodes.
    if (!this->is_mesh_loaded) {
        throw std::runtime_error(Logger::Error("Can not load nodes from FeBio I/O manager. "
                                               "No FeBio mesh has been loaded.").c_str());
    }

    // Clean the mesh nodes container.
    nodes.clear();

    // Get the root node of the FeBio XML tree.
    tinyxml2::XMLNode *xml_root = this->input_mesh_.RootElement();

    // Get the mesh nodes.
    tinyxml2::XMLElement *nodes_list = xml_root->FirstChildElement("Geometry")->FirstChildElement("Nodes");

    // Get the first node's information.
    tinyxml2::XMLElement *node_info = nodes_list->FirstChildElement("node");

    // The node to be stored.
    Node node;

    // The coordinates of the nodes in the mesh.
    double x = 0.; double y = 0.; double z = 0.;

    // The id of the nodes in the mesh. Initialized to invalid value (-1).
    int id = -1;

    // Iterate over nodes.
    while (node_info != nullptr) {
        // Extract node's id. Reduce index by 1 to account for the storage offset.
        id = node_info->IntAttribute("id") - 1;

        // Extract node's coordinates in string format.
        std::string coords_string(node_info->GetText());

        // Replace comma occurence in coordintates string with space.
        std::replace(coords_string.begin(), coords_string.end(), ',', ' ');

        // Convert to string stream for parsing.
        std::stringstream coords_ss(coords_string);

        // Extract coordinate values from the string stream as long as it applies.
        if (!(coords_ss >> x >> y >> z)) { break; }

        // Set the id and coordinates of the node to be stored.
        node.SetId(id);
        node.SetCoordinates(x, y, z);

        //Store the node in the nodes' container.
        nodes.emplace_back(node);

        // Move to the next node's information.
        node_info = node_info->NextSiblingElement("node");

    } // End iterate over nodes.

    // Reset the offsetted nodes indices, if any.
	uint32_t node_offset = 0;
    for (auto &node : nodes) {
        // Index of the node in the container.
		uint32_t id = (uint32_t)(&node - &nodes[0]);

        // Check consistency with stored node's id.
        if (id != node.Id()) {
            node_offset = node.Id() - id;

            // Store the offsetted node's id and it's offset from storage.
            this->offsetted_nodes_.emplace_back(std::make_pair(node.Id(), node_offset));

            // Reset the node's id.
            node.SetId(node.Id() - node_offset);
        }
    } // End Reset the offsetted nodes indices, if any.

    // Reset created pointers.
    xml_root = nullptr; nodes_list = nullptr; node_info = nullptr;

}


void FebioIO::LoadElementsIn(std::vector<Tetrahedron> &tetras)
{
    // Check if the FeBio mesh has already been loaded to extract the nodes.
    if (!this->is_mesh_loaded) {
        throw std::runtime_error(Logger::Error("Can not load nodes from FeBio I/O manager. "
                                               "No FeBio mesh has been loaded.").c_str());
    }

    // Clean the mesh tetrahedra container.
    tetras.clear();

    // A tetrahedron of the mesh.
    Tetrahedron tetrahedron;

    // The tetrahedron's index. Initialized in invalid value (-1).
    int id = -1;

    // The tetrahedron's connectivity. Initialized in invalid value (-1).
    int n1 = -1; int n2 = -1; int n3 = -1; int n4 = -1;

    // Get the root node of the FeBio XML tree.
    tinyxml2::XMLNode *xml_root = this->input_mesh_.RootElement();

    // Get the mesh elements set.
    tinyxml2::XMLElement *elem_set = xml_root->FirstChildElement("Geometry")->FirstChildElement("Elements");

    // The first element's information.
    tinyxml2::XMLElement *elem_info = nullptr;

    // Iterate over all element sets if more than one are available.
    while (elem_set != nullptr) {
        // Check if elements are tetrahedras.
        std::string elem_type(elem_set->Attribute("type"));
        if (elem_type != "tet4") {
            throw std::invalid_argument(Logger::Error("Could not load elements from FeBio I/O manager. "
                                                      "Unsupported type of elements was occured. Expected [tet4].").c_str());
        }

        // Get the elements set name.
        std::string elem_set_name(elem_set->Attribute("elset"));
        std::transform(elem_set_name.begin(), elem_set_name.end(), elem_set_name.begin(), ::tolower);

        // Get the first element's information.
        elem_info = elem_set->FirstChildElement("elem");

        // Iterate over elements to extract information.
        while (elem_info != nullptr) {

            // Extract element's id. Reduce index by 1 to account for the storage offset.
            id = elem_info->IntAttribute("id") - 1;
            tetrahedron.SetId(id);

            // Extract element's connectivity in string format.
            std::string conn_string(elem_info->GetText());

            // Replace comma occurence in connectivity string with space.
            std::replace(conn_string.begin(), conn_string.end(), ',', ' ');

            // Convert to string stream for parsing.
            std::stringstream conn_ss(conn_string);

            // Extract connectivity values from the string stream as long as it applies.
            if (!(conn_ss >> n1 >> n2 >> n3 >> n4)) { break; }

            // Correct connectivity for c++ storage by paddling 1.
            tetrahedron.SetConnectivity(n1 - 1, n2 - 1, n3 - 1, n4 - 1);

            // Set partition name with dummy id. Id will be deprecated.
            tetrahedron.SetPartition(-1, elem_set_name);

            // Store in the tetrhadra container.
            tetras.emplace_back(tetrahedron);

            // Move to the next element's information.
            elem_info = elem_info->NextSiblingElement("elem");

        } // End Iterate over elements to extract information.

        // Move to the next elements set.
        elem_set = elem_set->NextSiblingElement("Elements");

    } // End Iterate over all element sets if more than one are available.


    // Correct connectivity for offsetted nodes if necessary.
    if (this->offsetted_nodes_.size() != 0) {
        std::cout << Logger::Message("Correcting mesh connectivity for nodes ordering inconsistency...") << "\n";

        // Iterate over tetrahedra.
        for (auto &tetra : tetras) {

            // Check for offset at node N1.
            auto offsetted_n1 = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
                [&](const std::pair<int32_t, int32_t> &element){ return element.first == tetra.N1(); } );

            // Apply offset correction if necessary.
            if (offsetted_n1 != this->offsetted_nodes_.end()) {
                tetra.SetN1(tetra.N1() - offsetted_n1->second);
            }

            // Check for offset at node N2.
            auto offsetted_n2 = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
                [&](const std::pair<int32_t, int32_t> &element){ return element.first == tetra.N2(); } );

            // Apply offset correction if necessary.
            if (offsetted_n2 != this->offsetted_nodes_.end()) {
                tetra.SetN2(tetra.N2() - offsetted_n2->second);
            }

            // Check for offset at node N3.
            auto offsetted_n3 = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
                [&](const std::pair<int32_t, int32_t> &element){ return element.first == tetra.N3(); } );

            // Apply offset correction if necessary.
            if (offsetted_n3 != this->offsetted_nodes_.end()) {
                tetra.SetN3(tetra.N3() - offsetted_n3->second);
            }

            // Check for offset at node N4.
            auto offsetted_n4 = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
                [&](const std::pair<int32_t, int32_t> &element){ return element.first == tetra.N4(); } );

            // Apply offset correction if necessary.
            if (offsetted_n4 != this->offsetted_nodes_.end()) {
                tetra.SetN4(tetra.N4() - offsetted_n4->second);
            }
        } // End Iterate over tetrahedra.

        std::cout << Logger::Message("Mesh connectivity corrected!") << "\n";

    } // End Correct connectivity for offsetted nodes if necessary.

    // Reset the offsetted elements indices, if any.
	uint32_t tetra_offset = 0;
    for (auto &tetra : tetras) {
        // Index of the node in the container.
		uint32_t id = (uint32_t)(&tetra - &tetras[0]);

        // Check consistency with stored node's id.
        if (id != tetra.Id()) {
            tetra_offset = tetra.Id() - id;

            // Store the offsetted node's id and it's offset from storage.
            this->offsetted_elems_.push_back(std::make_pair(tetra.Id(), tetra_offset));

            // Reset the element's id.
            tetra.SetId(tetra.Id() - tetra_offset);
        }
    } // End Reset the offsetted elements indices, if any.

    // Reset created pointers.
    xml_root = nullptr; elem_set = nullptr; elem_info = nullptr;

}


void FebioIO::LoadBoundarySetsIn(std::vector<NodeSet> &node_sets)
{
    // Check if the FeBio mesh has already been loaded to extract the nodes.
    if (!this->is_mesh_loaded) {
        throw std::runtime_error(Logger::Error("Can not load nodes from FeBio I/O manager. "
                                               "No FeBio mesh has been loaded.").c_str());
    }

    // Check if boundaries exist for extraction.
    if (!this->boundaries_exist) {
        std::cout << Logger::Warning("Boundaries were not available for extraction "
                                     "by FeBio I/O manager. Request was skipped") << "\n";
        return;
    }

    // Clear node_sets where boundaries will be assigned.
    node_sets.clear();

    // Get the root node of the FeBio XML tree.
    tinyxml2::XMLNode *xml_root = this->input_mesh_.RootElement();

    // Get the first boundary.
    tinyxml2::XMLElement *boundary = xml_root->FirstChildElement("Boundary")->FirstChildElement();

    // Initialize pointer to the first node index of the boundary.
    tinyxml2::XMLElement *bound_node_id = nullptr;

    // Iterate over boundaries.
    while (boundary != nullptr) {

        // Initialize the boundary node set.
        NodeSet bound_nset;

        // Set node set's name.
        if (boundary->Attribute("set") == nullptr) {
            throw std::runtime_error(Logger::Error("Could not load boundary nodes from FeBio file. "
                                                   "The attribute [set] is missing from one or more boundary conditions.").c_str());
        }
        std::string bound_name(boundary->Attribute("set"));
        std::transform(bound_name.begin(), bound_name.end(), bound_name.begin(), ::tolower);
        bound_nset.SetNodeSetName(bound_name);

        // Get fist node index of the boundary.
        bound_node_id = boundary->FirstChildElement("node");

        // Iterate over the boundary node indices.
        while (bound_node_id != nullptr) {
            // Get the current boundary node index. Reduce by 1 to correct for storage.
			int32_t id = bound_node_id->IntAttribute("id") - 1;

            // Correct for offsetted nodes if necessary.
            if (this->offsetted_nodes_.size() != 0) {

                // Check for offset at boundary node.
                auto offsetted_node = std::find_if(this->offsetted_nodes_.begin(), this->offsetted_nodes_.end(),
                      [&](const std::pair<int32_t, int32_t> &element){ return element.first == id; } );

                // Apply offset correction if necessary.
                if (offsetted_node != this->offsetted_nodes_.end()) {
                    id -= offsetted_node->second;
                }
            }

            // Add node id in the current nodeset.
            bound_nset.EditNodeIds().emplace_back(id);

            // Move to the next boundary node index.
            bound_node_id = bound_node_id->NextSiblingElement("node");

        } // End Iterate over the boundary node indices.

        // Add boundary node set to the node_sets container.
        node_sets.emplace_back(bound_nset);

        // Move to the next boundary.
        boundary = boundary->NextSiblingElement();

    } // End Iterate over boundaries.

    // Reset created pointers.
    xml_root = nullptr; boundary = nullptr; bound_node_id = nullptr;

}



} // End of namespace ExplicitSim
