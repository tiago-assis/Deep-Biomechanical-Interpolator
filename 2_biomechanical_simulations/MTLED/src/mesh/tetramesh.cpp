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


#include "ExplicitSim/mesh/tetramesh.hpp"
#include "ExplicitSim/mesh_io/abaqus_io.hpp"
#include "ExplicitSim/mesh_io/febio_io.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared_array.hpp>

namespace ExplicitSim {


TetraMesh::TetraMesh() : mesh_type_(ExplicitSim::MeshType::tetrahedral), pMesh_io_(NULL), pui32SortedNodeArray(NULL), 
pui32ReindexingArray(NULL), pdNodesCoordinates(NULL)
{}

TetraMesh::~TetraMesh()
{}


void TetraMesh::LoadFrom(const std::string &mesh_filename)
{
    // Check if mesh filename is not empty.
    if (mesh_filename.empty()) {
        throw std::invalid_argument(Logger::Error("Could not load mesh. No mesh filename was given.").c_str());
    }

	// Get the extension of the mesh filename.
    auto ext = mesh_filename.substr(mesh_filename.length()-4);

    // Clear the mesh containers.
    this->nodes_.clear();
    this->tetras_.clear();
    this->node_sets_.clear();

    // Load the corresponding format.
    if (ext == ".inp") {
		ExplicitSim::AbaqusIO *pAbaqusIO = new ExplicitSim::AbaqusIO;
		pAbaqusIO->LoadMeshFrom(mesh_filename.c_str());
		pAbaqusIO->LoadNodesIn(this->nodes_);
		pAbaqusIO->LoadElementsIn(this->tetras_);
        if (pAbaqusIO->PartitionsExist()) {
			pAbaqusIO->LoadPartitionsIn(this->tetras_);
        }
        if (pAbaqusIO->NodeSetsExist()) {
			pAbaqusIO->LoadBoundarySetsIn(this->node_sets_);
        }
		pAbaqusIO->LoadSurfacesIn(this->surfaces_);
		pMesh_io_ = boost::shared_ptr<ExplicitSim::MeshIO>(pAbaqusIO);

    }
    else if (ext == ".feb") {
		ExplicitSim::FebioIO *pFebioIO = new ExplicitSim::FebioIO;
		pFebioIO->LoadMeshFrom(mesh_filename.c_str());
		pFebioIO->LoadNodesIn(this->nodes_);
		pFebioIO->LoadElementsIn(this->tetras_);
        if (pFebioIO->BoundariesExist()) {
			pFebioIO->LoadBoundarySetsIn(this->node_sets_);
        }
		pMesh_io_ = boost::shared_ptr<ExplicitSim::MeshIO>(pFebioIO);
    }
    else {
        std::string error = Logger::Error("Could not load mesh of unkown format. Expected [.inp | .feb] Check: ") + mesh_filename;
        throw std::invalid_argument(error.c_str());
    }
	 // reindex the nodes and use only the ones that appear in the mesh
	uint32_t ui32NumNodes = (uint32_t)this->nodes_.size();
	boost::shared_ptr<uint8_t[]> pu8UsedNodes(new uint8_t[ui32NumNodes]);
	pui32SortedNodeArray = boost::make_shared<uint32_t[]>(ui32NumNodes);  
	pui32ReindexingArray = boost::make_shared<uint32_t[]>(ui32NumNodes);
	pdNodesCoordinates = boost::make_shared<double[]>(3*ui32NumNodes);
	for (uint32_t i = 0; i < ui32NumNodes; i++)
	{
		pu8UsedNodes[i] = 0;
	}
	 // mark used nodes
	for (uint32_t t = 0; t < this->tetras_.size(); t++)
	{
		pu8UsedNodes[this->tetras_[t].N1()] = 1;
		pu8UsedNodes[this->tetras_[t].N2()] = 1;
		pu8UsedNodes[this->tetras_[t].N3()] = 1;
		pu8UsedNodes[this->tetras_[t].N4()] = 1;
	}
	// reindex all used nodes
	ui32NumUsedNodes = 0;
	for (uint32_t i = 0; i < ui32NumNodes; i++)
	{
		if (pu8UsedNodes[i] > 0)
		{
			pui32SortedNodeArray[ui32NumUsedNodes] = i;
			ui32NumUsedNodes++;
		}
	}
	uint32_t ui32UnusedNodesIdx = ui32NumUsedNodes;
	for (uint32_t i = 0; i < ui32NumNodes; i++)
	{
		if (pu8UsedNodes[i] == 0)
		{
			pui32SortedNodeArray[ui32UnusedNodesIdx] = i;
			ui32UnusedNodesIdx++;
		}
	}
	// reindex the nodes
	original_nodes_ = nodes_;
	for (uint32_t i = 0; i < ui32NumNodes; i++)
	{
		nodes_[i] = original_nodes_[pui32SortedNodeArray[i]];
	}
	// invert the sorted node array to compute the reindexing array
	for (uint32_t i = 0; i < ui32NumNodes; i++)
	{
		pui32ReindexingArray[pui32SortedNodeArray[i]] = i;
	}

	// reindex all nodes in the mesh
	for (uint32_t t = 0; t < this->tetras_.size(); t++)
	{
		this->tetras_[t].SetConnectivity(pui32ReindexingArray[this->tetras_[t].N1()], pui32ReindexingArray[this->tetras_[t].N2()], 
			pui32ReindexingArray[this->tetras_[t].N3()], pui32ReindexingArray[this->tetras_[t].N4()]);
	}

	// reindex all nodes in the node sets
	for (uint32_t s = 0; s < this->node_sets_.size(); s++)
	{
		for (uint32_t t = 0; t < this->node_sets_[s].EditNodeIds().size(); t++)
		{
			this->node_sets_[s].EditNodeIds()[t] = pui32ReindexingArray[this->node_sets_[s].EditNodeIds()[t]];
		}
	}
	// reindex all nodes in the surfaces
	for (uint32_t s = 0; s < this->surfaces_.size(); s++)
	{
		for (uint32_t t = 0; t < 3*this->surfaces_[s].ulNumTriangles(); t++)
		{
			uint32_t idx = this->surfaces_[s].pulConnectivity()[t];
			uint32_t new_idx = pui32ReindexingArray[idx];
			this->surfaces_[s].pulConnectivity()[t] = pui32ReindexingArray[idx];
		}
	}
	// get nodes coordinates
	for (uint32_t i = 0; i < nodes_.size(); i++)
	{
		ExplicitSim::Node &n = nodes_[i];
		pdNodesCoordinates[3 * i] = n.Coordinates().X();
		pdNodesCoordinates[3 * i + 1] = n.Coordinates().Y();
		pdNodesCoordinates[3 * i + 2] = n.Coordinates().Z();
	}
}


void TetraMesh::SaveTo(const std::string &mesh_filename)
{
    // Check if mesh filename is given.
    if (mesh_filename.empty()) {
        std::string error = "ERROR: No filename was given to save the mesh.";
        throw std::invalid_argument(error.c_str());
    }

    // Get the extension of the mesh filename.
    auto ext = mesh_filename.substr(mesh_filename.length()-4);


    if (ext == ".inp") {
        AbaqusIO abaqus_io;
        abaqus_io.SaveMesh<TetraMesh,Tetrahedron>(*this, mesh_filename.c_str());
    }
    else {
        std::string error = "ERROR: Given mesh file: \"" + mesh_filename + "\" is of unknown format.";
        throw std::invalid_argument(error.c_str());
    }
}


bool TetraMesh::operator == (const TetraMesh &tetramesh) const
{
    // Compare tetrahedral meshes for equality.
    return ((this->nodes_ == tetramesh.nodes_) &&
            (this->node_sets_ == tetramesh.node_sets_) &&
            (this->tetras_ == tetramesh.tetras_) &&
            (this->mesh_type_ == tetramesh.mesh_type_)
           );
}


bool TetraMesh::operator != (const TetraMesh &tetramesh) const
{
    // Compare tetrahedral meshes for inequality.
    return !(*this == tetramesh);
}



}  //end of namespace ExplicitSim
