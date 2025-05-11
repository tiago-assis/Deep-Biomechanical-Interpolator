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

#include "ExplicitSim/conditions/conditions_handler.hpp"


namespace ExplicitSim {
// fMaxAbsDisplacement changed from 0 to 0.015
ConditionsHandler::ConditionsHandler(uint32_t u32NumActiveNodes, const std::vector<NodeSet> *pNode_sets) : fMaxAbsDisplacement(0.015),u32NumActiveNodes_(u32NumActiveNodes),
pu32DOFsDisplacedIdx_(NULL), pu32DOFsExternalForcesIdx_(NULL), pu32DOFsFixedIdx_(NULL), u32NumDisplacedDOFs(0), u32NumExternalForcesDOFs(0), u32NumFixedDOFs(0), u32NumEssentialDOFs(0), boUseEBCIEM_(false), u32NumEssentialDOFsX(0), 
u32NumEssentialDOFsY(0), u32NumEssentialDOFsZ(0)
{
	pNode_sets_ = pNode_sets;
	pdPreviousNodalPositions_ = boost::make_shared<double[]>(3 * u32NumActiveNodes_);
	pdCurrentNodalPositions_ = boost::make_shared<double[]>(3 * u32NumActiveNodes_);
	pfDOFsDisplacements_ = boost::make_shared<float[]>(3 * u32NumActiveNodes_);
	pi32DOFsLoadCurves_ = boost::make_shared<int32_t[]>(3 * u32NumActiveNodes_);
	pu8DOFsDisplaced_ = boost::make_shared<uint8_t[]>(3 * u32NumActiveNodes_);
	pu8DOFsFixed_ = boost::make_shared<uint8_t[]>(3 * u32NumActiveNodes_);
	pu8NodesContacts_ = boost::make_shared<uint8_t[]>(u32NumActiveNodes_);
	pu8NodesContactsActive_ = boost::make_shared<uint8_t[]>(u32NumActiveNodes_);
	// External forces
	pu8DOFsExternal_ = boost::make_shared<uint8_t[]>(3 * u32NumActiveNodes_);
	pfDOFsExternalForces_ = boost::make_shared<float[]>(3 * u32NumActiveNodes_);
	//Gravity
	pfGravity_ = boost::make_shared<float[]>(1 * 3);
	for (uint32_t i = 0; i < 3 * u32NumActiveNodes_; i++)
	{
		pu8DOFsDisplaced_[i] = 0;
		pu8DOFsFixed_[i] = 0;
		pu8DOFsExternal_[i] = 0;
	}

	for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
	{
		pu8NodesContacts_[i] = 0;
		pu8NodesContactsActive_[i] = 0;
	}
}


ConditionsHandler::~ConditionsHandler()
{}


void ConditionsHandler::AddLoading(const std::string load_curve_name, const bool &x, const bool &y, const bool &z, const float dx, const float dy, const float dz, const std::string &load_boundary_name)
{
    // Convert the boundary name to low case.
    std::string low_case_boundary_name = load_boundary_name;
    std::transform(low_case_boundary_name.begin(), low_case_boundary_name.end(), low_case_boundary_name.begin(), ::tolower);
	bool boNodeSetFound = false;
	bool boNodeSetNotEmpty = false;

	// Find node set
	for (uint32_t i = 0; i < pNode_sets_->size(); i++) 
	{
		const NodeSet &nset = pNode_sets_->at(i);
		if (low_case_boundary_name == nset.Name()) 
		{
			boNodeSetFound = true;
			// Assign the nset's indices to the loading condition's nodes indices.
			for (uint32_t n = 0; n < nset.NodeIds().size(); n++)
			{
				boNodeSetNotEmpty = true;
				int32_t i32NodeId = nset.NodeIds()[n];
				int32_t i32LoadCurveIdx = LoadCurves.i32GetLoadCurveIndex(load_curve_name);
				if (x)
				{
					pu8DOFsDisplaced_[i32NodeId * 3] = 1;
					pfDOFsDisplacements_[i32NodeId * 3] = dx;
					pi32DOFsLoadCurves_[i32NodeId * 3] = i32LoadCurveIdx;
				}
				if (y)
				{
					pu8DOFsDisplaced_[i32NodeId * 3 + 1] = 1;
					pfDOFsDisplacements_[i32NodeId * 3 + 1] = dy;
					pi32DOFsLoadCurves_[i32NodeId * 3 + 1] = i32LoadCurveIdx;
				}
				if (z)
				{
					pu8DOFsDisplaced_[i32NodeId * 3 + 2] = 1;
					pfDOFsDisplacements_[i32NodeId * 3 + 2] = dz;
					pi32DOFsLoadCurves_[i32NodeId * 3 + 2] = i32LoadCurveIdx;
				}
			}
		}
	}

	if (boNodeSetNotEmpty)
	{
		if (x && (abs(dx) > fMaxAbsDisplacement)) fMaxAbsDisplacement = abs(dx);
		if (y && (abs(dy) > fMaxAbsDisplacement)) fMaxAbsDisplacement = abs(dy);
		if (z && (abs(dz) > fMaxAbsDisplacement)) fMaxAbsDisplacement = abs(dz);
	}
	else if (!boNodeSetFound)
	{
		std::string error = "[ExplicitSim ERROR] Node set " + load_boundary_name + " not found.";
		throw std::runtime_error(error.c_str());
	}
}

void ConditionsHandler::vReadLoadingFromFile(const std::string load_curve_name, std::string file_name, std::vector<std::pair<int32_t, int32_t> > &offsetted_nodes, const uint32_t *pu32ReindexingArray)
{
	boost::filesystem::ifstream file(file_name);

	std::string low_case_boundary_name = file_name;
	std::transform(low_case_boundary_name.begin(), low_case_boundary_name.end(), low_case_boundary_name.begin(), ::tolower);

	while (1)
	{
		std::string line;
		std::string load_name;
		std::getline(file, line);
		if (!file.good())
			break;

		std::stringstream iss(line);
		uint32_t uiNodeNumber;
		bool boX, boY, boZ;
		float dx, dy,dz;
		char delim;

		try
		{
			iss >> uiNodeNumber;
			iss >> delim;
			iss >> boX;
			iss >> delim;
			iss >> boY;
			iss >> delim;
			iss >> boZ;
			iss >> delim;

			iss >> dx;
			iss >> delim;
			iss >> dy;
			iss >> delim;
			iss >> dz;
			iss >> delim;
		}
		catch (...)
		{
			file.close();
		}
		// Check for offset at node.
		auto offsetted_node = std::find_if(offsetted_nodes.begin(), offsetted_nodes.end(),
			[&](const std::pair<int32_t, int32_t> &element) { return element.first == uiNodeNumber; });

		// Apply offset correction if necessary.
		if (offsetted_node == offsetted_nodes.end()) 
		{
			std::string error = "[ExplicitSim ERROR] Node number not found in the list of nodes.";
			throw std::runtime_error(error.c_str());
		}

		int32_t i32NodeId = pu32ReindexingArray[offsetted_node->second];
		int32_t i32LoadCurveIdx = LoadCurves.i32GetLoadCurveIndex(load_curve_name);
		if (boX)
		{
			pu8DOFsDisplaced_[i32NodeId * 3] = 1;
			pfDOFsDisplacements_[i32NodeId * 3] = dx;
			pi32DOFsLoadCurves_[i32NodeId * 3] = i32LoadCurveIdx;
		}
		if (boY)
		{
			pu8DOFsDisplaced_[i32NodeId * 3 + 1] = 1;
			pfDOFsDisplacements_[i32NodeId * 3 + 1] = dy;
			pi32DOFsLoadCurves_[i32NodeId * 3 + 1] = i32LoadCurveIdx;
		}
		if (boZ)
		{
			pu8DOFsDisplaced_[i32NodeId * 3 + 2] = 1;
			pfDOFsDisplacements_[i32NodeId * 3 + 2] = dz;
			pi32DOFsLoadCurves_[i32NodeId * 3 + 2] = i32LoadCurveIdx;
		}

		if (boX && (abs(dx) > fMaxAbsDisplacement)) fMaxAbsDisplacement = abs(dx);
		if (boY && (abs(dy) > fMaxAbsDisplacement)) fMaxAbsDisplacement = abs(dy);
		if (boZ && (abs(dz) > fMaxAbsDisplacement)) fMaxAbsDisplacement = abs(dz);
	}
	file.close();
}
void ConditionsHandler::AddGravity(const float g_x, const float g_y, const float g_z)
{
	pfGravity_[0] = g_x;
	pfGravity_[1] = g_y;
	pfGravity_[2] = g_z;
}

void ConditionsHandler::ApplyGravity()
{
	std::vector<float> gravity_vec = {0., 0., 0.};
	gravity_vec[0] = pfGravity_[0] * 9.81;
	gravity_vec[1] = pfGravity_[1] * 9.81;
	gravity_vec[2] = pfGravity_[2] * 9.81;
	//std::cout<<"gravity_vec[2] is "<< gravity_vec[2] <<std::endl;
}

void ConditionsHandler::AddExternalForce(const std::string load_curve_name, const bool &x, const bool &y, const bool &z,
											const float fx, const float fy, const float fz, const std::string &external_force_boundary_name)
{
	// Convert the boundary name to low case.
	std::string low_case_boundary_name = external_force_boundary_name;
	std::transform(low_case_boundary_name.begin(), low_case_boundary_name.end(), low_case_boundary_name.begin(), ::tolower);
	bool boNodeSetFound = false;
	bool boNodeSetNotEmpty = false;

	//Find node set
	for (uint32_t i = 0; i < pNode_sets_->size(); i++)
	{
		const NodeSet &nset = pNode_sets_->at(i);
		if (low_case_boundary_name == nset.Name())
		{
			boNodeSetFound = true;
			// Assign the nset's indices to the loading condition's nodes indices.
			for (uint32_t n = 0; n < nset.NodeIds().size(); n++)
			{
				boNodeSetNotEmpty = true;
				int32_t i32NodeId = nset.NodeIds()[n];
				int32_t i32LoadCurveIdx = LoadCurves.i32GetLoadCurveIndex(load_curve_name);
				if(x)
				{
					pu8DOFsExternal_[i32NodeId * 3] = 1;
					pfDOFsExternalForces_[i32NodeId * 3] = fx;
					pi32DOFsLoadCurves_[i32NodeId * 3] = i32LoadCurveIdx;
				}
				if(y)
				{
					pu8DOFsExternal_[i32NodeId * 3 + 1] = 1;
					pfDOFsExternalForces_[i32NodeId * 3 + 1] = fy;
					pi32DOFsLoadCurves_[i32NodeId * 3 + 1] = i32LoadCurveIdx;
				}
				if(z)
				{
					pu8DOFsExternal_[i32NodeId * 3 + 2] = 1;
					pfDOFsExternalForces_[i32NodeId * 3 + 2] = fz;
					pi32DOFsLoadCurves_[i32NodeId * 3 + 2] = i32LoadCurveIdx;
				}
			}/*end of for*/
		}/*end of if*/
	}/*end of for (uint32_t i = 0....)*/
}/*end of AddExternalForce*/


// Read External force from file
void ConditionsHandler::vReadExternalLoadingFromFile(const std::string load_curve_name, std::string file_name, std::vector<std::pair<int32_t, int32_t>> &external_loaded_nodes,const uint32_t *pu32ReindexingArray)
{
	boost::filesystem::ifstream file(file_name);

	std::string low_case_boundary_name = file_name;
	std::transform(low_case_boundary_name.begin(),low_case_boundary_name.end(), low_case_boundary_name.begin(), ::tolower);

	while (1)
	{
		std::string line;
		std::string load_name;
		std::getline(file, line);
		if(!file.good())
			break;

		std::stringstream iss(line);
		uint32_t uiNodeNumber;
		bool boX, boY, boZ;
		float fx, fy, fz;
		char delim;

		try
		{
			iss >> uiNodeNumber;
			iss >> delim;
			iss >> boX;
			iss >> delim;
			iss >> boY;
			iss >> delim;
			iss >> boZ;
			iss >> delim;

			iss >> fx;
			iss >> delim;
			iss >> fy;
			iss >> delim;
			iss >> fz;
			iss >> delim;
		}
		catch (...)
		{
			file.close();
		}
		//check for external force node.
		auto external_loaded_node = std::find_if(external_loaded_nodes.begin(), external_loaded_nodes.end(),
			[&](const std::pair<int32_t, int32_t> &element) {return element.first == uiNodeNumber; });

		// Apply offset correction if necessary.
		if (external_loaded_node == external_loaded_nodes.end())
		{
			std::string error = "[ExplicitSim ERROR] Node number not found in the list of nodes.";
			throw std::runtime_error(error.c_str());
		}

		int32_t i32NodeId = pu32ReindexingArray[external_loaded_node->second];
		int32_t i32LoadCurveIdx = LoadCurves.i32GetLoadCurveIndex(load_curve_name);
		if (boX)
		{
			pu8DOFsExternal_[i32NodeId * 3] = 1;
			pfDOFsExternalForces_[i32NodeId * 3] = fx;
			pi32DOFsLoadCurves_[i32NodeId * 3] = i32LoadCurveIdx;
		}
		if (boY)
		{
			pu8DOFsExternal_[i32NodeId * 3 + 1] = 1;
			pfDOFsExternalForces_[i32NodeId * 3 +1] = fy;
			pi32DOFsLoadCurves_[i32NodeId * 3 + 1] = i32LoadCurveIdx;
		}
		if (boZ)
		{
			pu8DOFsExternal_[i32NodeId * 3 + 2] = 1;
			pfDOFsExternalForces_[i32NodeId * 3 + 2] = fz;
			pi32DOFsLoadCurves_[i32NodeId * 3 + 2] = i32LoadCurveIdx;
		}

	}
	file.close();
}

void ConditionsHandler::ExtractExternalForcesIndices(void)
{
	u32NumExternalForcesDOFs = 0;
	for (uint32_t i = 0; i < 3 * u32NumActiveNodes_; i++)
	{
		if (pu8DOFsExternal_[i]) u32NumExternalForcesDOFs++;
	}

	pu32DOFsExternalForcesIdx_ = boost::make_shared<uint32_t[]>(u32NumExternalForcesDOFs);
	uint32_t idx = 0;
	for (uint32_t i = 0; i < 3 * u32NumActiveNodes_; i++)
	{
		if(pu8DOFsExternal_[i])
		{
			pu32DOFsExternalForcesIdx_[idx] = i;
			idx++;
		}
	}
}
void ConditionsHandler::ApplyExternalForce(const double relTime, Eigen::MatrixXd &ExternalForces)
{
	LoadCurves.vComputeValues(relTime);
	double d;
	for (uint32_t i = 0; i < u32NumExternalForcesDOFs; i++)
	{
		uint32_t idxDOF = pu32DOFsExternalForcesIdx_[i];
		ExternalForces.coeffRef(idxDOF/3, idxDOF - 3*(idxDOF/3)) = LoadCurves.dValue(pi32DOFsLoadCurves_[idxDOF]) * pfDOFsExternalForces_[idxDOF];


	}
}/*end of apply external force*/

void ConditionsHandler::AddDirichlet(const bool &x, const bool &y, const bool &z, const std::string &dirichlet_boundary_name)
{
    // Convert the boundary name to low case.
    std::string low_case_boundary_name = dirichlet_boundary_name;
    std::transform(low_case_boundary_name.begin(), low_case_boundary_name.end(), low_case_boundary_name.begin(), ::tolower);

	bool boNodeSetFound = false;

	// Find node set
	for (uint32_t i = 0; i < pNode_sets_->size(); i++)
	{
		const NodeSet &nset = pNode_sets_->at(i);
		if (low_case_boundary_name == nset.Name())
		{
			boNodeSetFound = true;
			// Assign the nset's indices to the loading condition's nodes indices.
			for (uint32_t n = 0; n < nset.NodeIds().size(); n++)
			{
				int32_t i32NodeId = nset.NodeIds()[n];
				if (x)
				{
					pu8DOFsFixed_[i32NodeId * 3] = 1;
				}
				if (y)
				{
					pu8DOFsFixed_[i32NodeId * 3 + 1] = 1;
				}
				if (z)
				{
					pu8DOFsFixed_[i32NodeId * 3 + 2] = 1;
				}
			}
		}
	}

	if (!boNodeSetFound)
	{
		std::string error = "[ExplicitSim ERROR] Node set " + dirichlet_boundary_name + " not found.";
		throw std::runtime_error(error.c_str());
	}
}

void ConditionsHandler::vAddContactPair(const ExplicitSim::Surface &surface, const std::string &contact_node_set, const ExplicitSim::TetraMesh &mesh)
{
	pdNodalCoordinates_ = (double *)mesh.pdGetNodalCoordinates();
	u32NumActiveNodes_ = mesh.NodesNum();
	//newly defined the max displacement
	//float fMaxAbsDisplacement = 0.015;

	// Convert the boundary name to low case.
	std::string low_case_contact_node_set = contact_node_set;
	std::transform(low_case_contact_node_set.begin(), low_case_contact_node_set.end(), low_case_contact_node_set.begin(), ::tolower);

	ExplicitSim::Surface *pSurface = (ExplicitSim::Surface *)&surface;
	if (pSurface->ulNumTriangles() == 0)
	{
		std::cout << Logger::Warning("No triangular surface found! Remove the contact from the input file if it is not used, otherwise check the mesh file.\n");
	}

	const std::vector<int32_t> *pNodeIds = NULL;

	for (auto &nset : mesh.NodeSets())
	{
		// Check if nset's name complies with the contact condition's boundary name.
		if (low_case_contact_node_set == nset.Name()) {
			// Assign the nset's indices to the loading condition's nodes indices.
			pNodeIds = &nset.NodeIds();
			if (pNodeIds->size() > 0)
			{
				ContactPair contact_pair((ExplicitSim::Surface &)surface, nset.NodeIds(), pdNodalCoordinates_, fMaxAbsDisplacement);
				apCContactPairs_.push_back(contact_pair);
				for (uint32_t n = 0; n < nset.NodeIds().size(); n++)
				{
					int32_t i32NodeId = nset.NodeIds()[n];
					pu8NodesContacts_[i32NodeId] = 1;
				}
			}
			else
			{
				std::cout << Logger::Warning("Contact node set [") << contact_node_set
					<< "] is empty. Remove the contact if it is not used, otherwise check the mesh file.\n";
			}
			return;
		}
	}
	std::cout << Logger::Warning("Contact node set [") << contact_node_set
			 << "] not found. Remove the contact if it is not used, otherwise check the mesh file.\n";
}

void ConditionsHandler::vApplyContacts(Eigen::MatrixXd &displacements, bool boInitialDisplacements, const Mmls3d &nodal_approximant)
{
	if (apCContactPairs_.size() == 0) return;
	//std::cout << "fMaxAbsDisplacement is: "<< fMaxAbsDisplacement << std::endl;

	double dMaxExpectedPenetration = 0;
	if (!boInitialDisplacements)
	{
		assert(pdPreviousNodalPositions_ != NULL);
	//std::cout << "assert pdPreviousNodalPositions_ ! = NULL is done "<<  std::endl;
	}
	// compute nodal positions
	for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
	{
		double dx = 0;
		double dy = 0;
		double dz = 0;
		// Add the values of the displaced nodal positions * the shape function value for the neighbor nodes of the nodal node.
		for (Eigen::SparseMatrix<double>::InnerIterator sf(nodal_approximant.ShapeFunction(), i); sf; ++sf) {
			dx += displacements.coeff(sf.row(), 0)*sf.value();
			dy += displacements.coeff(sf.row(), 1)*sf.value();
			dz += displacements.coeff(sf.row(), 2)*sf.value();
		}

		pdCurrentNodalPositions_[3 * i] = pdNodalCoordinates_[3 * i] + dx;
		pdCurrentNodalPositions_[3 * i + 1] = pdNodalCoordinates_[3 * i + 1] + dy;
		pdCurrentNodalPositions_[3 * i + 2] = pdNodalCoordinates_[3 * i + 2] + dz;
	}
	if (!boInitialDisplacements)
	{
		// compute maximum possible penetration
		for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
		{
			if (pu8NodesContacts_[i])
			{
				double dx = abs(pdPreviousNodalPositions_[3 * i] - pdCurrentNodalPositions_[3 * i]);
				if (dx > dMaxExpectedPenetration) dMaxExpectedPenetration = dx;
				dx = abs(pdPreviousNodalPositions_[3 * i + 1] - pdCurrentNodalPositions_[3 * i + 1]);
				if (dx > dMaxExpectedPenetration) dMaxExpectedPenetration = dx;
				dx = abs(pdPreviousNodalPositions_[3 * i + 2] - pdCurrentNodalPositions_[3 * i + 2]);
				if (dx > dMaxExpectedPenetration) dMaxExpectedPenetration = dx;
				//std::cout << "dMaxExpectedPenetration is:  "<<  dMaxExpectedPenetration << std::endl;
			}
		}
	}
	else
	{
		dMaxExpectedPenetration = fMaxAbsDisplacement;
		//dMaxExpectedPenetration = 0.01;
	}

	for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
	{
		pu8NodesContactsActive_[i] = 0;
	}

	// apply contacts
	for (uint32_t i = 0; i < apCContactPairs_.size(); i++)
	{
		apCContactPairs_[i].vApplyContacts(pdCurrentNodalPositions_.get(), dMaxExpectedPenetration, pu8NodesContactsActive_.get());
	}

	if (!boInitialDisplacements)
	{
		for (uint32_t i = 0; i < 3 * u32NumActiveNodes_; i++)
		{
			pdPreviousNodalPositions_[i] = pdCurrentNodalPositions_[i];
		}
	}

	// update displacement matrix
	for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
	{
		displacements(i, 0) = pdCurrentNodalPositions_[3 * i] - pdNodalCoordinates_[3 * i];
		displacements(i, 1) = pdCurrentNodalPositions_[3 * i + 1] - pdNodalCoordinates_[3 * i + 1];
		displacements(i, 2) = pdCurrentNodalPositions_[3 * i + 2] - pdNodalCoordinates_[3 * i + 2];
	}
}

void ConditionsHandler::vComputeSimplifiedVmat(Eigen::SparseMatrix<double> &VmatX, Eigen::SparseMatrix<double> &VmatY, Eigen::SparseMatrix<double> &VmatZ, const Mmls3d &nodal_mmls)
{
	uint32_t neX = 0;
	uint32_t neY = 0;
	uint32_t neZ = 0;
	for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
	{
		if (pu8DOFsDisplaced_[3 * i])
		{
			VmatX.col(neX) = nodal_mmls.ShapeFunction().col(i);
			neX++;
		}
		if (pu8DOFsDisplaced_[3 * i + 1])
		{
			VmatY.col(neY) = nodal_mmls.ShapeFunction().col(i);
			neY++;
		}
		if (pu8DOFsDisplaced_[3 * i + 2])
		{
			VmatZ.col(neZ) = nodal_mmls.ShapeFunction().col(i);
			neZ++;
		}
	}
}

void ConditionsHandler::vComputeVmat(Eigen::SparseMatrix<double> &VmatX, Eigen::SparseMatrix<double> &VmatY, Eigen::SparseMatrix<double> &VmatZ, const WeakModel3D &model, const Mmls3d &nodal_mmls)
{
	vComputeSimplifiedVmat(VmatX, VmatY, VmatZ, nodal_mmls);  // currently not implemented; fallback to simplified version
}

void ConditionsHandler::vComputeCorrectionMatrix(const WeakModel3D &model, const Mmls3d &nodal_mmls, bool boUseEBCIEM, bool boSimplified)
{
	if ((boUseEBCIEM == false) || (u32NumEssentialDOFs == 0))
	{
		boUseEBCIEM_ = false;
		return;
	}
	boUseEBCIEM_ = true;
	Eigen::SparseMatrix<double> VmatX(u32NumActiveNodes_, u32NumEssentialDOFsX ? u32NumEssentialDOFsX : 1);
	Eigen::SparseMatrix<double> VmatY(u32NumActiveNodes_, u32NumEssentialDOFsY ? u32NumEssentialDOFsY : 1);
	Eigen::SparseMatrix<double> VmatZ(u32NumActiveNodes_, u32NumEssentialDOFsZ ? u32NumEssentialDOFsZ : 1);
	VmatX.setZero();
	VmatY.setZero();
	VmatZ.setZero();
	if (boSimplified)
	{
		vComputeSimplifiedVmat(VmatX, VmatY, VmatZ, nodal_mmls);
	}
	else
	{
		vComputeVmat(VmatX, VmatY, VmatZ, model, nodal_mmls);
	}
	// Initialize inverse mass matrix triplets
	std::vector<Eigen::Triplet<double> > inv_mass_trip;
	inv_mass_trip.reserve(u32NumActiveNodes_);

	// Get inverse matrix values for each node.
	for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
	{
		inv_mass_trip.emplace_back(Eigen::Triplet<double>(i, i, 1. / model.Mass()[i]));
	}
	// Create inverse mass matrix.
	Eigen::SparseMatrix<double> Minv(u32NumActiveNodes_, u32NumActiveNodes_);
	Minv.setFromTriplets(inv_mass_trip.begin(), inv_mass_trip.end());

	// Mass matrix
	inv_mass_trip.clear();
	for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
	{
		inv_mass_trip.emplace_back(Eigen::Triplet<double>(i, i, model.Mass()[i]));
	}
	MassMatrix.resize(u32NumActiveNodes_, u32NumActiveNodes_);
	MassMatrix.setFromTriplets(inv_mass_trip.begin(), inv_mass_trip.end());

	// Create Fi matrices
	if (u32NumEssentialDOFsX)
	{
		FImatX = Eigen::SparseMatrix<double>(u32NumEssentialDOFsX, u32NumActiveNodes_);
	}
	if (u32NumEssentialDOFsY)
	{
		FImatY = Eigen::SparseMatrix<double>(u32NumEssentialDOFsY, u32NumActiveNodes_);
	}
	if (u32NumEssentialDOFsZ)
	{
		FImatZ = Eigen::SparseMatrix<double>(u32NumEssentialDOFsZ, u32NumActiveNodes_);
	}
	uint32_t neX = 0;
	uint32_t neY = 0;
	uint32_t neZ = 0;
	for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
	{
		if (u32NumEssentialDOFsX && pu8DOFsDisplaced_[3 * i])
		{
			FImatX.row(neX) = nodal_mmls.ShapeFunction().col(i);
			neX++;
		}
		if (u32NumEssentialDOFsY && pu8DOFsDisplaced_[3 * i + 1])
		{
			FImatY.row(neY) = nodal_mmls.ShapeFunction().col(i);
			neY++;
		}
		if (u32NumEssentialDOFsZ && pu8DOFsDisplaced_[3 * i + 2])
		{
			FImatZ.row(neZ) = nodal_mmls.ShapeFunction().col(i);
			neZ++;
		}
	}
	
	if (u32NumEssentialDOFsX)
	{
		// multiply Vmat with Minv
		VmatX = Minv * VmatX;
		// Compute decomposition of the FI*inv_mass*Vmat product.
		Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > decomp_prodX;
		decomp_prodX.compute(FImatX*VmatX);
		Eigen::SparseMatrix<double> identityX(u32NumEssentialDOFsX, u32NumEssentialDOFsX);
		identityX.setIdentity();
		// Compute the inverse matrix of the product.
		Eigen::SparseMatrix<double> inv_prodX = decomp_prodX.solve(identityX);
		// Compute the correction matrix.
		correctionMatrixX = VmatX * inv_prodX;
	}

	if (u32NumEssentialDOFsY)
	{
		VmatY = Minv * VmatY;
		Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > decomp_prodY;
		decomp_prodY.compute(FImatY*VmatY);
		Eigen::SparseMatrix<double> identityY(u32NumEssentialDOFsY, u32NumEssentialDOFsY);
		identityY.setIdentity();
		Eigen::SparseMatrix<double> inv_prodY = decomp_prodY.solve(identityY);
		correctionMatrixY = VmatY * inv_prodY;
	}

	if (u32NumEssentialDOFsZ)
	{
		VmatZ = Minv * VmatZ;
		Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > decomp_prodZ;
		decomp_prodZ.compute(FImatZ*VmatZ);
		Eigen::SparseMatrix<double> identityZ(u32NumEssentialDOFsZ, u32NumEssentialDOFsZ);
		identityZ.setIdentity();
		Eigen::SparseMatrix<double> inv_prodZ = decomp_prodZ.solve(identityZ);
		correctionMatrixZ = VmatZ * inv_prodZ;
	}
}

void ConditionsHandler::vExtractConstrainedDOFs(void)
{
	// extract displaced indexes
	u32NumDisplacedDOFs = 0;
	for (uint32_t i = 0; i < 3 * u32NumActiveNodes_; i++)
	{
		if (pu8DOFsDisplaced_[i]) u32NumDisplacedDOFs++;
	}
	pu32DOFsDisplacedIdx_ = boost::make_shared<uint32_t[]>(u32NumDisplacedDOFs);
	uint32_t idx = 0;
	for (uint32_t i = 0; i < 3 * u32NumActiveNodes_; i++)
	{
		if (pu8DOFsDisplaced_[i])
		{
			pu32DOFsDisplacedIdx_[idx] = i;
			idx++;
		}
	}

	// extract fixed indexes
	u32NumFixedDOFs = 0;
	for (uint32_t i = 0; i < 3 * u32NumActiveNodes_; i++)
	{
		if (pu8DOFsFixed_[i])
		{
			u32NumFixedDOFs++;
			pu8DOFsDisplaced_[i] = 1;
		}
	}
	pu32DOFsFixedIdx_ = boost::make_shared<uint32_t[]>(u32NumFixedDOFs);
	idx = 0;
	for (uint32_t i = 0; i < 3 * u32NumActiveNodes_; i++)
	{
		if (pu8DOFsFixed_[i])
		{
			pu32DOFsFixedIdx_[idx] = i;
			idx++;
		}
	}

	// mark all constraied DOFs (in pu8DOFsDisplaced_)
	for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
	{
		if (pu8NodesContacts_[i])
		{
			pu8DOFsDisplaced_[3 * i] = 1;
			pu8DOFsDisplaced_[3 * i + 1] = 1;
			pu8DOFsDisplaced_[3 * i + 2] = 1;
		}
	}

	u32NumEssentialDOFs = 0;
	u32NumEssentialDOFsX = 0;
	u32NumEssentialDOFsY = 0;
	u32NumEssentialDOFsZ = 0;
	for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
	{
		if (pu8DOFsDisplaced_[3*i])
		{
			u32NumEssentialDOFsX++;
		}
		if (pu8DOFsDisplaced_[3 * i + 1])
		{
			u32NumEssentialDOFsY++;
		}
		if (pu8DOFsDisplaced_[3 * i + 2])
		{
			u32NumEssentialDOFsZ++;
		}
	}
	u32NumEssentialDOFs = u32NumEssentialDOFsX + u32NumEssentialDOFsY + u32NumEssentialDOFsZ;

}

void ConditionsHandler::ApplyBoundaryConditions(const double relTime, Eigen::MatrixXd &displacements, const Mmls3d &nodal_approximant, 
		Eigen::MatrixXd &ExternalForces, double alfa)
{
	Eigen::MatrixXd predicted_displacements;
	if (boUseEBCIEM_)
	{
		predicted_displacements  = displacements;
	}

	// apply fixed BCs
	for (uint32_t i = 0; i < u32NumFixedDOFs; i++)
	{
		uint32_t idxDOF = pu32DOFsFixedIdx_[i];
		displacements.coeffRef(idxDOF / 3, idxDOF - 3 * (idxDOF / 3)) = 0;
	}

	// apply displacements
	//LoadCurves.vComputeValues(relTime);
	//double d;
	//for (uint32_t i = 0; i < u32NumDisplacedDOFs; i++)
	//{
	//	uint32_t idxDOF = pu32DOFsDisplacedIdx_[i];
	//	displacements.coeffRef(idxDOF/3, idxDOF - 3*(idxDOF/3)) = LoadCurves.dValue(pi32DOFsLoadCurves_[idxDOF])*pfDOFsDisplacements_[idxDOF];
	//	d = displacements(idxDOF / 3, idxDOF - 3 * (idxDOF / 3));
	//}

	// apply contacts
	vApplyContacts(displacements, false, nodal_approximant);

	// enforce essential boundary conditions
	if (boUseEBCIEM_)
	{
		if (u32NumEssentialDOFsX)
		{
			// apply correction to the predicted displacements
			Eigen::MatrixXd essential_nodes_disp = FImatX * predicted_displacements.col(0);
			Eigen::MatrixXd desired_nodes_disp = essential_nodes_disp;
			// set desired nodes displacements for the essential DOFs only
			uint32_t ne = 0;
			for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
			{
				if (pu8DOFsDisplaced_[3 * i])
				{
					desired_nodes_disp.coeffRef(ne, 0) = displacements.coeffRef(i, 0);
					ne++;
				}
			}
			// correct displacements
			ExternalForces.col(0) = correctionMatrixX * (desired_nodes_disp - essential_nodes_disp);
			displacements.col(0) = predicted_displacements.col(0) + ExternalForces.col(0);
		}
		if (u32NumEssentialDOFsY)
		{
			// apply correction to the predicted displacements
			Eigen::MatrixXd essential_nodes_disp = FImatY * predicted_displacements.col(1);
			Eigen::MatrixXd desired_nodes_disp = essential_nodes_disp;
			// set desired nodes displacements for the essential DOFs only
			uint32_t ne = 0;
			for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
			{
				if (pu8DOFsDisplaced_[3 * i + 1])
				{
					desired_nodes_disp.coeffRef(ne, 0) = displacements.coeffRef(i, 1);
					ne++;
				}
			}
			// correct displacements
			ExternalForces.col(1) = correctionMatrixY * (desired_nodes_disp - essential_nodes_disp);
			displacements.col(1) = predicted_displacements.col(1) + ExternalForces.col(1);
		}
		if (u32NumEssentialDOFsZ)
		{
			// apply correction to the predicted displacements
			Eigen::MatrixXd essential_nodes_disp = FImatZ * predicted_displacements.col(2);
			Eigen::MatrixXd desired_nodes_disp = essential_nodes_disp;
			// set desired nodes displacements for the essential DOFs only
			uint32_t ne = 0;
			for (uint32_t i = 0; i < u32NumActiveNodes_; i++)
			{
				if (pu8DOFsDisplaced_[3 * i + 2])
				{
					desired_nodes_disp.coeffRef(ne, 0) = displacements.coeffRef(i, 2);
					ne++;
				}
			}
			// correct displacements
			ExternalForces.col(2) = correctionMatrixZ * (desired_nodes_disp - essential_nodes_disp);
			displacements.col(2) = predicted_displacements.col(2) + ExternalForces.col(2);
		}

		ExternalForces = (-1. / alfa) * this->MassMatrix * ExternalForces;
	}
}


void ConditionsHandler::vZeroConstrainedDOFs(Eigen::MatrixXd &forces) const
{
	for (uint32_t i = 0; i < 3 * u32NumActiveNodes_; i++)
	{
		if (pu8DOFsDisplaced_[i]) forces.coeffRef(i/3, i-3*(i/3)) = 0;
	}
}


} //end of namespace ExplicitSim

 