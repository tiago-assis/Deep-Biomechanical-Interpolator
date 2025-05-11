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

#include <boost/smart_ptr/make_shared_array.hpp>

#include "ExplicitSim/conditions/contact_pair.hpp"


namespace ExplicitSim {


ContactPair::ContactPair(Surface &surface, const std::vector<int32_t> &boundary_node_ids, const double *pdNodeCoordinates, double dMaxPossiblePenetration)
{
	unsigned long ulNumContactNodes = (unsigned long)boundary_node_ids.size();
	pulContactNodesIndexes = boost::make_shared<unsigned long[]>(ulNumContactNodes);
	for (unsigned long i = 0; i < ulNumContactNodes; i++)
	{
		pulContactNodesIndexes[i] = boundary_node_ids[i];
	}

	CContactPair *pContactPair = new CContactPair(surface.ulNumTriangles(), surface.pulConnectivity().get(), (double *)pdNodeCoordinates, ulNumContactNodes, pulContactNodesIndexes.get(), dMaxPossiblePenetration);
	pCContactPair_ = boost::shared_ptr<CContactPair>(pContactPair);
}

void ContactPair::vApplyContacts(double *pdCurrentNodalPositions, double dMaxPossiblePenetration, uint8 *pu8NodeInContact)
{
	pCContactPair_->vSetMaxPenetration(dMaxPossiblePenetration);
	pCContactPair_->vApplyContacts(pdCurrentNodalPositions, pu8NodeInContact);
}

ContactPair::~ContactPair()
{}

} //end of namespace ExplicitSim
