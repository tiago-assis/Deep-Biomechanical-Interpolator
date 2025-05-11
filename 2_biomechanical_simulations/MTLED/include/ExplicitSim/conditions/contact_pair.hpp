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



#ifndef EXPLICITSIM_CONDITIONS_CONTACT_PAIR_HPP_
#define EXPLICITSIM_CONDITIONS_CONTACT_PAIR_HPP_

/*!
   \file contact_pair.hpp
   \brief Contact pair class header file.
   \author Konstantinos A. Mountris
   \date 30/09/2017
*/

#include <vector>
#include <boost/smart_ptr.hpp>

#include "ExplicitSim/elements/surface.hpp"

#include "contacts/ContactPair.hpp" 

namespace ExplicitSim {

/*!
 *  \addtogroup Conditions
 *  @{
 */


/*!
 * \class ContactPair
 * \brief Class implemmenting the contact boundary condition.
 *
 */

class ContactPair {
public:


    /*!
     * \brief The ContactPair initialization constructor.
     */
	ContactPair(Surface &surface, const std::vector<int32_t> &boundary_node_ids, const double *pdNodeCoordinates, double dMaxPossiblePenetration);


    /*!
     * \brief The ContactPair destructor.
     */
    virtual ~ContactPair();

	void vApplyContacts(double *pdCurrentNodalPositions, double dMaxPossiblePenetration, uint8 *pu8NodeInContact);
    
private:

	boost::shared_ptr<unsigned long[]> pulContactNodesIndexes;

	boost::shared_ptr<CContactPair> pCContactPair_; /*!< The contact pair handler. */

};




/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif //EXPLICITSIM_CONDITIONS_CONTACT_PAIR_HPP_

