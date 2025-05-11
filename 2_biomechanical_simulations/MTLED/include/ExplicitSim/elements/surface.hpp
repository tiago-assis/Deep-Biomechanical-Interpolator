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


/*!
   \file surface.hpp
   \brief Surface class header file.
   \author Grand Joldes
   \date 15/09/2018
*/

#ifndef EXPLICITSIM_ELEMENTS_SURFACE_HPP_
#define EXPLICITSIM_ELEMENTS_SURFACE_HPP_

#include <boost/smart_ptr.hpp>
#include <boost/smart_ptr/make_shared_array.hpp>

namespace ExplicitSim {

/*!
 *  \addtogroup Elements
 *  @{
 */

/*!
 * \class Surface
 * \brief Class implemmenting a triangular surface to be used in contact definition.
 *
 */
class Surface{
public:
    /*!
     * \brief Surface constructor.
     */
	Surface(const unsigned long ulNumTriangles) 
	{
		pulConnectivity_ = boost::make_shared<unsigned long[]>(3*ulNumTriangles);
		ulNumTriangles_ = ulNumTriangles;
	};

     /*!
     * \brief Surface destructor.
     */
	~Surface() 	{};

	
    /*!
     * \brief Access to the connectivity list (nodes indices) of the Surface.
     *
     * \return pointer to the array of node indexes
     */
	inline boost::shared_ptr<unsigned long[]> pulConnectivity(void)
	{
		return pulConnectivity_;
	}

	inline unsigned long ulNumTriangles(void) {
		return ulNumTriangles_;
	}

private:
	unsigned long ulNumTriangles_;               /*!< Number of triangles in the surface. */

	boost::shared_ptr<unsigned long[]> pulConnectivity_; /*!< The list of nodes (3*ulNumTriangles). */
};


/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif // EXPLICITSIM_ELEMENTS_SURFACE_HPP_
