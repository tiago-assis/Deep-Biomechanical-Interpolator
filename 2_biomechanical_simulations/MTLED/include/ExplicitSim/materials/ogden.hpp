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
 */


#ifndef EXPLICITSIM_MATERIALS_OGDEN_HPP_
#define EXPLICITSIM_MATERIALS_OGDEN_HPP_

/*!
  \file ogden.hpp
  \brief Ogden class header file.
  \author Benjamin Zwick
  \date 23/02/2019
*/

#include <Eigen/Dense>

#include "ExplicitSim/materials/material.hpp"

namespace ExplicitSim {

/*!
 *  \addtogroup Materials
 *  @{
 */

/*!
 * \class Ogden
 * \brief Ogden material.
 */

class Ogden: public Material {
public:
	/*!
	 * \brief Ogden constructor.
	 */
	Ogden();

	/*!
	 * \brief Ogden destructor.
	 */
	virtual ~Ogden();

	/*!
	 * \brief Set alpha of the material points.
	 * \param [in] alpha_value The alpha value.
	 * \return [void]
	 */
	//void SetAlpha(const double &alpha_value);

	Eigen::Matrix3d SpkStress(const Eigen::Matrix3d &FT, const size_t &integ_point_id) const;

private:
	//std::vector<double> alpha_;             /*!< The alpha value of the material points. */
};

/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif //EXPLICITSIM_MATERIALS_OGDEN_HPP_
