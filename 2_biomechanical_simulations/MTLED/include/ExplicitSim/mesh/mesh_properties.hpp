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
   \file mesh_properties.hpp
   \brief Collection of mesh properties header file.
   \author Konstantinos A. Mountris
   \date 16/11/2017
*/

#ifndef EXPLICITSIM_MESH_MESH_PROPERTIES_HPP_
#define EXPLICITSIM_MESH_MESH_PROPERTIES_HPP_


namespace ExplicitSim {

/*!
 *  \addtogroup Mesh
 *  @{
 */


/*!
 * \enum
 */
enum struct MeshType: int {triangular = 1,          /*!< Mesh type -> triangular. */
                           quadrilateral = 2,       /*!< Mesh type -> quadrilateral. */
                           tetrahedral = 3,         /*!< Mesh type -> tetrahedral. */
                           hexahedral = 4,          /*!< Mesh type -> hexahedral. */
                          };



/*! @} End of Doxygen Groups*/

} //end of namespace ExplicitSim

#endif //EXPLICITSIM_MESH_MESH_PROPERTIES_HPP_
