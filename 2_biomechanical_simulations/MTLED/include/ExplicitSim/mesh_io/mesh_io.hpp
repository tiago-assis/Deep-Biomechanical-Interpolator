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
   \file mesh_io.hpp
   \brief Header file collecting the header files of the various mesh input/output classes.
   \author Konstantinos A. Mountris
   \date 16/11/2017
*/

#ifndef EXPLICITSIM_MESH_IO_MESH_IO_HPP_
#define EXPLICITSIM_MESH_IO_MESH_IO_HPP_

#include "ExplicitSim/vectors/vectors.hpp"
#include "ExplicitSim/elements/elements.hpp"

#include <vector>

namespace ExplicitSim {

	/*!
	*  \addtogroup MeshIO
	*  @{
	*/

	/*!
	* \class MeshIO
	* \brief Interface for mesh input/output functionality.
	*
	*/
	class MeshIO {
	public:
		/*!
		* \brief MeshIO constructor.
		*/
		MeshIO() {};


		/*!
		* \brief MeshIO destructor.
		*/
		~MeshIO() {};


		/*!
		* \brief Loads a mesh from a file.
		*
		* \param [in] mesh_filename The filename (full path) of the mesh to be loaded.
		* \return [void]
		*/
		virtual void LoadMeshFrom(const std::string &mesh_filename) = 0;


		/*!
		* \brief Load the vertices of the readed mesh in the given vertices container.
		* \param [out] vertices The vertices container to load the vertices of the mesh.
		* \return [void] The vertices of the readed mesh.
		*/
		virtual void LoadNodesIn(std::vector<Node> &nodes) = 0;


		/*!
		* \brief Load the elements of the readed mesh in the given elements container.
		* \param [out] tetras The tetrahedra container to load the tetrahedra of the mesh.
		* \return [void]
		*/
		virtual void LoadElementsIn(std::vector<Tetrahedron> &tetras) = 0;

		/*!
		* \brief Access to the offsetted nodes information.
		*/
		std::vector<std::pair<int32_t, int32_t> > &pvGetOffsettedNodes(void) { return offsetted_nodes_; };


	protected:
		std::vector<std::pair<int32_t, int32_t> > offsetted_nodes_;       /*!< The indices of the offsetted nodes from the storage order and their offsets. */

		std::vector<std::pair<int32_t, int32_t> > offsetted_elems_;       /*!< The indices of the offsetted elements from the storage order and their offsets. */

	};


	/*! @} End of Doxygen Groups*/

} //end of namespace ExplicitSim

#endif //EXPLICITSIM_MESH_IO_MESH_IO_HPP_
