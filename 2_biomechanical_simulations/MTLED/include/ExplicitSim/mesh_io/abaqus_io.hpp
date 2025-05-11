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
   \file abaqus_io.hpp
   \brief Abaqus input/output class header file.
   \author Konstantinos A. Mountris
   \date 16/11/2017
*/

#ifndef EXPLICITSIM_MESH_IO_ABAQUS_IO_HPP_
#define EXPLICITSIM_MESH_IO_ABAQUS_IO_HPP_

#include "ExplicitSim/mesh_io/mesh_io.hpp"

#include "ExplicitSim/sets/node_set.hpp"
#include "ExplicitSim/elements/surface.hpp"
#include "ExplicitSim/mesh/mesh_properties.hpp"

#include <vector>
#include <map>

#include <utility>
#include <cctype>

#include <exception>
#include <stdexcept>

#include <sstream>
#include <iostream>
#include <fstream>

#include <algorithm>
#include <string>

namespace ExplicitSim {

/*!
 *  \addtogroup MeshIO
 *  @{
 */

/*!
 * \class AbaqusIO
 * \brief Class implemmenting input/output functionality for mesh in Abaqus format (.inp).
 *
 */
class AbaqusIO:public MeshIO {
public:
    /*!
     * \brief AbaqusIO constructor.
     */
    AbaqusIO();


    /*!
     * \brief AbaqusIO destructor.
     */
    virtual ~AbaqusIO();


    /*!
     * \brief Load an abaqus mesh.
     *
     * The mesh to be loaded should be in abaqus format (.inp).
     *
     * \param [in] mesh_filename The filename (full path) of the mesh to be loaded.
     * \return [void]
     */
    void LoadMeshFrom(const std::string &mesh_filename);


    /*!
     * \brief Save mesh in abaqus format.
     *
     * Templated function to save a mesh in abaqus format.
     * The template type must be one of the available mesh types: triangular,
     * quadrangular, tetrahedral, hexahedral.
     *
     * \param [in] mesh The mesh to be saved.
     * \param [in] mesh_filename The filename (full path) where the mesh will be saved.
     * \return [void]
     */
    template <class MESHTYPE, class ELEMTYPE>
    inline void SaveMesh(const MESHTYPE &mesh, const std::string &mesh_filename);


    /*!
     * \brief Load the vertices of the readed mesh in the given vertices container.
     * \param [out] vertices The vertices container to load the vertices of the mesh.
     * \return [void] The vertices of the readed mesh.
     */
    void LoadNodesIn(std::vector<Node> &nodes);

	/*!
	* \brief Load the surfaces in the given surfaces container.
	* \param [out] vertices The Surfaces container where to place the Surfaces.
	* \return [void].
	*/
	void LoadSurfacesIn(std::vector<Surface> &surfaces);


    /*!
     * \brief Load the elements of the readed mesh in the given elements container.
     * \param [out] tetras The tetrahedra container to load the tetrahedra of the mesh.
     * \return [void]
     */
    void LoadElementsIn(std::vector<Tetrahedron> &tetras);


    /*!
     * \brief Assign the partitions of the readed mesh to the elements of the given container.
     * \param [out] tetras The tetrahedra container where partitions will be assigned to the stored tetrahedra.
     * \return [void]
     */
    void LoadPartitionsIn(std::vector<Tetrahedron> &tetras);


    /*!
     * \brief Assign the boundary node sets of the loaded mesh to the given container.
     * \param node_sets The container where the loaded mesh nodesets will be stored.
     * \return [void]
     */
    void LoadBoundarySetsIn(std::vector<NodeSet> &node_sets);


    /*!
     * \brief Conditional to check if the mesh has multiple partitions (domains)
     * \return [bool] True if the mesh has more than one partitions (domains), false differently.
     */
    inline const bool & PartitionsExist() const { return this->partitions_exist; }


    inline const bool & NodeSetsExist() const { return this->node_sets_exist; }


private:
    std::vector<std::string> input_mesh_;                    /*!< The parsed mesh with an index per line for easy access. */

    int nodes_startline_;                                /*!< The index of the starting line of the nodes set in the mesh file. */

    int elems_startline_;                                  /*!< The index of the starting line of the elements set in the mesh file. */

	int S3_elems_startline_;                                  /*!< The index of the starting line of the S3 elements in the mesh file. */

    bool partitions_exist;                                   /*!< The conditional of the mesh partitions' existence. */

    bool node_sets_exist;                                    /*!< The conditional of the mesh boundaries' existence */

    std::vector<int> parts_startlines_;                      /*!< The indices of the starting line of the partition sets in the mesh filie. */

    std::vector<int> node_sets_startlines_;                    /*!< The indices of the starting line of the boundary sets in the mesh filie. */
};


/*! @} End of Doxygen Groups*/

} //end of namespace ExplicitSim


#include "ExplicitSim/mesh_io/abaqus_io.tpp"

#endif // EXPLICITSIM_MESH_IO_ABAQUS_IO_HPP_
