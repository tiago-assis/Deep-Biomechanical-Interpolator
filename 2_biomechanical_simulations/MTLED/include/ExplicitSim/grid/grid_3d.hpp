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
   \file grid_3d.hpp
   \brief Grid3D class header file.
   \author Konstantinos A. Mountris
   \date 16/11/2017
*/

#ifndef EXPLICITSIM_GRID_GRID_3D_HPP_
#define EXPLICITSIM_GRID_GRID_3D_HPP_

#include "ExplicitSim/elements/node.hpp"
#include "ExplicitSim/mesh/tetramesh.hpp"

#include <vector>

namespace ExplicitSim {

/*!
 *  \addtogroup Grid
 *  @{
 */


/*!
 * \class Grid3D
 * \brief Class implemmenting a 3D grid of nodes.
 */
class Grid3D{
public:
    /*!
     * \brief Grid3D constructor.
     */
    Grid3D();

    /*!
     * \brief Grid3D copy constructor.
     * \param [in] grid3D The 3D grid to be copied.
     */
    Grid3D(const Grid3D &grid3D);


    /*!
     * \brief Grid3D destructor.
     */
    virtual ~Grid3D();


    /*!
     * \brief Copy the grid nodes from a tetrahedral mesh.
     * \param tetramesh The tetrahedral mesh to copy the nodes from.
     * \return [void]
     */
    void CopyNodesFromMesh(const TetraMesh &tetramesh);


    /*!
     * \brief Write access to the nodes of the mesh.
     * \return [std::vector<ExplicitSim::Node>] the mesh nodes with write access.
     */
    inline std::vector<Node> & EditNodes() { return this->nodes_; }


    /*!
     * \brief Read-only access to the nodes of the mesh.
     * \return [std::vector<ExplicitSim::Node>] the mesh nodes with read-only access.
     */
    inline const std::vector<Node> & Nodes() const { return this->nodes_; }


    /*!
     * \brief Get the number of the grid's nodes.
     * \return [int] The number of the grid's nodes.
     */
    inline uint32_t NodesNum() const { return ((uint32_t)this->nodes_.size()); }


    /*!
     * \brief Equal to operator.
     *
     * Compares 3D grids for equality.
     *
     * \param [in] grid3D The 3D grid mesh to compare.
     * \return [bool] TRUE if 3D grids are identical.
     */
    bool operator == (const Grid3D &grid3D) const;


    /*!
     * \brief Not equal to operator.
     *
     * Compares 3D grids for inequality.
     *
     * \param [in] grid3D The 3D grids to compare.
     * \return [bool] TRUE if 3D grids are not identical.
     */
    bool operator != (const Grid3D &grid3D) const;


    /*!
     * \brief Assignment operator.
     *
     * Assigns all the properties of a given 3D grid (nodes).
     *
     * \param [in] grid3D The 3D grid to assign.
     * \return [Grid3D] The assigned 3D grid.
     */
    Grid3D & operator = (const Grid3D &grid3D);

private:
    std::vector<Node> nodes_;        /*!< The nodes of the grid. */

};



/*! @} End of Doxygen Groups*/
} // End of namespace ExplicitSim

#endif //EXPLICITSIM_GRID_GRID_3D_HPP_
