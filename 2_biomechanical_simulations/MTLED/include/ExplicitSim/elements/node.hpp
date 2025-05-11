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
   \file node.hpp
   \brief Node class header file.
   \author Konstantinos A. Mountris
   \date 16/11/2017
*/

#ifndef EXPLICITSIM_ELEMENTS_NODE_HPP_
#define EXPLICITSIM_ELEMENTS_NODE_HPP_

#include "ExplicitSim/vectors/vectors.hpp"
#include "ExplicitSim/elements/element_properties.hpp"

namespace ExplicitSim {

/*!
 *  \addtogroup Elements
 *  @{
 */

/*!
 * \class Node
 * \brief Class implemmenting a node in 3 dimensions for meshfree applications.
 *
 * Node can be used to represent lower-dimension geometries too.
 *
 */
class Node {
public:
    /*!
     * \brief Node class constructor.
     */
    Node();


    /*!
     * \brief Node copy constructor.
     * \param [in] node The node to be copied on construction.
     */
    Node(const Node &node);


    /*!
     * \brief Node class destructor.
     */
    virtual ~Node();


    /*!
     * \brief Set the id of the node in a list of nodes.
     * \param id The id of the node.
     * \return [void]
     */
    void SetId(const int32_t &id);


    /*!
     * \brief Sets the coordinates of the node.
     * \param [in] x The X coordinate.
     * \param [in] y The Y coordinate.
     * \param [in] z The Z coordinate.
     * \return [void]
     */
    void SetCoordinates(const double &x, const double &y, const double &z);


    /*!
     * \overload
     * \brief Sets the coordinates of the node.
     * \param [in] coords The X, Y, Z coordinates.
     * \return [void]
     */
    void SetCoordinates(const Vec3<double> &coords);


    /*!
     * \brief Sets the X coordinate of the node.
     * \param [in] x The X coordinate.
     * \return [void]
     */
    void SetCoordX(const double &x);


    /*!
     * \brief Sets the Y coordinate of the node.
     * \param [in] y The Y coordinate.
     * \return [void]
     */
    void SetCoordY(const double &y);


    /*!
     * \brief Sets the Z coordinate of the node.
     * \param [in] z The Z coordinate.
     * \return [void]
     */
    void SetCoordZ(const double &z);


    /*!
     * \brief Sets the normal of the node.
     * \param [in] n_x The X component of the normal.
     * \param [in] n_y The Y component of the normal.
     * \param [in] n_z The Z component of the normal.
     * \return [void]
     */
    void SetNormal(const double &n_x, const double &n_y, const double &n_z);


    /*!
     * \brief Sets the X component of the node's normal.
     * \param [in] n_x The X component of the normal.
     * \return [void]
     */
    void SetNormalX(const double &n_x);


    /*!
     * \brief Sets the Y component of the node's normal.
     * \param [in] n_y The Y component of the normal.
     * \return [void]
     */
    void SetNormalY(const double &n_y);


    /*!
     * \brief Sets the Z component of the node's normal.
     * \param [in] n_z The Z component of the normal.
     * \return [void]
     */
    void SetNormalZ(const double &n_z);


    /*!
     * \brief Sets the partition the node belongs.
     * \param [in] id The partition's index.
     * \param [in] name The partiotion's name. If omitted, it is considered empty.
     * \return [void]
     */
    void SetPartition(const int &id, std::string name="");


    /*!
     * \brief Sets the boundary the node belongs.
     * \param [in] id The boundary's index. If zero, the node does not belong on boundary.
     * \param [in] name The boundary's name. If omitted, it is considered empty.
     * \return [void]
     */
    void SetBoundary(const int &id, std::string name="");


    /*!
     * \brief Copy coordinates by a Vec3 vector.
     * \param [in] coordinates The Vec3 vector containing the coordinates to be copied.
     * \return [void]
     */
    void CopyCoordinates(const Vec3<double> &coordinates);


    /*!
     * \brief Copy normal by a Vec3 vector.
     * \param [in] normal The Vec3 vector containing the normal to be copied.
     * \return [void]
     */
    void CopyNormal(const Vec3<double> &normal);


    /*!
     * \brief Copy partition from other element object.
     * \param [in] part The partition to be copied.
     * \return [void]
     */
    void CopyPartition(const Partition &part);


    /*!
     * \brief Copy boundary from other element object.
     * \param [in] boundary The boundary to be copied.
     * \return [void]
     */
    void CopyBoundary(const Boundary &boundary);


    /*!
     * \brief Get the Id of the node.
     * \return [int] The node's id, in other words its position in a list of nodes.
     */
    inline const int32_t & Id() const { return this->id_; }


    /*!
     * \brief Get the node's coordinates.
     * \return [Vec3] The node's coordinates.
     */
    inline const Vec3<double> & Coordinates() const { return this->coordinates_; }


    /*!
     * \brief Get the node's normal.
     * \return [Vec3] The node's normal.
     */
    inline const Vec3<double> & Normal() const { return this->normal_; }


    /*!
     * \brief Get the index of the partition the node belongs to.
     * \return [int] The node's partition index.
     */
    inline const int & PartitionId() const { return this->part_.id_; }


    /*!
     * \brief Get the name of the partition the node belongs to.
     * \return [std::string] The node's partition name.
     */
    inline const std::string & PartitionName() const { return this->part_.name_; }


    /*!
     * \brief Get the index of the boundary the node belongs to.
     * \return [int] The node's boundary index.
     */
    inline const int & BoundaryId() const { return this->boundary_.id_; }


    /*!
     * \brief Get the name of the boundary the node belongs to.
     * \return [std::string] The node's boundary name.
     */
    inline const std::string & BoundaryName() const { return this->boundary_.name_; }


    /*!
     * \brief Get the element type of the node.
     * \return [ElementType] The node's element type.
     */
    inline const ExplicitSim::ElementType & ElementType() const { return this->element_type_; }


    /*!
     * \brief Equal to operator.
     *
     * Compares nodes for equality.
     *
     * \param [in] node The node to compare.
     * \return [bool] TRUE if nodes are identical.
     */
    bool operator == (const Node &node) const;


    /*!
     * \brief Not equal to operator.
     *
     * Compares nodes for inequality.
     *
     * \param [in] node The node to compare.
     * \return [bool] TRUE if nodes are not identical.
     */
    bool operator != (const Node &node) const;


    /*!
     * \brief Assignment operator.
     *
     * Assigns all the properties of a given node (id, coordinates, normal, partition, boundary, and element type).
     *
     * \param [in] node The node to assign.
     * \return [Node] The assigned node.
     */
    Node & operator = (const Node &node);


private:
	int32_t id_;                            /*!< The node's id (position in a list of nodes). */

    Vec3<double> coordinates_;     /*!< The node's coordinates. */

    Vec3<double> normal_;          /*!< The node's normal vector. */

    Partition part_;               /*!< The node's partition. */

    Boundary boundary_;            /*!< The node's boundary. */

    ExplicitSim::ElementType element_type_;     /*!< The node's element type is set during construction. */

};


/*! @} End of Doxygen Groups*/

} //end of namespace ExplicitSim


#endif //EXPLICITSIM_ELEMENTS_NODE_HPP_
