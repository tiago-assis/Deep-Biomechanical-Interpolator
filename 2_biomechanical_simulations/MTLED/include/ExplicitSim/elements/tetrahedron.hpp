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
   \file tetrahedron.hpp
   \brief Tetrahedron class header file.
   \author Konstantinos A. Mountris
   \date 16/11/2017
*/

#ifndef EXPLICITSIM_ELEMENTS_TETRAHEDRON_HPP_
#define EXPLICITSIM_ELEMENTS_TETRAHEDRON_HPP_

#include "ExplicitSim/vectors/vectors.hpp"
#include "ExplicitSim/elements/element_properties.hpp"
#include "ExplicitSim/elements/node.hpp"

#include <string>

#include <cmath>

namespace ExplicitSim {

/*!
 *  \addtogroup Elements
 *  @{
 */

/*!
 * \class Tetrahedron
 * \brief Class implemmenting a tetrahedron of a mesh.
 *
 */
class Tetrahedron{
public:
    /*!
     * \brief Tetrahedron constructor.
     */
    Tetrahedron();


    /*!
     * \brief Tetrahedron copy constructor.
     * \param [in] tetrahedron The tetrahedron to be copied on construction.
     */
    Tetrahedron(const Tetrahedron &tetrahedron);


    /*!
     * \brief Tetrahedron destructor.
     */
    virtual ~Tetrahedron();


    /*!
     * \brief Set the tetrahedron's index in a list of tetrahedra.
     * \param id The index of the triangle in a list of tetrahedra.
     * \return [void]
     */
    void SetId(const int32_t &id);


    /*!
     * \brief Set the connectivity (nodes indices) of the tetrahedron.
     * \param n1 The number of the first node of the tetrahedron.
     * \param n2 The number of the second node of the tetrahedron.
     * \param n3 The number of the third node of the tetrahedron.
     * \param n4 The number of the fourth node of the tetrahedron.
     * \return [void]
     */
    void SetConnectivity(const uint32_t &n1, const uint32_t &n2,
                     const uint32_t &n3, const uint32_t &n4);


    /*!
     * \brief Set the first node of the tetrahedron.
     * \param n1 The number of the first node of the tetrahedron.
     * \return [void]
     */
    void SetN1(const uint32_t &n1);


    /*!
     * \brief Set the second node of the tetrahedron.
     * \param v2 The number of the second node of the tetrahedron.
     * \return [void]
     */
    void SetN2(const uint32_t &n2);


    /*!
     * \brief Set the third node of the tetrahedron.
     * \param n3 The number of the third node of the tetrahedron.
     * \return [void]
     */
    void SetN3(const uint32_t &n3);


    /*!
     * \brief Set the fourth node of the tetrahedron.
     * \param n4 The number of the fourth node of the tetrahedron.
     * \return [void]
     */
    void SetN4(const uint32_t &n4);


    /*!
     * \brief Set the partition the tetrahedron belongs.
     * \param [in] id The partition's index.
     * \param [in] name The partiotion's name. If omitted, it is considered empty.
     * \return [void]
     */
    void SetPartition(const int &id, std::string name="");


    /*!
     * \brief Set the boundary the tetrahedron belongs.
     * \param [in] id The boundary's index. If zero, the tetrahedron does not belong on boundary.
     * \param [in] name The boundary's name. If omitted, it is considered empty.
     * \return [void]
     */
    void SetBoundary(const int &id, std::string name="");


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
     * \brief Calculate the volume of the tetrahedron.
     * \param [in] nodes Container of nodes to retrieve coordinates according to the tetrahedron's connectivity.
     * \return [double] The volume of the tetrahedron.
     */
    double Volume(const std::vector<Node> &nodes) const;


    /*!
     * \brief Get the Id of the tetrahedron.
     * \return [int] The tetrahedron's id, in other words its position in a list of tetrahedra.
     */
    inline const int32_t & Id() const { return this->id_; }


    /*!
     * \brief Get the first node of the tetrahedron.
     * \return [int] The first node of the tetrahedron.
     */
    inline const uint32_t & N1() const { return this->n1_; }


    /*!
     * \brief Get the second node of the tetrahedron.
     * \return [int] The second node of the tetrahedron.
     */
    inline const uint32_t & N2() const { return this->n2_; }


    /*!
     * \brief Get the third node of the tetrahedron.
     * \return [int] The third node of the tetrahedron.
     */
    inline const uint32_t & N3() const { return this->n3_; }


    /*!
     * \brief Get the fourth node of the tetrahedron.
     * \return [int] The fourth node of the tetrahedron.
     */
    inline const uint32_t & N4() const { return this->n4_; }


    /*!
     * \brief Get the index of the partition the tetrahedron belongs to.
     * \return [int] The tetrahedron's partition index.
     */
    inline const int & PartitionId() const { return this->part_.id_; }


    /*!
     * \brief Get the name of the partition the tetrahedron belongs to.
     * \return [std::string] The tetrahedron's partition name.
     */
    inline const std::string & PartitionName() const { return this->part_.name_; }


    /*!
     * \brief Get the index of the boundary the tetrahedron belongs to.
     * \return [int] The tetrahedron's boundary index.
     */
    inline const int & BoundaryId() const { return this->boundary_.id_; }


    /*!
     * \brief Get the name of the boundary the tetrahedron belongs to.
     * \return [std::string] The tetrahedron's boundary name.
     */
    inline const std::string & BoundaryName() const { return this->boundary_.name_; }


    /*!
     * \brief Get the element type of the tetrahedron.
     * \return [ElementType] The tetrahedron's element type.
     */
    inline const ExplicitSim::ElementType & ElementType() const { return this->element_type_; }


    /*!
     * \brief Equal to operator.
     *
     * Compares tetrahedra for equality.
     *
     * \param [in] tetrahedron The tetrahedron to compare.
     * \return [bool] TRUE if tetrahedra are identical.
     */
    bool operator == (const Tetrahedron &tetrahedron) const;


    /*!
     * \brief Not equal to operator.
     *
     * Compares tetrahedra for inequality.
     *
     * \param [in] tetrahedron The tetrahedron to compare.
     * \return [bool] TRUE if tetrahedra are not identical.
     */
    bool operator != (const Tetrahedron &tetrahedron) const;


    /*!
     * \brief Assignment operator.
     *
     * Assigns all the properties of a given tetrahedron (id, vertices, partition, boundary, and element type).
     *
     * \param [in] tetrahedron The tetrahedron to assign.
     * \return [Tetrahedron] The assigned tetrahedron.
     */
    Tetrahedron & operator = (const Tetrahedron &tetrahedron);


private:
	int32_t id_;                            /*!< The id of the tetrahedron in a list of tetrahedra. */

	uint32_t n1_;                            /*!< The number of the first node of the tetrahedron. */

	uint32_t n2_;                            /*!< The number of the second node of the tetrahedron. */

	uint32_t n3_;                            /*!< The number of the third node of the tetrahedron. */

	uint32_t n4_;                            /*!< The number of the fourth node of the tetrahedron. */

    Partition part_;               /*!< The partition where the tetrahedron belongs. */

    Boundary boundary_;            /*!< The boundary where the tetrahedron belongs (zero if not on boundary). */

    ExplicitSim::ElementType element_type_;     /*!< The element type of the tetrahedron is set during construction. */
};


/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif // EXPLICITSIM_ELEMENTS_TETRAHEDRON_HPP_
