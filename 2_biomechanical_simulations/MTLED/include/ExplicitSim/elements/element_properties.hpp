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
   \file element_properties.hpp
   \brief Collection of element properties header file.
   \author Konstantinos A. Mountris
   \date 16/11/2017
*/

#ifndef EXPLICITSIM_ELEMENTS_ELEMENT_PROPERTIES_HPP_
#define EXPLICITSIM_ELEMENTS_ELEMENT_PROPERTIES_HPP_

#include <string>

namespace ExplicitSim {

/*!
 *  \addtogroup Elements
 *  @{
 */


/*!
 * \enum
 */
enum struct ElementType: int {node = 1,                 /*!< Element type -> node. */
                              edge = 2,                 /*!< Element type -> edge. */
                              triangle = 3,             /*!< Element type -> triangle. */
                              quadrangle = 4,           /*!< Element type -> quadrangle. */
                              tetrahedron = 5,          /*!< Element type -> tetrahedron. */
                              hexahedron = 6            /*!< Element type -> hexahedron. */
                             };

/*!
 * \struct Partition
 * \brief Structure describing the partition an element belongs.
 */
typedef struct Partition{
    int id_;                /*!< The index of the partition. */
    std::string name_;      /*!< The name of the partition. */

    bool operator == (const Partition &part) const
    {
        return ((this->id_ == part.id_) &&
                (this->name_ == part.name_) );
    }


    bool operator != (const Partition &part) const
    {
        return !(*this == part);
    }


    Partition & operator = (const Partition &part) {
        if (this != &part) {
            this->id_ = part.id_;
            this->name_ = part.name_;
        }
        return *this;
    }
} Partition;


/*!
 * \struct Boundary
 * \brief Structure describing the boundary an element belongs.
 */
typedef struct Boundary {
    int id_;                /*!< The index of the boundary. */
    std::string name_;      /*!< The name of the boundary. */


    Boundary() : id_(0), name_("") {}


    bool operator == (const Boundary &boundary) const
    {
        return ((this->id_ == boundary.id_) &&
                (this->name_ == boundary.name_) );
    }


    bool operator != (const Boundary &boundary) const
    {
        return !(*this == boundary);
    }


    Boundary & operator = (const Boundary &boundary) {
        if (this != &boundary) {
            this->id_ = boundary.id_;
            this->name_ = boundary.name_;
        }
        return *this;
    }
} Boundary;


/*! @} End of Doxygen Groups*/

} //end of namespace ExplicitSim

#endif //EXPLICITSIM_ELEMENTS_ELEMENT_PROPERTIES_HPP_
