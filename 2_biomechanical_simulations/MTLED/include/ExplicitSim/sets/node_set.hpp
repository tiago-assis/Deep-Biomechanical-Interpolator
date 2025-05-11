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
   \file node_set.hpp
   \brief NodeSet class header file.
   \author Konstantinos A. Mountris
   \date 30/11/2017
*/


#ifndef EXPLICITSIM_SETS_NODE_SET_HPP_
#define EXPLICITSIM_SETS_NODE_SET_HPP_

#include <vector>
#include <string>

#include <exception>
#include <stdexcept>

namespace ExplicitSim {

/*!
 *  \addtogroup Sets
 *  @{
 */


/*!
 * \class NodeSet
 * \brief Class implemmenting a set of nodes to be used for boundary conditions application.
 */
class NodeSet {
public:
    /*!
     * \brief NodeSet constructor.
     */
    NodeSet();


    /*!
     * \brief NodeSet copy constructor.
     * \param [in] node_set The node set to be copied.
     */
    NodeSet(const NodeSet &node_set);


    /*!
     * \brief NodeSet destructor.
     */
    virtual ~NodeSet();


    inline void SetNodeSetName(const std::string &name) { this->name_ = name; }


    inline std::vector<int32_t> & EditNodeIds() { return this->node_ids_; }


    inline const std::string & Name() const { return this->name_; }


    inline const std::vector<int32_t> & NodeIds() const { return this->node_ids_; }


    /*!
     * \brief Equal to operator.
     *
     * Compares node sets for equality.
     *
     * \param [in] node_set The node set to compare.
     * \return [bool] TRUE if node sets are identical.
     */
    bool operator == (const NodeSet &node_set) const;


    /*!
     * \brief Not equal to operator.
     *
     * Compares node sets for inequality.
     *
     * \param [in] node_set The node set to compare.
     * \return [bool] TRUE if node sets are not identical.
     */
    bool operator != (const NodeSet &node_set) const;


    /*!
     * \brief Assignment operator.
     *
     * Assigns all the properties of a given node sets (node set's name, node indices).
     *
     * \param [in] node_set The node set to assign.
     * \return [NodeSet] The assigned node set.
     */
    NodeSet & operator = (const NodeSet &node_set);


private:

    std::string name_;      /*!< The name of the node set. */

    std::vector<int32_t> node_ids_;      /*!< The indices of the nodes belonging to the set. */

};


/*! @} End of Doxygen Groups*/
} //namespace ExplicitSim

#endif //EXPLICITSIM_SETS_NODE_SET_HPP_
