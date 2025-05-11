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

#include "ExplicitSim/sets/node_set.hpp"


namespace ExplicitSim {


NodeSet::NodeSet() : name_("")
{}


NodeSet::NodeSet(const NodeSet &node_set)
{
    this->name_ = node_set.name_;
    this->node_ids_ = node_set.node_ids_;
}


NodeSet::~NodeSet()
{}


bool NodeSet::operator == (const NodeSet &node_set) const
{
    // Compare node sets for equality.
    return ((this->name_ == node_set.name_) &&
            (this->node_ids_ == node_set.node_ids_)
           );
}


bool NodeSet::operator != (const NodeSet &node_set) const
{
    // Compare node sets for inequality.
    return !(*this == node_set);
}


NodeSet & NodeSet::operator = (const NodeSet &node_set)
{
    if (this != &node_set) {
        // Assign values from node_set.
        this->name_ = node_set.name_;
        this->node_ids_ = node_set.node_ids_;
    }

    return *this;
}


} // End of namespace ExplicitSim
