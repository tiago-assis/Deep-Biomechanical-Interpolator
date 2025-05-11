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

#include "ExplicitSim/elements/node.hpp"


namespace ExplicitSim {


Node::Node() : id_(-1), element_type_(ExplicitSim::ElementType::node)
{
    //Id is initialized to -1 to indicate that node is not listed.

    // Initialize coordinates and normal.
    this->coordinates_.Set(0., 0., 0.);
    this->normal_.Set(0., 0., 0.);

    // Initialize partition.
    part_.id_ = 0;
    part_.name_ = "";

    // Initialize boundary (zero -> no boundary).
    boundary_.id_ = 0;
    boundary_.name_ = "";
}


Node::Node(const Node &node)
{
    // Assign by copying node.
    *this = node;
}


Node::~Node()
{}


void Node::SetId(const int32_t &id)
{
    // Set the nodes Id in a list of nodes.
    this->id_ = id;
}


void Node::SetCoordinates(const double &x, const double &y, const double &z)
{
    // Set the coordinates of the node.
    this->coordinates_.Set(x, y, z);
}


void Node::SetCoordinates(const Vec3<double> &coords)
{
    //Set the coordinates of the node.
    this->coordinates_ = coords;
}


void Node::SetCoordX(const double &x)
{
    // Set the node's x coordinate.
    this->coordinates_.SetX(x);
}


void Node::SetCoordY(const double &y)
{
    // Set the node's y coordinate.
    this->coordinates_.SetY(y);
}


void Node::SetCoordZ(const double &z)
{
    // Set the node's z coordinate.
    this->coordinates_.SetZ(z);
}


void Node::SetNormal(const double &n_x, const double &n_y, const double &n_z)
{
    // Set the node's normal.
    this->normal_.Set(n_x, n_y, n_z);
}


void Node::SetNormalX(const double &n_x)
{
    // Set the x component of the node's normal.
    this->normal_.SetX(n_x);
}


void Node::SetNormalY(const double &n_y)
{
    // Set the y component of the node's normal.
    this->normal_.SetY(n_y);
}


void Node::SetNormalZ(const double &n_z)
{
    // Set the z component of the node's normal.
    this->normal_.SetZ(n_z);
}


void Node::SetPartition(const int &id, std::string name)
{
    // Set the id of the partition the node belongs to.
    this->part_.id_ = id;

    // Set the name of the partition the node belongs to.
    this->part_.name_ = name;
}


void Node::SetBoundary(const int &id, std::string name)
{
    // Set the id of the boundary the node belongs to.
    this->boundary_.id_ = id;

    // Set the name of the boundary the node belongs to.
    this->boundary_.name_ = name;
}


void Node::CopyCoordinates(const Vec3<double> &coordinates)
{
    // Set the node's coordinates by a copy.
    this->coordinates_ = coordinates;
}


void Node::CopyNormal(const Vec3<double> &normal)
{
    // Set the node's normal by a copy.
    this->normal_ = normal;
}


void Node::CopyPartition(const Partition &part)
{
    // Set the node's partition by a copy.
    this->part_ = part;
}


void Node::CopyBoundary(const Boundary &boundary)
{
    // Set the node's boundary by a copy.
    this->boundary_ = boundary;
}


bool Node::operator == (const Node &node) const
{
    // Compare nodes.
    return ((this->id_ == node.id_) &&
            (this->coordinates_ == node.coordinates_) &&
            (this->normal_ == node.normal_) &&
            (this->part_ == node.part_) &&
            (this->boundary_ == node.boundary_) &&
            (this->element_type_ == node.element_type_)
           );
}


bool Node::operator != (const Node &node) const
{
    // Compare nodes.
    return !(*this == node);
}


Node & Node::operator = (const Node &node)
{
    if (this != &node) {
        // Assign values from node.
        this->id_ = node.id_;
        this->coordinates_ = node.coordinates_;
        this->normal_ = node.normal_;
        this->part_ = node.part_;
        this->boundary_ = node.boundary_;
        this->element_type_ = node.element_type_;
    }

    return *this;

}


} //end of namespace ExplicitSim


