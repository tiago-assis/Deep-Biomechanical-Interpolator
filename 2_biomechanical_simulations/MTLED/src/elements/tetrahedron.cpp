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


#include "ExplicitSim/elements/tetrahedron.hpp"


namespace ExplicitSim {


Tetrahedron::Tetrahedron() : id_(-1), element_type_(ExplicitSim::ElementType::tetrahedron)
{
    // Id and vertices are initialized to -1 to indicate that tetrahedron is not listed and invalid.

    // Initialize partition.
    part_.id_ = 0;
    part_.name_ = "";

    // Initialize boundary (zero -> no boundary).
    boundary_.id_ = 0;
    boundary_.name_ = "";
}


Tetrahedron::Tetrahedron(const Tetrahedron &tetrahedron)
{
    // Assign by copying tetrahedron.
    *this = tetrahedron;
}


Tetrahedron::~Tetrahedron()
{}


void Tetrahedron::SetId(const int32_t &id)
{
    // Set the tetrahedron's id in a list of tetrahedra.
    this->id_ = id;
}


void Tetrahedron::SetConnectivity(const uint32_t &n1, const uint32_t &n2, const uint32_t &n3, const uint32_t &n4)
{
    // Set the connectivity (nodes indices) of the tetrahedron.
    this->n1_ = n1;
    this->n2_ = n2;
    this->n3_ = n3;
    this->n4_ = n4;
}


void Tetrahedron::SetN1(const uint32_t &n1)
{
    // Set the first vertex of the tetrahedron.
    this->n1_ = n1;
}


void Tetrahedron::SetN2(const uint32_t &n2)
{
    // Set the second vertex of the tetrahedron.
    this->n2_ = n2;
}


void Tetrahedron::SetN3(const uint32_t &n3)
{
    // Set the third vertex of the tetrahedron.
    this->n3_ = n3;
}


void Tetrahedron::SetN4(const uint32_t &n4)
{
    // Set the fourth vertex of the tetrahedron.
    this->n4_ = n4;
}


void Tetrahedron::SetPartition(const int &id, std::string name)
{
    // Set the id of the partition the tetrahedron belongs to.
    this->part_.id_ = id;

    // Set the name of the partition the tetrahedron belongs to.
    this->part_.name_ = name;
}


void Tetrahedron::SetBoundary(const int &id, std::string name)
{
    // Set the id of the boundary the tetrahedron belongs to.
    this->boundary_.id_ = id;

    // Set the name of the boundary the tetrahedron belongs to.
    this->boundary_.name_ = name;
}


void Tetrahedron::CopyPartition(const Partition &part)
{
    // Set the tetrahedron's partition by a copy.
    this->part_ = part;
}


void Tetrahedron::CopyBoundary(const Boundary &boundary)
{
    // Set the tetrahedron's boundary by a copy.
    this->boundary_ = boundary;
}


double Tetrahedron::Volume(const std::vector<Node> &nodes) const
{
    // Check if nodes container is empty.
    if (nodes.empty()) {
        std::string error = "ERROR: Empty nodes list given. Cannot calculate tetrahedron's volume.";
        throw std::invalid_argument(error.c_str());
    }

    double a1 = nodes[this->n1_].Coordinates().Y() * (nodes[this->n2_].Coordinates().Z() - nodes[this->n3_].Coordinates().Z()) -
                nodes[this->n1_].Coordinates().Z() * (nodes[this->n2_].Coordinates().Y() - nodes[this->n3_].Coordinates().Y()) +
                (nodes[this->n2_].Coordinates().Y() * nodes[this->n3_].Coordinates().Z() - nodes[this->n3_].Coordinates().Y() * nodes[this->n2_].Coordinates().Z());

    double a2 = nodes[this->n1_].Coordinates().X() * (nodes[this->n2_].Coordinates().Z() - nodes[this->n3_].Coordinates().Z()) -
                nodes[this->n1_].Coordinates().Z() * (nodes[this->n2_].Coordinates().X() - nodes[this->n3_].Coordinates().X()) +
                (nodes[this->n3_].Coordinates().Z() * nodes[this->n2_].Coordinates().X() - nodes[this->n2_].Coordinates().Z() * nodes[this->n3_].Coordinates().X());

    double a3 = nodes[this->n1_].Coordinates().X() * (nodes[this->n2_].Coordinates().Y() - nodes[this->n3_].Coordinates().Y()) -
                nodes[this->n1_].Coordinates().Y() * (nodes[this->n2_].Coordinates().X() - nodes[this->n3_].Coordinates().X()) +
                (nodes[this->n2_].Coordinates().X() * nodes[this->n3_].Coordinates().Y() - nodes[this->n3_].Coordinates().X() * nodes[this->n2_].Coordinates().Y());

    double a4 = nodes[this->n1_].Coordinates().X() * (nodes[this->n2_].Coordinates().Y() * nodes[this->n3_].Coordinates().Z() - nodes[this->n3_].Coordinates().Y() * nodes[this->n2_].Coordinates().Z() ) -
                nodes[this->n1_].Coordinates().Y() * (nodes[this->n2_].Coordinates().X() * nodes[this->n3_].Coordinates().Z() - nodes[this->n3_].Coordinates().X() * nodes[this->n2_].Coordinates().Z() ) +
                nodes[this->n1_].Coordinates().Z() * (nodes[this->n2_].Coordinates().X() * nodes[this->n3_].Coordinates().Y() - nodes[this->n3_].Coordinates().X() * nodes[this->n2_].Coordinates().Y() );

    // The volume of the tetrahedron.
    return std::fabs( (nodes[this->n4_].Coordinates().Y() * a2 - nodes[this->n4_].Coordinates().Z() * a3 +
                       a4 - nodes[this->n4_].Coordinates().X() * a1) ) / 6.;

}


bool Tetrahedron::operator == (const Tetrahedron &tetrahedron) const
{
    // Compare tetrahedra for equality.
    return ((this->id_ == tetrahedron.id_) &&
            (this->n1_ == tetrahedron.n1_) &&
            (this->n2_ == tetrahedron.n2_) &&
            (this->n3_ == tetrahedron.n3_) &&
            (this->n4_ == tetrahedron.n4_) &&
            (this->part_ == tetrahedron.part_) &&
            (this->boundary_ == tetrahedron.boundary_) &&
            (this->element_type_ == tetrahedron.element_type_)
           );
}


bool Tetrahedron::operator != (const Tetrahedron &tetrahedron) const
{
    // Compare tetrahedra for inequality.
    return !(*this == tetrahedron);
}


Tetrahedron & Tetrahedron::operator = (const Tetrahedron &tetrahedron)
{
    if (this != &tetrahedron) {
        // Assign values from tetrahedron.
        this->id_ = tetrahedron.id_;
        this->n1_ = tetrahedron.n1_;
        this->n2_ = tetrahedron.n2_;
        this->n3_ = tetrahedron.n3_;
        this->n4_ = tetrahedron.n4_;
        this->part_ = tetrahedron.part_;
        this->boundary_ = tetrahedron.boundary_;
        this->element_type_ = tetrahedron.element_type_;
    }

    return *this;
}


} //end of namespace ExplicitSim
