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


#include "ExplicitSim/grid/grid_3d.hpp"

namespace ExplicitSim {


Grid3D::Grid3D()
{}


Grid3D::Grid3D(const Grid3D &grid3D)
{
    // Assign by copying grid3D.
    *this = grid3D;
}


Grid3D::~Grid3D()
{}


void Grid3D::CopyNodesFromMesh(const TetraMesh &tetramesh)
{
    // Copy the mesh nodes.
   this->nodes_ = tetramesh.Nodes();
}


bool Grid3D::operator == (const Grid3D &grid3D) const
{
    // Compare tetrahedral meshes for equality.
    return (this->nodes_ == grid3D.nodes_);
}


bool Grid3D::operator != (const Grid3D &grid3D) const
{
    // Compare tetrahedral meshes for inequality.
    return !(*this == grid3D);
}


Grid3D & Grid3D::operator = (const Grid3D &grid3D)
{
    if (this != &grid3D) {
        // Assign values from tetrahedron.
        this->nodes_ = grid3D.nodes_;
    }

    return *this;
}



} //end of namespace ExplicitSim
