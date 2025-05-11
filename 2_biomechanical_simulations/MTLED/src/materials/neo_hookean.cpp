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
 *      Benjamin F. ZWICK
 */

#include "ExplicitSim/materials/neo_hookean.hpp"

namespace ExplicitSim {

NeoHookean::NeoHookean()
{}

NeoHookean::~NeoHookean()
{}

Eigen::Matrix3d NeoHookean::SpkStress(const Eigen::Matrix3d &FT, const size_t &integ_point_id) const
{
    // Determinant of deformation gradient.
    double det = std::abs(FT.determinant());

    // 3x3 Identity matrix.
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity(3, 3);

    // Right Cauchy Green deformation tensor (Bathe P506).
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero(3, 3);
    C.noalias() = FT * FT.transpose();

    // Inverse of the right Cauchy Green deformation tensor.
    Eigen::Matrix3d Cinv = C.inverse();

    return this->mu_[integ_point_id] * std::pow(det, -(2./3.)) * (identity - C.trace()/3.*Cinv) +
           this->bulk_modulus_[integ_point_id] * det * (det - 1.) * Cinv;

}

} //end of namespace ExplicitSim
