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



/*!
   \file explicit_sim.hpp
   \brief ExplicitSim software header file. Should be included in a project to use ExplicitSim.
   \author Konstantinos A. Mountris
   \date 16/05/2017
*/

#ifndef EXPLICITSIM_EXPLICIT_SIM_HPP_
#define EXPLICITSIM_EXPLICIT_SIM_HPP_

// Collecting ExplicitSim modules' header files.

#include "ExplicitSim/approximants/mmls_3d.hpp"
#include "ExplicitSim/conditions/conditions.hpp"
#include "ExplicitSim/elements/elements.hpp"
#include "ExplicitSim/exporters/exporters.hpp"
#include "ExplicitSim/grid/grids.hpp"
#include "ExplicitSim/integration/integration.hpp"
#include "ExplicitSim/materials/neo_hookean.hpp"
#include "ExplicitSim/materials/ogden.hpp"
#include "ExplicitSim/mesh/mesh.hpp"
#include "ExplicitSim/mesh_io/mesh_io.hpp"
#include "ExplicitSim/models/models.hpp"
#include "ExplicitSim/options_configuration/config_manager.hpp"
#include "ExplicitSim/sets/sets.hpp"
#include "ExplicitSim/solvers/solvers.hpp"
#include "ExplicitSim/support_domain/support_domain.hpp"
#include "ExplicitSim/utilities/utilities.hpp"
#include "ExplicitSim/vectors/vectors.hpp"

#endif //EXPLICITSIM_EXPLICIT_SIM_HPP_
