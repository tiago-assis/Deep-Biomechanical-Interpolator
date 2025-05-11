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


#ifndef EXPLICITSIM_INTEGRATION_INTEG_OPTIONS_HPP_
#define EXPLICITSIM_INTEGRATION_INTEG_OPTIONS_HPP_

/*!
   \file integ_options.hpp
   \brief IntegOption class header file.
   \author Konstantinos A. Mountris
   \date 16/05/2017
*/


namespace ExplicitSim {

/*!
 *  \addtogroup Integration
 *  @{
 */


/*!
 * \struct IntegOptions
 * \brief Structure implemmenting integration options for the generation of integration points of weak-form models.
 */

typedef struct IntegOptions {

    /*!
     * \brief IntegOptions constructor.
     */
    IntegOptions() : is_adaptive_(false), tetra_divisions_(0), integ_points_per_tetra_(0),
        adaptive_level_(0), adaptive_eps_(0.)
    {}


    bool is_adaptive_;                      /*!< The type of integration. If true, adaptive integration is performed. */

    int tetra_divisions_;                   /*!< The number of tetrahedral elements' divisions.
                                                 Admissible divisions [2/4/8]. Used for adaptive integration. */

    int integ_points_per_tetra_;            /*!< The number of integration points per tetrahedron.
                                                 Admissible points [1/4/5]. Used for non adaptive integration. */

    int adaptive_level_;                    /*!< The number of recursive iteration for adaptive integration points generation.*/

    double adaptive_eps_;                   /*!< The approximation estimation error for adaptation termination. */

} IntegOptions;


/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif //EXPLICITSIM_INTEGRATION_INTEG_OPTIONS_HPP_
