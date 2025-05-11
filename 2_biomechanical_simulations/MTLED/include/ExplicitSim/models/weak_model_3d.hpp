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


#ifndef EXPLICITSIM_MODELS_WEAK_MODEL_3D_HPP_
#define EXPLICITSIM_MODELS_WEAK_MODEL_3D_HPP_

/*!
   \file weak_model_3d.hpp
   \brief WeakModel3D class header file.
   \author Konstantinos A. Mountris
   \date 16/05/2017
*/


#include "ExplicitSim/integration/integration.hpp"
#include "ExplicitSim/support_domain/support_domain.hpp"
#include "ExplicitSim/utilities/logger.hpp"
#include "ExplicitSim/grid/grids.hpp"
#include "ExplicitSim/mesh/mesh.hpp"

#include <Eigen/Dense>

#include <string>
#include <algorithm>

#include <stdexcept>
#include <exception>


namespace ExplicitSim {

/*!
 *  \addtogroup Models
 *  @{
 */


/*!
 * \class WeakModel3D
 * \brief Class implemmenting a geometrical 3D model for weak-form meshless analysis.
 *
 */

class WeakModel3D {
public:
    /*!
     * \brief WeakModel3D constructor.
     */
    WeakModel3D();


    /*!
     * \brief WeakModel3D destructor.
     */
    virtual ~WeakModel3D();


    /*!
     * \brief Load the mesh representation of the model.
     * \param [in] mesh_filename The filename of the tetrahedral mesh to load.
     * \return [void]
     */
    void LoadMeshRepresentation(const std::string &mesh_filename);


    /*!
     * \brief Create the grid representation of the model.
     *
     * In order to create the grid representation of the model,
     * the mesh representation must has been already loaded.
     *
     * \return [void]
     */
    void CreateGridRepresentation();


    /*!
     * \brief Create the ntegration points of the model.
     *
     * The integration points are points distributed in the elements of the model's mesh
     * according to the given integration options. They are used for approximation of the
     * shape functions and derivatives.
     *
     * \param [in] options The integration options for the creation of the integration points.
     * \param [in] support_dom The support (influence) domains of the model's nodes. Used for adaptive integration.
     * \return [void]
     */
    void CreateIntegrationPoints(const IntegOptions &options, const SupportDomain &support_dom);


    /*!
     * \brief Compute the distributed mass on the model's grid points.
     * \param [in] density The density values associated to the model's grid points.
     * \param [in] time_steps The time steps associated to the model's integration points.
     * \param [in] support_nodes_ids The indices of the nodes belonging in the support domain of its evaluation node.
     * \param [in] scaling The conditional determining if mass scaling will be applied. [Default: No scaling].
     */
    void ComputeMass(const std::vector<double> &density, const std::vector<double> &time_steps, const double &max_time_step,
                     const std::vector<std::vector<uint32_t> > &support_nodes_ids, bool scaling=false);


    /*!
     * \brief Get the tetrahedral mesh representation of the model.
     * \return [ExplitSim::TetraMesh] The model's tetrahedral mesh representation.
     */
    inline const TetraMesh & TetrahedralMesh() const { return this->tetramesh_; }


    /*!
     * \brief Get the number of elements of the model's tetrahedra mesh.
     * \return [int] The number of elements of the model's tetrahedra mesh.
     */
    inline int TetrahedraMeshElemsNum() const { return static_cast<int>(this->tetramesh_.Elements().size()); }


    /*!
     * \brief Get the 3D grid representation of the model.
     * \return [ExplitSim::Grid3D] The model's 3D grid representation.
     */
    inline const Grid3D & Grid() const { return this->grid_; }


    /*!
     * \brief Get the integration points of the model.
     * \return [IntegPoints] The model's integration points.
     */
    inline IntegPoints & IntegrationPoints() { return this->integ_points_; }


    /*!
     * \brief Get the distributed mass of the model.
     * \return [std::vector<double>] The distributed mass of the model.
     */
    inline const std::vector<double> & Mass() const { return this->mass_; }


    /*!
     * \brief Get the distributed mass of the model in 3d matrix representation.
     *
     * Each column of the 3d matrix stores the distributed mass vector container.
     *
     * \return [Eigen::MatrixXd] The distributed mass of the model in 3d matrix representation.
     */
    inline const Eigen::MatrixXd & MassMatrix() const { return this->mass_matrix_; }


    /*!
     * \brief Get the mass scaling conditional of the model.
     * \return [bool] The mass scaling conditional of the model.
     */
    inline const bool & IsMassScaled() const { return this->is_mass_scaled_; }


    /*!
     * \brief Get the minimum mass scaling factor of the model's grid distributed mass.
     * \return [double] The minimum mass scaling factor of the model's grid distributed mass.
     */
    inline const double & MinMassScaleFactor() const { return this->min_mass_scale_factor_; }


    /*!
     * \brief Get the maximum mass scaling factor of the model's grid distributed mass.
     * \return [double] The maximum mass scaling factor of the model's grid distributed mass.
     */
    inline const double & MaxMassScaleFactor() const { return this->max_mass_scale_factor_; }



private:
    TetraMesh tetramesh_;             /*!< The tetrahedral mesh representation of the model. */

    Grid3D grid_;                     /*!< The 3D grid representation of the model. */

    IntegPoints integ_points_;             /*!< The integration points of the model. */

    std::vector<double> mass_;             /*!< The distributed mass over the model's grid points. */

    Eigen::MatrixXd mass_matrix_;          /*!< The distributed mass over the model's grid points in 3d matrix representation. */

    bool is_mass_scaled_;                  /*!< Conditional for scaling the distributed mass over the model's grid points. */

    double min_mass_scale_factor_;         /*!< The minimum mass scaling factor. */

    double max_mass_scale_factor_;         /*!< The maximum mass scaling factor. */

};




/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif //EXPLICITSIM_MODELS_WEAK_MODEL_3D_HPP_


