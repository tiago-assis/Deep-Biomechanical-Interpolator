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
   \file mmls_3d.hpp
   \brief Mmls3d class header file.
   \author Konstantinos A. Mountris
   \date 06/06/2017
*/

#ifndef EXPLICITSIM_APPROXIMANTS_MMLS_3D_HPP_
#define EXPLICITSIM_APPROXIMANTS_MMLS_3D_HPP_


#include "ExplicitSim/vectors/vectors.hpp"
#include "ExplicitSim/elements/node.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <exception>

namespace ExplicitSim {

/*!
 *  \addtogroup ShapeFunctions
 *  @{
 */


/*!
 * \class Mmls3d
 * \brief Class implemmenting modified MLS (moving least squares) shape functions in 3 dimensions.
 */

class Mmls3d
{

public:
    /*!
     * \brief Mmls3d constructor.
     */
    Mmls3d();


    /*!
     * \brief Mmls3d destructor.
     */
    virtual ~Mmls3d();


    /*!
     * \brief Set the type of the basis function for shape functions and derivatives computation.
     *
     * The available types for the basis function are [linear], [quadratic]. The input type is NOT case-sensitive.
     *
     * \param [in] basis_function_type The type of the basis function to be used.
     * \return [void]
     */
    void SetBasisFunctionType(const std::string &basis_function_type);


    /*!
     * \brief Set the exact derivatives mode.
     *
     * If the exact derivatives conditional is true exact derivatives will be computed, otherwise difuse ones will be computed.
     * Default: [false].
     *
     * \param [in] exact_derivatives The conditional of exact derivatives.
     * \return [void]
     */
    void SetExactDerivativesMode(const bool &exact_derivatives);


    /*!
     * \brief Compute the shape functions and their derivatives.
     * \param [in] geom_nodes The nodes describing the model's geometry.
     * \param [in] eval_nodes_coords The coordinates of the evaluation nodes of the shape functions.
     * \param [in] support_nodes_ids The indices of the nodes belonging in the support domain of its evaluation node.
     * \param [in] influence_radiuses The radiuses of influence of each evaluation point.
     * \return [void]
     */
    void ComputeShFuncAndDerivs(const std::vector<Node> &geom_nodes, uint32_t u32NumNodes,
                                const std::vector<Vec3<double> > &eval_nodes_coords, uint32_t u32NumEvalPoints,
                                const std::vector<std::vector<uint32_t> > &support_nodes_ids,
                                const std::vector<double> &influence_radiuses);


    /*!
     * \brief BaseFunctionType
     * \return [std::string]
     */
    inline const std::string & BaseFunctionType() const { return this->base_function_type_; }


    /*!
     * \brief UseExactDerivatives
     * \return [bool]
     */
    inline const bool & UseExactDerivatives() const { return this->exact_derivatives_; }


    /*!
     * \brief Get the shape function sparse matrix.
     * \return [Eigen::SparseMatrix<double>] The shape function sparse matrix.
     */
    inline const Eigen::SparseMatrix<double> & ShapeFunction() const { return this->sh_func_; }


    /*!
     * \brief Get the shape function first X derivative sparse matrix.
     * \return [Eigen::SparseMatrix<double>] The shape function first X derivative sparse matrix.
     */
    inline const Eigen::SparseMatrix<double> & ShapeFunctionDx() const { return this->sh_func_dx_; }


    /*!
     * \brief Get the shape function first Y derivative sparse matrix.
     * \return [Eigen::SparseMatrix<double>] The shape function first Y derivative sparse matrix.
     */
    inline const Eigen::SparseMatrix<double> & ShapeFunctionDy() const { return this->sh_func_dy_; }


    /*!
     * \brief Get the shape function first Z derivative sparse matrix.
     * \return [Eigen::SparseMatrix<double>] The shape function first Z derivative sparse matrix.
     */
    inline const Eigen::SparseMatrix<double> & ShapeFunctionDz() const { return this->sh_func_dz_; }



private:
    std::string base_function_type_;            /*!< The type of the base function to be used during shape functions and derivatives calculation. */

    bool exact_derivatives_;                    /*!< Conditional to state the computation of exact [true] or diffuse [false] derivatives. */

    Eigen::SparseMatrix<double> sh_func_;       /*!< The shape function matrix. */

    Eigen::SparseMatrix<double> sh_func_dx_;     /*!< The shape function x derivative matrix. */

    Eigen::SparseMatrix<double> sh_func_dy_;     /*!< The shape function y derivative matrix. */

    Eigen::SparseMatrix<double> sh_func_dz_;     /*!< The shape function z derivative matrix. */

    double epsilon;
};



/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif //EXPLICITSIM_APPROXIMANTS_MMLS_3D_HPP_
