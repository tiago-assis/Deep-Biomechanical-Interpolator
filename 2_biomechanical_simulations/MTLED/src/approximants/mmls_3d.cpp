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
 


#include "ExplicitSim/approximants/mmls_3d.hpp"


namespace ExplicitSim {


Mmls3d::Mmls3d() : base_function_type_(""), exact_derivatives_(false)
{}


Mmls3d::~Mmls3d()
{}


void Mmls3d::SetBasisFunctionType(const std::string &basis_function_type)
{
    // Copy base function type to process it.
    std::string type = basis_function_type;

    // Convert to lower-case.
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    // Check that given type is supported by ExplicitSim.
    if ( (type != "linear") && (type != "quadratic") ) {
        std::string error = "[ExplicitSim ERROR] The given base function type is not supported. Supported types are [linear], [quadratic]";
        throw std::invalid_argument(error.c_str());
    }

    // Set the base function type.
    this->base_function_type_ = type;

}


void Mmls3d::SetExactDerivativesMode(const bool &exact_derivatives)
{
    // Set the exact derivatives mode.
    this->exact_derivatives_ = exact_derivatives;
}


void Mmls3d::ComputeShFuncAndDerivs(const std::vector<Node> &geom_nodes, uint32_t u32NumNodes,
                                         const std::vector<Vec3<double> > &eval_nodes_coords, uint32_t u32NumEvalPoints,
                                         const std::vector<std::vector<uint32_t> > &support_nodes_ids,
                                         const std::vector<double> &influence_radiuses)
{
    // Check if base function type is initialized.
    if (this->base_function_type_ == "") {
        std::string error = "[ExplicitSim ERROR] Cannot compute shape functions and derivatives. Set first the base function type.";
        throw std::runtime_error(error.c_str());
    }

    // Set moment matrix A rows, cols size. For base function type linear.
    int m = 4;
    if (this->base_function_type_ == "quadratic") { m = 10; }

    // Get the number of support nodes of the first node to estimate the reserved memory.
    int estim_support_size = (int)support_nodes_ids[0].size();

    // Reserve triplet vector to store shape function values.
    std::vector<Eigen::Triplet<double, int32_t> > triplet_sh_func;
    triplet_sh_func.reserve(u32NumNodes*estim_support_size);

    // Reserve triplet vector to store shape function x derivative values.
    std::vector<Eigen::Triplet<double, int32_t> > triplet_sh_func_x;
    triplet_sh_func_x.reserve(u32NumNodes*estim_support_size);

    // Reserve triplet vector to store shape function y derivative values.
    std::vector<Eigen::Triplet<double, int32_t> > triplet_sh_func_y;
    triplet_sh_func_y.reserve(u32NumNodes*estim_support_size);

    // Reserve triplet vector to store shape function z derivative values.
    std::vector<Eigen::Triplet<double, int32_t> > triplet_sh_func_z;
    triplet_sh_func_z.reserve(u32NumNodes*estim_support_size);

    // Iterate over the evaluation nodes.
    for (uint32_t eval_id = 0; eval_id < u32NumEvalPoints; eval_id++)
	{
		Vec3<double> eval_node = eval_nodes_coords[eval_id];
		
		// The parts of the distance vector.
        Eigen::VectorXd Dx(support_nodes_ids[eval_id].size());
        Eigen::VectorXd Dy(support_nodes_ids[eval_id].size());
        Eigen::VectorXd Dz(support_nodes_ids[eval_id].size());

        // The distance vector between the evaluation node and the support nodes.
        Eigen::VectorXd dist(support_nodes_ids[eval_id].size());

        // The weight function vector for the evaluation node.
        Eigen::VectorXd weight(support_nodes_ids[eval_id].size());

        // Initialize weighted moment matrix A, B matrix, and px vector.
        Eigen::MatrixXd a = Eigen::MatrixXd::Zero(m, m);
        Eigen::MatrixXd b(m, support_nodes_ids[eval_id].size());
        Eigen::RowVectorXd px(m);

        // Iterate over the support nodes
        for (auto &neighbor_id : support_nodes_ids[eval_id]) {
            // The index of the ith iteration.
            auto i = &neighbor_id - &support_nodes_ids[eval_id][0];

            // Set the ith elements of the distance vector parts.
            Dx[i] = (geom_nodes[neighbor_id].Coordinates().X() - eval_node.X());
            Dy[i] = (geom_nodes[neighbor_id].Coordinates().Y() - eval_node.Y());
            Dz[i] = (geom_nodes[neighbor_id].Coordinates().Z() - eval_node.Z());

            // Set the ith element of the distance vector.
            dist[i] = std::sqrt((Dx[i]*Dx[i]) + (Dy[i]*Dy[i]) + (Dz[i]*Dz[i]));

            // Normalize the ith element of the distance vector with the support node's influence radius.
            dist[i] = dist[i] / influence_radiuses[neighbor_id];

            // Set the limit results epsilon
            double epsilon = 1e-5;

            // Compute the ith element of the weight function vector using a quatric spline weight function.
            //weight[i] = 1. - 6.*dist[i]*dist[i] + 8.*dist[i]*dist[i]*dist[i] - 3.*dist[i]*dist[i]*dist[i]*dist[i];
            double Coeff_b = pow(epsilon,-2)-pow(1+epsilon,-2);
            double Coeff_a = pow(1+epsilon,-2);
            weight[i] = (pow(dist[i]*dist[i]+epsilon,-2)-Coeff_a)/Coeff_b;


            // Compute px vector according to base function type for each neighbor node.
            px(0) = 1.;
            px(1) = geom_nodes[neighbor_id].Coordinates().X();
            px(2) = geom_nodes[neighbor_id].Coordinates().Y();
            px(3) = geom_nodes[neighbor_id].Coordinates().Z();

            if (this->base_function_type_ == "quadratic") {
                px(4) = geom_nodes[neighbor_id].Coordinates().X()*geom_nodes[neighbor_id].Coordinates().X();
                px(5) = geom_nodes[neighbor_id].Coordinates().Y()*geom_nodes[neighbor_id].Coordinates().Y();
                px(6) = geom_nodes[neighbor_id].Coordinates().Z()*geom_nodes[neighbor_id].Coordinates().Z();
                px(7) = geom_nodes[neighbor_id].Coordinates().X()*geom_nodes[neighbor_id].Coordinates().Y();
                px(8) = geom_nodes[neighbor_id].Coordinates().Y()*geom_nodes[neighbor_id].Coordinates().Z();
                px(9) = geom_nodes[neighbor_id].Coordinates().X()*geom_nodes[neighbor_id].Coordinates().Z();
            }

            // Compute moment matrix A and B matrix.
            a.noalias() += (weight[i] * px.transpose() * px);
            b.col(i) = weight[i] * px.transpose();
        }

        // Apply correction factor on moment matrix A lower diagonal if quadratic base function is used.
        if (this->base_function_type_ == "quadratic") {
            double cf = 1e-7;
            a(4,4) += cf; a(5,5) += cf; a(6,6) += cf;
            a(7,7) += cf; a(8,8) += cf; a(9,9) += cf;
        }

        // Compute the transpose of the functions coefficients.
        Eigen::MatrixXd coeff_T = a.llt().solve(b);

        // Compute the functions coefficients.
        Eigen::MatrixXd coeff = coeff_T.transpose();

        // Compute exact shape function derivatives if requested.
        if (this->exact_derivatives_ == true) {

            // Initialize derivative of the weight function vector divided by distance [(dw/dr)/r].
            Eigen::VectorXd weight_dist_div_dist(support_nodes_ids[eval_id].size());

            // The spatial derivatives of the distance multiplied by infl. radius.
            // [(ddist/dx)*r_infl.]
            Eigen::VectorXd dist_x_mul_dist(support_nodes_ids[eval_id].size());
            // [(ddist/dy)*r_infl.]
            Eigen::VectorXd dist_y_mul_dist(support_nodes_ids[eval_id].size());
            // [(ddist/dz)*r_infl.]
            Eigen::VectorXd dist_z_mul_dist(support_nodes_ids[eval_id].size());

            // The spatial derivatives of the weight function vector.
            // [dweight/dx = (dweight/ddist) * (ddist/dx)]
            Eigen::VectorXd weight_x(support_nodes_ids[eval_id].size());
            // [dweight/dy = (dweight/ddist) * (ddist/dy)]
            Eigen::VectorXd weight_y(support_nodes_ids[eval_id].size());
            // [dweight/dz = (dweight/ddist) * (ddist/dz)]
            Eigen::VectorXd weight_z(support_nodes_ids[eval_id].size());

            // Initialize derivatives of moment matrix A.
            Eigen::MatrixXd a_x = Eigen::MatrixXd::Zero(m, m);
            Eigen::MatrixXd a_y = Eigen::MatrixXd::Zero(m, m);
            Eigen::MatrixXd a_z = Eigen::MatrixXd::Zero(m, m);

            // Initialize derivatives of matrix B.
            Eigen::MatrixXd b_x(m, support_nodes_ids[eval_id].size());
            Eigen::MatrixXd b_y(m, support_nodes_ids[eval_id].size());
            Eigen::MatrixXd b_z(m, support_nodes_ids[eval_id].size());

            // Iterate over the support nodes
            for (auto &neigh_id : support_nodes_ids[eval_id]) {
                // The index of the ith iteration.
                auto i = &neigh_id - &support_nodes_ids[eval_id][0];

                // Compute derivative of the weight function vector divided by distance.
                //weight_dist_div_dist[i] = - 12. + 24.*dist[i] - 12.*dist[i]*dist[i];
                weight_dist_div_dist[i] = -4.*pow(dist[i]*dist[i]+epsilon,-3)/(pow(epsilon,-2)-pow(1+epsilon,-2));

                // Compute spatial derivatives of the distance multiplied by distance.
                dist_x_mul_dist[i] = - Dx[i] / (influence_radiuses[neigh_id]*influence_radiuses[neigh_id]);
                dist_y_mul_dist[i] = - Dy[i] / (influence_radiuses[neigh_id]*influence_radiuses[neigh_id]);
                dist_z_mul_dist[i] = - Dz[i] / (influence_radiuses[neigh_id]*influence_radiuses[neigh_id]);

                weight_x[i] = weight_dist_div_dist[i] * dist_x_mul_dist[i];
                weight_y[i] = weight_dist_div_dist[i] * dist_y_mul_dist[i];
                weight_z[i] = weight_dist_div_dist[i] * dist_z_mul_dist[i];

                // Recompute px vector according to base function type for each neighbor node.
                px(0) = 1.;
                px(1) = geom_nodes[neigh_id].Coordinates().X();
                px(2) = geom_nodes[neigh_id].Coordinates().Y();
                px(3) = geom_nodes[neigh_id].Coordinates().Z();

                if (this->base_function_type_ == "quadratic") {
                    px(4) = geom_nodes[neigh_id].Coordinates().X()*geom_nodes[neigh_id].Coordinates().X();
                    px(5) = geom_nodes[neigh_id].Coordinates().Y()*geom_nodes[neigh_id].Coordinates().Y();
                    px(6) = geom_nodes[neigh_id].Coordinates().Z()*geom_nodes[neigh_id].Coordinates().Z();
                    px(7) = geom_nodes[neigh_id].Coordinates().X()*geom_nodes[neigh_id].Coordinates().Y();
                    px(8) = geom_nodes[neigh_id].Coordinates().Y()*geom_nodes[neigh_id].Coordinates().Z();
                    px(9) = geom_nodes[neigh_id].Coordinates().X()*geom_nodes[neigh_id].Coordinates().Z();
                }

                // Compute derivatives of moment matrix A.
                a_x.noalias() += (weight_x[i] * px.transpose() * px);
                a_y.noalias() += (weight_y[i] * px.transpose() * px);
                a_z.noalias() += (weight_z[i] * px.transpose() * px);

                // Compute derivatives of matrix B.
                b_x.col(i) = weight_x[i] * px.transpose();
                b_y.col(i) = weight_y[i] * px.transpose();
                b_z.col(i) = weight_z[i] * px.transpose();
            }

            // Compute the transpose of the functions coefficients derivatives.
            Eigen::MatrixXd coeff_T_x = a.llt().solve(b_x - a_x*coeff_T);
            Eigen::MatrixXd coeff_T_y = a.llt().solve(b_y - a_y*coeff_T);
            Eigen::MatrixXd coeff_T_z = a.llt().solve(b_z - a_z*coeff_T);

            // Compute the the functions coefficients derivatives.
            Eigen::MatrixXd coeff_x = coeff_T_x.transpose();
            Eigen::MatrixXd coeff_y = coeff_T_y.transpose();
            Eigen::MatrixXd coeff_z = coeff_T_z.transpose();

            // Initialize shape function and derivatives values.
            Eigen::VectorXd sh_func_value(support_nodes_ids[eval_id].size());
            Eigen::VectorXd sh_func_x_value(support_nodes_ids[eval_id].size());
            Eigen::VectorXd sh_func_y_value(support_nodes_ids[eval_id].size());
            Eigen::VectorXd sh_func_z_value(support_nodes_ids[eval_id].size());

            // Compute shape function derivatives values according to selected base function type.
            if (this->base_function_type_ == "linear") {
                sh_func_value = coeff.col(0) + coeff.col(1)*eval_node.X() +
                                coeff.col(2)*eval_node.Y() + coeff.col(3)*eval_node.Z();

                sh_func_x_value = coeff.col(1) + coeff_x.col(0) + coeff_x.col(1)*eval_node.X() +
                                  coeff_x.col(2)*eval_node.Y() + coeff_x.col(3)*eval_node.Z();

                sh_func_y_value = coeff.col(2) + coeff_y.col(0) + coeff_y.col(1)*eval_node.X() +
                                  coeff_y.col(2)*eval_node.Y()+ coeff_y.col(3)*eval_node.Z();

                sh_func_z_value = coeff.col(3) + coeff_z.col(0) + coeff_z.col(1)*eval_node.X() +
                                  coeff_z.col(2)*eval_node.Y() + coeff_z.col(3)*eval_node.Z();

            }
            else if (this->base_function_type_ == "quadratic") {
                sh_func_value = coeff.col(0) + coeff.col(1)*eval_node.X() +
                                coeff.col(2)*eval_node.Y() + coeff.col(3)*eval_node.Z() +
                                coeff.col(4)*eval_node.X()*eval_node.X() + coeff.col(5)*eval_node.Y()*eval_node.Y() +
                                coeff.col(6)*eval_node.Z()*eval_node.Z() + coeff.col(7)*eval_node.X()*eval_node.Y() +
                                coeff.col(8)*eval_node.Y()*eval_node.Z() + coeff.col(9)*eval_node.X()*eval_node.Z();

                sh_func_x_value = coeff.col(1) + 2.*eval_node.X()*coeff.col(4) + eval_node.Y()*coeff.col(7) +
                                  eval_node.Z()*coeff.col(9) + coeff_x.col(0) + coeff_x.col(1)*eval_node.X() +
                                  coeff_x.col(2)*eval_node.Y() + coeff_x.col(3)*eval_node.Z() +
                                  coeff_x.col(4)*eval_node.X()*eval_node.X() + coeff_x.col(5)*eval_node.Y()*eval_node.Y() +
                                  coeff_x.col(6)*eval_node.Z()*eval_node.Z() + coeff_x.col(7)*eval_node.X()*eval_node.Y() +
                                  coeff_x.col(8)*eval_node.Y()*eval_node.Z() + coeff_x.col(9)*eval_node.X()*eval_node.Z();

                sh_func_y_value = coeff.col(2) + 2.*eval_node.Y()*coeff.col(5) + eval_node.X()*coeff.col(7) +
                                  eval_node.Z()*coeff.col(8) + coeff_y.col(0) + coeff_y.col(1)*eval_node.X() +
                                  coeff_y.col(2)*eval_node.Y() + coeff_y.col(3)*eval_node.Z() +
                                  coeff_y.col(4)*eval_node.X()*eval_node.X() + coeff_y.col(5)*eval_node.Y()*eval_node.Y() +
                                  coeff_y.col(6)*eval_node.Z()*eval_node.Z() + coeff_y.col(7)*eval_node.X()*eval_node.Y() +
                                  coeff_y.col(8)*eval_node.Y()*eval_node.Z() + coeff_y.col(9)*eval_node.X()*eval_node.Z();

                sh_func_z_value = coeff.col(3) + 2.*eval_node.Z()*coeff.col(6) + eval_node.Y()*coeff.col(8) +
                                  eval_node.X()*coeff.col(9) + coeff_z.col(0) + coeff_z.col(1)*eval_node.X() +
                                  coeff_z.col(2)*eval_node.Y() + coeff_z.col(3)*eval_node.Z() +
                                  coeff_z.col(4)*eval_node.X()*eval_node.X() + coeff_z.col(5)*eval_node.Y()*eval_node.Y() +
                                  coeff_z.col(6)*eval_node.Z()*eval_node.Z() + coeff_z.col(7)*eval_node.X()*eval_node.Y() +
                                  coeff_z.col(8)*eval_node.Y()*eval_node.Z() + coeff_z.col(9)*eval_node.X()*eval_node.Z();

            }

            // Iterate over the support nodes
            for (auto &neigh_id : support_nodes_ids[eval_id]) {
                // The index of the ith iteration.
                auto i = &neigh_id - &support_nodes_ids[eval_id][0];

                // Store the shape function and derivatives values in triplets.
                triplet_sh_func.emplace_back(Eigen::Triplet<double, int32_t>(neigh_id, eval_id, sh_func_value[i]));
                triplet_sh_func_x.emplace_back(Eigen::Triplet<double, int32_t>(neigh_id, eval_id, sh_func_x_value[i]));
                triplet_sh_func_y.emplace_back(Eigen::Triplet<double, int32_t>(neigh_id, eval_id, sh_func_y_value[i]));
                triplet_sh_func_z.emplace_back(Eigen::Triplet<double, int32_t>(neigh_id, eval_id, sh_func_z_value[i]));
            }

        }
        else {
            // Initialize shape function and derivatives values. Difuse shape function derivatives mode.
            Eigen::VectorXd sh_func_value(support_nodes_ids[eval_id].size());
            Eigen::VectorXd sh_func_x_value(support_nodes_ids[eval_id].size());
            Eigen::VectorXd sh_func_y_value(support_nodes_ids[eval_id].size());
            Eigen::VectorXd sh_func_z_value(support_nodes_ids[eval_id].size());

            if (this->base_function_type_ == "linear") {
                sh_func_value = coeff.col(0) + coeff.col(1)*eval_node.X() +
                                coeff.col(2)*eval_node.Y() + coeff.col(3)*eval_node.Z();

                sh_func_x_value = coeff.col(1);
                sh_func_y_value = coeff.col(2);
                sh_func_z_value = coeff.col(3);

            }
            else if (this->base_function_type_ == "quadratic") {
                sh_func_value = coeff.col(0) + coeff.col(1)*eval_node.X() +
                                coeff.col(2)*eval_node.Y() + coeff.col(3)*eval_node.Z() +
                                coeff.col(4)*eval_node.X()*eval_node.X() + coeff.col(5)*eval_node.Y()*eval_node.Y() +
                                coeff.col(6)*eval_node.Z()*eval_node.Z() + coeff.col(7)*eval_node.X()*eval_node.Y() +
                                coeff.col(8)*eval_node.Y()*eval_node.Z() + coeff.col(9)*eval_node.X()*eval_node.Z();

                sh_func_x_value = coeff.col(1) + 2.*eval_node.X()*coeff.col(4) +
                                  eval_node.Y()*coeff.col(7) + eval_node.Z()*coeff.col(9);

                sh_func_y_value = coeff.col(2) + 2.*eval_node.Y()*coeff.col(5) +
                                  eval_node.X()*coeff.col(7) + eval_node.Z()*coeff.col(8);

                sh_func_z_value = coeff.col(3) + 2.*eval_node.Z()*coeff.col(6) +
                                  eval_node.Y()*coeff.col(8) + eval_node.X()*coeff.col(9);
            }

            // Iterate over the support nodes
            for (auto &neigh_id : support_nodes_ids[eval_id]) {
                // The index of the ith iteration.
                auto i = &neigh_id - &support_nodes_ids[eval_id][0];

                // Store the shape function and derivatives values in triplets.
                triplet_sh_func.emplace_back(Eigen::Triplet<double, int32_t>(neigh_id, eval_id, sh_func_value[i]));
                triplet_sh_func_x.emplace_back(Eigen::Triplet<double, int32_t>(neigh_id, eval_id, sh_func_x_value[i]));
                triplet_sh_func_y.emplace_back(Eigen::Triplet<double, int32_t>(neigh_id, eval_id, sh_func_y_value[i]));
                triplet_sh_func_z.emplace_back(Eigen::Triplet<double, int32_t>(neigh_id, eval_id, sh_func_z_value[i]));
            }
        }

    } // End of iteration over the evaluation nodes.

    // Initialize shape function and derivative matrices.
    this->sh_func_ = Eigen::SparseMatrix<double>(u32NumNodes, u32NumEvalPoints);
    this->sh_func_dx_ = Eigen::SparseMatrix<double>(u32NumNodes, u32NumEvalPoints);
    this->sh_func_dy_ = Eigen::SparseMatrix<double>(u32NumNodes, u32NumEvalPoints);
    this->sh_func_dz_ = Eigen::SparseMatrix<double>(u32NumNodes, u32NumEvalPoints);

    // Populate shape function and derivatives matrices.
    this->sh_func_.setFromTriplets(triplet_sh_func.begin(), triplet_sh_func.end());
    this->sh_func_dx_.setFromTriplets(triplet_sh_func_x.begin(), triplet_sh_func_x.end());
    this->sh_func_dy_.setFromTriplets(triplet_sh_func_y.begin(), triplet_sh_func_y.end());
    this->sh_func_dz_.setFromTriplets(triplet_sh_func_z.begin(), triplet_sh_func_z.end());

}


} //end of namespace ExplicitSim
