/*
 * ExplicitSim - Software for solving PDEs using explicit methods.
 * Copyright (C) 2017  <Konstantinos . Mountris> <konstantinos.mountris@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR  PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received  copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors (alphabetically):
 *      George C. BOURANTAS
 *      Grand R. JOLDES
 *      Konstantinos . MOUNTRIS
 *      Benjamin F. ZWICK
 */

#include <boost/thread/thread.hpp>

#include "ExplicitSim/solvers/mtled.hpp"
#include<ctime>

namespace ExplicitSim {

Mtled::Mtled() : min_step_(0.), max_step_(0.), stable_step_(0.), total_time_steps_num(0), save_progress_steps_(1)
{
	uiNumberOfThreads = boost::thread::hardware_concurrency();
	std::cout << "[ExplicitSim] Number of threads: " << uiNumberOfThreads << std::endl;
}


Mtled::~Mtled()
{}


void Mtled::ComputeTimeSteps(const std::vector<double> &wave_speed,
                             const std::vector< std::vector<uint32_t> > &neighbors_ids,
                             const Mmls3d &model_approximant)
{
    // Check that size of containers is consistent.
    if ( (wave_speed.size() != neighbors_ids.size()) ||
         (static_cast<int>(wave_speed.size()) != model_approximant.ShapeFunctionDx().cols()) ||
         (static_cast<int>(wave_speed.size()) != model_approximant.ShapeFunctionDy().cols()) ||
         (static_cast<int>(wave_speed.size()) != model_approximant.ShapeFunctionDz().cols()) ) {

        throw std::invalid_argument(Logger::Error("Could not compute time steps. "
                                                  "Check the size consistency of the input variables.").c_str());
    }

    // Clear the solver's time steps container.
    this->time_steps_.clear();

    // Compute time step for each evaluation point.
    double step = 0.;
    for (auto &point_speed : wave_speed) {

        // The index of the ith evaluation point.
        auto i = &point_speed - &wave_speed[0];

        // Gather x, y, z derivatives in single matrix.
        Eigen::MatrixXd derivs_mat(neighbors_ids[i].size(), 3);

        int row_id = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(model_approximant.ShapeFunctionDx(),i); it; ++it) {
            derivs_mat(row_id, 0) = it.value();
            row_id++;
        }

        row_id = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(model_approximant.ShapeFunctionDy(),i); it; ++it) {
            derivs_mat(row_id, 1) = it.value();
            row_id++;
        }

        row_id = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(model_approximant.ShapeFunctionDz(),i); it; ++it) {
            derivs_mat(row_id, 2) = it.value();
            row_id++;
        }

        // Square the elements of the derivatives matrix.
        derivs_mat = derivs_mat.cwiseProduct(derivs_mat);

        // Compute the time step for the ith evaluation point.
        step = point_speed * std::sqrt(neighbors_ids[i].size() * derivs_mat.sum());
        step = 2. / step;

        // Store the time step for the ith evaluation point.
        this->time_steps_.emplace_back(step);

    }

    // Get the value of the minimum time step.
    this->min_step_ = *std::min_element(this->time_steps_.begin(), this->time_steps_.end());

    // Get the value of the maximum time step.
    this->max_step_ = *std::max_element(this->time_steps_.begin(), this->time_steps_.end());

}


void Mtled::ComputeStableStep(const bool &is_mass_scaled, double safety_factor)
{
    // Set the stable time step according to the mass scaling.
    if (is_mass_scaled) { this->stable_step_ = this->max_step_ / safety_factor; }
    else { this->stable_step_ = this->min_step_ / safety_factor; }
}


void Mtled::ComputeTotalTimeStepsNum(const int &load_steps_num, const int &equilibrium_steps_num)
{
    this->total_time_steps_num = load_steps_num + equilibrium_steps_num;
}


void Mtled::Solve(WeakModel3D &weak_model_3d, std::vector<std::vector<uint32_t> > &neighbor_ids, ConditionsHandler &cond_handler,
                  const Mmls3d &model_approximant, Material &material, const DynRelaxProp &dyn_relax_prop, const bool &use_ebciem, const Mmls3d &nodal_approximant)
{
	pweak_model_3d_ = &weak_model_3d;
	pneighbor_ids_ = &neighbor_ids;
	pmaterial_ = &material;

    // Displacements and forces matrices initialization.
    Eigen::MatrixXd disp = Eigen::MatrixXd::Zero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);
    Eigen::MatrixXd disp_new = Eigen::MatrixXd::Zero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);
    Eigen::MatrixXd disp_old = Eigen::MatrixXd::Zero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);
    Eigen::MatrixXd disp_saved = Eigen::MatrixXd::Zero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);
    Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);
    Eigen::MatrixXd forces_saved = Eigen::MatrixXd::Zero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);
	Eigen::MatrixXd external_forces = Eigen::MatrixXd::Zero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);
	Eigen::MatrixXd disp_diff = Eigen::MatrixXd::Zero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);
	Eigen::MatrixXd force_diff = Eigen::MatrixXd::Zero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);
    // Total forces and gravity forces matrices initialization.
    Eigen::MatrixXd total_forces = Eigen::MatrixXd::Zero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);
    Eigen::MatrixXd gravity_forces = Eigen::MatrixXd::Ones(weak_model_3d.TetrahedralMesh().NodesNum(), 3);
    Eigen::MatrixXd gravity_forces_in_use = Eigen::MatrixXd::Zero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);
    Eigen::MatrixXd inter_external_forces = Eigen::MatrixXd::Zero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);



    // Set up gravity matrix.
    std::vector<float> gravity_vec = {9.81 * cond_handler.gx(), 9.81 * cond_handler.gy(), 9.81 * cond_handler.gz()};
    //std::vector<float> gravity_vec = {0., 0., 0};
    //std::cout <<"gx is " <<gravity_vec3[3]
    //cond_handler.ApplyGravity();
    //std:: cout << "gravity_vec [2] is "<<gravity_vec[2]<<std::endl;

    for (unsigned int i = 0; i < weak_model_3d.TetrahedralMesh().NodesNum(); i++)
    {
        gravity_forces(i,0) = gravity_vec[0] * gravity_forces(i,0);
        
        gravity_forces(i,1) = gravity_vec[1] * gravity_forces(i,1);
        
        gravity_forces(i,2) = gravity_vec[2] * gravity_forces(i,2);
    }/*end of set up gravity matrix for loop*/
	
	pforces_ = &forces;
	pdisplacements_ = &disp;

	// init contacts
    //set boInitialDisplacement = false
	cond_handler.vApplyContacts(disp_new, true, nodal_approximant);
    //cond_handler.vApplyContacts(disp_new, false, nodal_approximant);

	// multi-threading control
	astThreadControl = new tstThreadControl[uiNumberOfThreads];

	size_t numIPs = weak_model_3d.IntegrationPoints().Weights().size();
	size_t uiNumIPsPerThread = numIPs / uiNumberOfThreads + 1;
	for (unsigned int i = 0; i < uiNumberOfThreads; i++)
	{
		tstThreadControl *pstThreadControl = &astThreadControl[i];
		pstThreadControl->uiFirstIdx = i*uiNumIPsPerThread;
		if (pstThreadControl->uiFirstIdx >= numIPs) pstThreadControl->boDoesWork = false;
		else {
			pstThreadControl->boDoesWork = true;
			pstThreadControl->uiLastIdx = pstThreadControl->uiFirstIdx + uiNumIPsPerThread - 1;
			if (pstThreadControl->uiLastIdx >= numIPs) pstThreadControl->uiLastIdx = numIPs - 1;
		}
	}
	

	 // Clear saved disps and forces.
    this->saved_disps_.clear();
    this->saved_forces_.clear();
	this->saved_external_forces_.clear();
	this->saved_times_.clear();

    // Reserve memory for the states to be recorded
	uint32_t u32NumSavedResults = 2;
	if (this->save_progress_steps_ != 0) {
		u32NumSavedResults = this->total_time_steps_num / this->save_progress_steps_;
		if (u32NumSavedResults == 0) u32NumSavedResults = 1;
	}
	this->saved_disps_.reserve(u32NumSavedResults);
	this->saved_external_forces_.reserve(u32NumSavedResults);
    this->saved_forces_.reserve(u32NumSavedResults);
	this->saved_times_.reserve(u32NumSavedResults);

    // Store initial disps and forces.
    this->saved_disps_.emplace_back(disp);
	this->saved_external_forces_.emplace_back(external_forces);
    this->saved_forces_.emplace_back(forces);
	this->saved_times_.emplace_back(0.0f);

    // Retrieve the load steps number from the total steps and the equilibrium steps difference.
    int step_num_load = this->total_time_steps_num - dyn_relax_prop.EquilibriumStepsNum();

    // Dynamic Relaxation variables.
    bool stabilized_conv_rate = false;
    bool conv_disp_updated = false;

    double conv_rate = dyn_relax_prop.LoadConvRate();
    double old_conv_rate = dyn_relax_prop.LoadConvRate();

    int termination_count = 0;
    int samples_num = 0;
    int no_update_steps_num = 0;

    // Apply load condition at first time step (0) on new displacements.
    //cond_handler.ApplyLoadingConditions(0, disp_new);

    // Iterate over integration points to collect xyz derivatives matrices.
    std::vector<Eigen::MatrixXd> deriv_mats;
    deriv_mats.reserve(weak_model_3d.IntegrationPoints().PointsNum());
    for (int ip = 0; ip != weak_model_3d.IntegrationPoints().PointsNum(); ++ip) {

        // Gather x, y, z derivatives in single matrix.
        Eigen::MatrixXd xyz_derivs(neighbor_ids[ip].size(), 3);

        int row_id = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(model_approximant.ShapeFunctionDx(),ip); it; ++it) {
            xyz_derivs(row_id, 0) = it.value();
            row_id++;
        }

        row_id = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(model_approximant.ShapeFunctionDy(),ip); it; ++it) {
            xyz_derivs(row_id, 1) = it.value();
            row_id++;
        }

        row_id = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(model_approximant.ShapeFunctionDz(),ip); it; ++it) {
            xyz_derivs(row_id, 2) = it.value();
            row_id++;
        }

        // Store the derivatives matrix for each integration point in the container.
        deriv_mats.emplace_back(xyz_derivs);

    }

	pderiv_mats_ = &deriv_mats;

    // Iterate over the total number of time steps.
    int steps_counter = 0;
    for (auto step = 0; step < this->total_time_steps_num; ++step) {
        // Increase steps_counter to count the performing steps.
        steps_counter++;

        // Update displacements and make force zero (note the order).
        disp_old = disp;
        disp = disp_new;
        forces.setZero(weak_model_3d.TetrahedralMesh().NodesNum(), 3);

        // Compute the forces at each time step.
        this->ComputeForces();

        // Update saved displacements, forces and convergence rates.
        if (steps_counter == step_num_load) {
            disp_saved = disp;
            forces_saved = forces;
            conv_rate = dyn_relax_prop.AfterLoadConvRate();
            old_conv_rate = dyn_relax_prop.AfterLoadConvRate();
        }

        // Use forces to update displacements using explicit integration and
        // mass proportional damping (Dynamic Relaxation)
        double f8x = (conv_rate + 1.) * (this->stable_step_/2.);

        // Compute new displacements.
        /*disp_new = -f8x*f8x*(forces.array() / weak_model_3d.MassMatrix().array()).matrix() -
                    conv_rate*conv_rate*disp_old + (1. + conv_rate*conv_rate)*disp;*/

        // Compute new displacements (include gravity)
        
        float8 f8RelT = (float8)steps_counter / step_num_load;
        if (steps_counter > step_num_load)
        {
            f8RelT = 1;
        }

        gravity_forces_in_use = LC.dValue(f8RelT) * gravity_forces; 
       
        cond_handler.ApplyExternalForce(f8RelT, external_forces);

        inter_external_forces = external_forces - forces;


        disp_new = f8x*f8x*(gravity_forces_in_use.array() + inter_external_forces.array() / weak_model_3d.MassMatrix().array()).matrix() - 
                            conv_rate*conv_rate*disp_old + (1. + conv_rate*conv_rate)*disp;

        //std::cout << "disp_new after explicit eq last value (point on the top surface) :  "<< disp_new(1935,2) <<std::endl;
        

		// Apply boundary conditions.
/*		float8 f8RelT = (float8)steps_counter / step_num_load;
		if (steps_counter > step_num_load)
		{
			f8RelT = 1;
		}*/
		cond_handler.ApplyBoundaryConditions(f8RelT, disp_new, nodal_approximant, external_forces, f8x*f8x);

        // Check for divergence.
        double max_current_disp = disp_new.cwiseAbs().col(0).maxCoeff();
        //double max_load_disp = 10 * cond_handler.fGetMaxAbsDisplacement();
        double max_disp = 100;


        // Use squared values to avoid using absolute.
        if (max_current_disp*max_current_disp > max_disp*max_disp) {
            std::cout << "[ExplicitSim WARNING] MTLED solution has become unbounded at step " +
                                std::to_string(step) + ". Reduce time step.\n\r";
            // Stop solution if become unstable and return.
            std::cout << "The current max disp is: " << max_current_disp <<std::endl;
            break;
        }

        // Termination criteria and convergence rate.
        if (steps_counter > step_num_load) {
            // Estimate the lower oscilation frequency.
            disp_diff = disp - disp_saved;
            force_diff = forces - forces_saved;

            // Apply loading conditions to force difference matrix.
            cond_handler.vZeroConstrainedDOFs(force_diff);

            double k_sum = (disp_diff.array() * force_diff.array()).sum();
            double m_sum = (disp_diff.array() * disp_diff.array() * weak_model_3d.MassMatrix().array()).sum();

            // Update convergence rate adaptively.
            if (steps_counter < (step_num_load + dyn_relax_prop.StopUpdateConvRateStepsNum()) ) {
                // Ensure k_sum is possitive.
                k_sum = std::abs(k_sum);

                // Updates in convergence rates.
                if ((m_sum > 1.e-13) && (k_sum > 1.e-13)) {
                    // Reset no update steps number.
                    no_update_steps_num = 0;

                    // Minimum frequency.
                    double min_freq = std::sqrt(k_sum / m_sum);

                    // Kmat condition number (square root).
                    double k_cond_number_root = 2. / (min_freq * this->stable_step_);

                    double temp_conv_rate = (k_cond_number_root - 1.) / (k_cond_number_root + 1.);

                    if (std::abs(temp_conv_rate - old_conv_rate) < dyn_relax_prop.ConvRateDeviation()) {
                        samples_num++;
                        if (samples_num >= dyn_relax_prop.StableConvRateStepsNum()) {
                            if (conv_disp_updated) {
                                conv_disp_updated = false;

                                // Update stabilized state if convergence rate deviation criterion is satisfied.
                                if (std::abs(temp_conv_rate - conv_rate) < dyn_relax_prop.ConvRateStopDeviation()) { stabilized_conv_rate = true; }

                                // Update convergence rate.
                                conv_rate = temp_conv_rate;
                            }
                        }
                    }
                    else {
                        // Reset number of samples to zero.
                        samples_num = 0;
                    }

                    // Update old convergence rate.
                    old_conv_rate = temp_conv_rate;

                } // End of Updates in convergence rates.

            }
            else {
                no_update_steps_num++;
                if (no_update_steps_num >= 10*dyn_relax_prop.StableConvRateStepsNum()) { 
					stabilized_conv_rate = true; 
				}
            }


            // Check termination criteria.
            if (stabilized_conv_rate) {

                // Compute maximum displacement variation.
                double max_disp_var = (disp_new - disp).cwiseAbs().maxCoeff();

                // Adjust convergence rate value.
                double estim_conv_rate = conv_rate + dyn_relax_prop.StopConvRateError()*(1. - conv_rate);
                double estim_error = max_disp_var * estim_conv_rate / (1. - estim_conv_rate);

                if (estim_error < dyn_relax_prop.StopAbsError()) {
                    termination_count++;
                    if (termination_count >= dyn_relax_prop.StopStepsNum()) {
                        std::cout << "[ExplicitSim] MTLED solution tolerance has been satisfied at step: " << step+1 << "\n";
                        break;
                    }
                }
                else { termination_count = 0; }

            } //End of check termination criteria.

            // Update saved forces and displacements.
            if ( (steps_counter - step_num_load) % dyn_relax_prop.ForceDispUpdateStepsNum() == 0 ) {
                disp_saved = disp;
                forces_saved = forces;
                conv_disp_updated = true;
            }

        } // End of Termination criteria and convergence rate.

        // Output MTLED progress.
        if (this->save_progress_steps_ > 0) {
            if(steps_counter % this->save_progress_steps_ == 0) {
                this->saved_disps_.emplace_back(disp);
				this->saved_external_forces_.emplace_back(external_forces);
                this->saved_forces_.emplace_back(forces);
				this->saved_times_.emplace_back((float)(steps_counter*this->stable_step_));
                std::cout << "[ExplicitSim] MTLED solver completed " << steps_counter
                          << " / " << this->total_time_steps_num << " steps\n";
            }
        }

    } //End of time steps iteration.

    // Check for convergence satisfaction.
    if (steps_counter == this->total_time_steps_num) {
		std::cout << Logger::Message("Convergence rate of MTLED solution at termination: ") << conv_rate << std::endl;
		std::cout << Logger::Message("WARNING: Solution tolerance has not been satisfied. Increase the equilibrium time.") << std::endl;
    }
    else {
        std::cout << Logger::Message("Convergence rate of MTLED solution at termination: ") << conv_rate << std::endl;
    }


	delete[] astThreadControl;

    // Store final disps and forces if they haven't been stored during progress storing.
    if (this->save_progress_steps_ > 0) {
        if (steps_counter % this->save_progress_steps_ != 0) {
            this->saved_disps_.emplace_back(disp);
            this->saved_forces_.emplace_back(forces);
			this->saved_external_forces_.emplace_back(external_forces);
			this->saved_times_.emplace_back((float)(steps_counter*this->stable_step_));
        }
    }
    else {  // Store final state
        this->saved_disps_.emplace_back(disp);
		this->saved_external_forces_.emplace_back(external_forces);
        this->saved_forces_.emplace_back(forces);
		this->saved_times_.emplace_back((float)(steps_counter*this->stable_step_));
    }
}


void Mtled::ApplyShapeFuncToDisplacements(const WeakModel3D &weak_model_3d, const Mmls3d &nodal_approximant)
{
    // Iterate over saved displacements
    for (std::size_t disp_id = 0; disp_id != this->saved_disps_.size(); ++disp_id) {

        // Compute the displaced nodal positions.
        Eigen::MatrixXd saved_disp_backup = this->saved_disps_[disp_id];

        // Iterate over model nodes.
		for (uint32_t node_id = 0; node_id < weak_model_3d.TetrahedralMesh().NodesNum(); node_id++)
		{
			// Initialize the final nodal position.
			Vec3<double> final_pos(0., 0., 0.);

			// Add the values of the displaced nodal positions * the shape function value for the neighbor nodes of the nodal node.
			for (Eigen::SparseMatrix<double>::InnerIterator sf(nodal_approximant.ShapeFunction(), node_id); sf; ++sf) {
				final_pos.SetX(final_pos.X() + saved_disp_backup.coeff(sf.row(), 0)*sf.value());
				final_pos.SetY(final_pos.Y() + saved_disp_backup.coeff(sf.row(), 1)*sf.value());
				final_pos.SetZ(final_pos.Z() + saved_disp_backup.coeff(sf.row(), 2)*sf.value());
			}

			// Update the nodal values of the saved displacements with the final nodal positions.
			this->saved_disps_[disp_id].coeffRef(node_id, 0) = final_pos.X();
			this->saved_disps_[disp_id].coeffRef(node_id, 1) = final_pos.Y();
			this->saved_disps_[disp_id].coeffRef(node_id, 2) = final_pos.Z();
		}
    } // End Iterate over saved displacements.
}

void ComputeForcesThread(Mtled *pMtled, const Mtled::tstThreadControl *pstThreadControl)
{
	if (pstThreadControl->boDoesWork)
	{

		Eigen::MatrixXd forces_thread = Eigen::MatrixXd::Zero(pMtled->pweak_model_3d_->TetrahedralMesh().NodesNum(), 3);

		// Iterate over integration points for force generation.
		for (size_t idxIP = pstThreadControl->uiFirstIdx; idxIP < pstThreadControl->uiLastIdx; idxIP++) 
		{

			double int_point_weight = pMtled->pweak_model_3d_->IntegrationPoints().Weights().at(idxIP);

			// Initialize deformation gradient and 2nd Piola-Kirchhoff stress tensor.
			Eigen::Matrix3d FT = Eigen::Matrix3d::Zero(3, 3);
			Eigen::Matrix3d spk_stress = Eigen::Matrix3d::Zero(3, 3);

			// The integration point's index.
			auto int_point_id = idxIP;

			// Local displacements and forces at integration point's support domain.
			Eigen::MatrixXd disp_local = Eigen::MatrixXd::Zero((*(pMtled->pneighbor_ids_))[int_point_id].size(), 3);
			Eigen::MatrixXd forces_local = Eigen::MatrixXd::Zero((*(pMtled->pneighbor_ids_))[int_point_id].size(), 3);

			// Iterate over neighbor nodes indices.
			for (auto &neigh_id : (*(pMtled->pneighbor_ids_))[int_point_id]) {
				// The neighbors index in the container.
				auto id = &neigh_id - &(*(pMtled->pneighbor_ids_))[int_point_id][0];

				// Populate disp_local.
				disp_local.row(id) = (*(pMtled->pdisplacements_)).row(neigh_id);

			}

			// Compute deformation gradient.
			FT.noalias() = (*(pMtled->pderiv_mats_))[int_point_id].transpose() * disp_local;
			// Add identity matrix contribution to diagonal elements of the deformation gradient tensor.
			FT.coeffRef(0, 0) += 1.; FT.coeffRef(1, 1) += 1.; FT.coeffRef(2, 2) += 1.;

			// Compute the 2nd Piola-Kirchhoff stress tensor.
			spk_stress.noalias() = std::move((*(pMtled->pmaterial_)).SpkStress(FT, int_point_id));

			// Compute the force contribution of the current integration point.
			forces_local = (*(pMtled->pderiv_mats_))[int_point_id] * spk_stress.transpose() * FT * int_point_weight;

			// Update total force.
			for (auto &neigh_id : (*(pMtled->pneighbor_ids_))[int_point_id]) {
				// The neighbors index in the container.
				auto id = &neigh_id - &(*(pMtled->pneighbor_ids_))[int_point_id][0];

				// Add integration point's contribution in force matrix.
				forces_thread.row(neigh_id) += forces_local.row(id);
			}

		} // End iteration over integration points.

		// this is gravity_forces gather information - needs access control between threads
		pMtled->mutex.lock();
		*(pMtled->pforces_) += forces_thread;
		pMtled->mutex.unlock();
	}
}

void Mtled::ComputeForces(void)
{

	boost::thread_group ThreadGroup;
	for (unsigned int t = 0; t < uiNumberOfThreads; t++)
	{
		tstThreadControl *pstThreadControl = &astThreadControl[t];
		//ComputeForcesThread(this, weak_model_3d, neighbor_ids, deriv_mats, material, displacements, forces, pstThreadControl);
		ThreadGroup.create_thread(boost::bind(ComputeForcesThread, this, pstThreadControl));
	}
	ThreadGroup.join_all();
}

} //end of namespace ExplicitSim
