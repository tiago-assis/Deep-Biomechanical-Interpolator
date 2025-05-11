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



#ifndef EXPLICITSIM_SOLVERS_MTLED_HPP_
#define EXPLICITSIM_SOLVERS_MTLED_HPP_

/*!
   \file mtled.hpp
   \brief Mtled class header file.
   \author Konstantinos A. Mountris
   \date 27/09/2017
*/


#include "ExplicitSim/models/models.hpp"
#include "ExplicitSim/approximants/mmls_3d.hpp"
#include "ExplicitSim/integration/integration.hpp"
#include "ExplicitSim/materials/material.hpp"
#include "ExplicitSim/solvers/dyn_relax_prop.hpp"
#include "ExplicitSim/conditions/conditions_handler.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <boost/thread/mutex.hpp>

#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <utility>

#include <stdexcept>
#include <exception>

#include <thread>
#include<mutex>


namespace ExplicitSim {

/*!
 *  \addtogroup Solvers
 *  @{
 */


/*!
 * \class Mtled
 * \brief Class implemmenting the Meshfree Total Lagrangian Explicit Dynamics (MTLED) pde solver.
 *
 */

class Mtled {
public:
    /*!
     * \brief Mtled constructor.
     */
    Mtled();


    /*!
     * \brief Mtled destructor.
     */
    virtual ~Mtled();

	// Structure with thread control variables
	typedef struct stThreadControl {
		bool boDoesWork;
		size_t uiFirstIdx, uiLastIdx;
	} tstThreadControl;

    /*!
     * \brief Computes the time steps for each evaluation point and sets the minimum and maximum time step.
     *
     * \param [in] wave_speed The wave_speed of the evaluation points.
     * \param [in] neighbors_ids The indices of the neighbor points to the evaluation points.
     * \param [in] model_approximant The approximant of the shape function and derivatives on the model's integration points.
     * \return [void]
     */
    void ComputeTimeSteps(const std::vector<double> &wave_speed, const std::vector< std::vector<uint32_t> > &neighbors_ids,
                          const Mmls3d &model_approximant);


    /*!
     * \brief Set the stable step.
     * \param [in] stable_step The stable step to be setted.
     * \return [void]
     */
    inline void SetStableStep(const double &stable_step) { this->stable_step_ = stable_step; }


    /*!
     * \brief Compute the stable step according to the model's mass scaling.
     * \param [in] is_mass_scaled The conditional declaring if the models mass is scaled.
     * \param [in] safety_factor The division factor to ensure stability by decreasing the stable time step. [Default: 1.]
     * \return [void]
     */
    void ComputeStableStep(const bool &is_mass_scaled, double safety_factor=1.);


    /*!
     * \brief Compute the total number of time steps for the explicit solution.
     *
     * The total number of time steps is the summation of the number of steps for loading application
     * and the number of steps for dynamic relaxation equilibrium achievement.
     *
     * \param [in] load_steps_num The number of time steps for the loading application.
     * \param [in] equilibrium_steps_num The number of time steps for achieving dynamic relaxation equilibrium.
     * \return [void]
     */
    void ComputeTotalTimeStepsNum(const int &load_steps_num, const int &equilibrium_steps_num);


    /*!
     * \brief Set the number of steps for saving progress.
     * \param [in] save_progress_steps The number of steps for saving progress.
     */
    inline void SetSaveProgressSteps(const unsigned int &save_progress_steps) { this->save_progress_steps_ = save_progress_steps; }


    /*!
     * \brief Solve the displacement & forces fields explicitly using the MTLED with dynamic relaxation.
     *
     * Provides iterative solution of the model and calculation of displacements and
     * forces using the MTLED and applying dynamic relaxation for fast convergence.
     *
     * \param [in] weak_model_3d The weak formulation 3D model to be solved.
     * \param [in] neighbor_ids The indices of neighbor nodes (support domain) to each node of the 3D model.
     * \param [in] cond_handler The handler of conditions imposition.
     * \param [in] model_approximant The approximant of the shape function and derivatives on the model's integration points.
     * \param [in] material The assigned material to the 3D model.
     * \param [in] dyn_relax_prop The dynamic relaxation properties to be used by the MTLED.
     * \return [void]
     * \note Applying loading conditions at the first timestep looks not necessary and has been commented out for now.
     */
    void Solve(WeakModel3D &weak_model_3d, std::vector<std::vector<uint32_t> > &neighbor_ids, ConditionsHandler &cond_handler,
               const Mmls3d &model_approximant, Material &material, const DynRelaxProp &dyn_relax_prop, const bool &use_ebciem, const Mmls3d &nodal_approximant);


    /*!
     * \brief ApplyShapeFuncToDisplacements
     * \return [void]
     */
    void ApplyShapeFuncToDisplacements(const WeakModel3D &weak_model_3d, const Mmls3d &nodal_approximant);
	
    /*!
     * \brief Get the time steps of the evaluation points.
     * \return [std::vector<double>] The time steps of the evaluation points.
     */
    inline const std::vector<double> & TimeSteps() const { return this->time_steps_; }


    /*!
     * \brief Get the minimum step.
     * \return [double] The minimum step.
     */
    inline const double & MinStep() const { return this->min_step_; }


    /*!
     * \brief Get the maximum step.
     * \return [double] The maximum step.
     */
    inline const double & MaxStep() const { return this->max_step_; }


    /*!
     * \brief Get the stable step.
     * \return [double] The stable step.
     */
    inline const double & StableStep() const { return this->stable_step_; }


    /*!
     * \brief Get the total number of time steps for the explicit solution.
     * \return [int] The total number of time steps for the explicit solution.
     */
    inline const int & TotalTimeStepsNum() const { return this->total_time_steps_num; }


    /*!
     * \brief Get the saved displacements at the pre-defined step intervals.
     * \return [std::vector<Eigen::MatrixXd>] The saved displacements at the pre-defined step intervals.
     */
    inline const std::vector<Eigen::MatrixXd> & SavedDisplacements() const { return this->saved_disps_; }


    /*!
     * \brief Get the saved forces at the pre-defined step intervals.
     * \return [std::vector<Eigen::MatrixXd>] The saved forces at the pre-defined step intervals.
     */
    inline const std::vector<Eigen::MatrixXd> & SavedForces() const { return this->saved_forces_; }

	inline const std::vector<Eigen::MatrixXd> & SavedExternalForces() const { return this->saved_external_forces_; }

	/*!
	* \brief Get the saved times.
	* \return [std::vector<float>] The saved times.
	*/
	inline const std::vector<float> & SavedTimes() const { return this->saved_times_; }

	friend void ComputeForcesThread(Mtled *pMtled, const tstThreadControl *astThreadControl);

protected:

    /*!
     * \brief Compute the acting forces on the nodes of a weak formulation 3D model.
     * \param [in] weak_model_3d The weak formulation 3D model.
     * \param [in] neighbor_ids The list of neighbor nodes' indices to the model's integration points.
     * \param [in] deriv_mats The list of first derivatives (x, y, z) matrices for the model's integration points.
     * \param [in] material The material of the model.
     * \param [in] displacements The displacements of the model's nodes.
     * \param [out] forces The computed acting forces on the model's nodes.
     */
    void ComputeForces(void);


	


private:
    std::vector<double> time_steps_;                    /*!< The container of time steps for each evaluation point. */

    double min_step_;                                   /*!< The minimum available time step to use in explicit solution. */

    double max_step_;                                   /*!< The maximum available time step to use in explicit solution. */

    double stable_step_;                                /*!< The time step that can ensure stable explicit solution. */

    int total_time_steps_num;                           /*!< The number of the total time steps of the explicit solution. */

    unsigned int save_progress_steps_;                           /*!< The number of steps after which the solution progress is saved. */

    std::vector<Eigen::MatrixXd> saved_disps_;          /*!< The saved displacements at pre-defined step intervals. */

    std::vector<Eigen::MatrixXd> saved_forces_;         /*!< The saved forces at pre-defined step intervals. */

	std::vector<Eigen::MatrixXd> saved_external_forces_;  /*!< The saved external forces at pre-defined step intervals. */

	std::vector<float> saved_times_;						/*!< The saved time. */

    LoadCurveSmooth LC;


	WeakModel3D *pweak_model_3d_;
	std::vector<std::vector<uint32_t> > *pneighbor_ids_;
	std::vector<Eigen::MatrixXd> *pderiv_mats_;
	Material *pmaterial_;
	Eigen::MatrixXd *pdisplacements_;
	Eigen::MatrixXd *pforces_;

	// thread synchronisation and control
	tstThreadControl *astThreadControl;
	boost::mutex mutex;
	unsigned int uiNumberOfThreads;

};


/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif //EXPLICITSIM_SOLVERS_MTLED_HPP_
